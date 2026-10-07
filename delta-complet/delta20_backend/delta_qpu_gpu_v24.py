#!/usr/bin/env python3
"""DELTA QPU EMULE V24 — v22/v23 passent en SIMPLE PRECISION (complex64) sur RTX 4080.
Diagnostic v23 : le temps ne depend plus du nombre de passages -> le goulot est le CALCUL.
Or la RTX 4080 calcule le double (FP64) ~64x moins vite que le simple (FP32) :
29 q x 841 portes x ~14 flops/amplitude ~ 6e12 flops FP64 -> ~8 s au plafond FP64 : v22 (6,6 s) y est deja.
v24 : vecteur d'etat en complex64 (comme Qiskit Aer precision='single') -> calcul ~64x plus rapide, memoire / 2,
30 qubits possibles (8 Go). Prix : precision ~1e-6 au lieu de ~1e-15 ; mesuree ici (1 - fidelite vs v21 FP64).
Validation : emulation NumPy vs Qiskit (n=12) ; GPU v24 FP32 vs GPU v21 FP64 (n=16, 20, 24) en infidelite."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_arena_v20 import circuit, qiskit_sv
from delta_qpu_gpu_v21 import compile_ops, fid, GPU
if GPU:
    import cupy as cp
    from delta_qpu_gpu_v21 import GPUSim
SRC=r'''
extern "C" __global__ void blk(float2 *psi, int n, int B, const long long *OFF, const int *Cg, int ng, const float *G){
  extern __shared__ float2 sh[];
  __shared__ int C[48]; __shared__ long long boff_s;
  if(threadIdx.x<n-B) C[threadIdx.x]=Cg[threadIdx.x];
  __syncthreads();
  if(threadIdx.x==0){ long long base=blockIdx.x, b=0; for(int k=0;k<n-B;k++) if((base>>k)&1LL) b|=1LL<<C[k]; boff_s=b; }
  __syncthreads();
  long long boff=boff_s; int M=1<<B;
  for(int j=threadIdx.x;j<M;j+=blockDim.x) sh[j]=psi[boff|OFF[j]];
  __syncthreads();
  for(int g=0;g<ng;g++){
    const float *d=G+11*g; int ty=(int)d[0], a=(int)d[1], b=(int)d[2];
    if(ty==0){
      for(int i=threadIdx.x;i<(M>>1);i+=blockDim.x){
        int i0=((i>>a)<<(a+1))|(i&((1<<a)-1)), i1=i0|(1<<a); float2 x=sh[i0], y=sh[i1], r0, r1;
        r0.x=d[3]*x.x-d[4]*x.y+d[5]*y.x-d[6]*y.y; r0.y=d[3]*x.y+d[4]*x.x+d[5]*y.y+d[6]*y.x;
        r1.x=d[7]*x.x-d[8]*x.y+d[9]*y.x-d[10]*y.y; r1.y=d[7]*x.y+d[8]*x.x+d[9]*y.y+d[10]*y.x;
        sh[i0]=r0; sh[i1]=r1; }
    } else {
      int lo=a<b?a:b, hi=a<b?b:a;
      for(int i=threadIdx.x;i<(M>>2);i+=blockDim.x){
        int x=((i>>lo)<<(lo+1))|(i&((1<<lo)-1)); x=((x>>hi)<<(hi+1))|(x&((1<<hi)-1));
        int i0=x|(1<<a), i1=i0|(1<<b); float2 t=sh[i0]; sh[i0]=sh[i1]; sh[i1]=t; }
    }
    __syncthreads();
  }
  for(int j=threadIdx.x;j<M;j+=blockDim.x) psi[boff|OFF[j]]=sh[j];
}
'''
def schedule(n,prog,B,L=3):
    """decoupe gloutonne en passages : chaque passage = (S trie, [portes en indices locaux])"""
    B=min(B,n); FIXED=tuple(range(min(L,B))); passes=[]; cur=set(FIXED); gs=[]
    def close():
        S=set(cur); k=0
        while len(S)<B:
            if k not in S: S.add(k)
            k+=1
        S=sorted(S); pos={q:i for i,q in enumerate(S)}; loc=[]
        for g in gs: loc.append(("u",pos[g[1]],g[2]) if g[0]=="u" else ("cx",pos[g[1]],pos[g[2]]))
        passes.append((S,loc))
    for g in prog:
        qs={g[1]} if g[0]=="u" else {g[1],g[2]}
        if len(cur|qs)<=B: cur|=qs; gs.append(g)
        else:
            close(); cur=set(FIXED)|qs; gs=[g]
    if gs: close()
    return passes
def run_numpy_blocked(n,passes):
    N=1<<n; psi=np.zeros(N,np.complex128); psi[0]=1
    for S,loc in passes:
        B=len(S); C=[q for q in range(n) if q not in S]
        bases=np.arange(1<<(n-B),dtype=np.int64); boff=np.zeros_like(bases)
        for k,q in enumerate(C): boff|=((bases>>k)&1)<<q
        j=np.arange(1<<B,dtype=np.int64); off=np.zeros_like(j)
        for k,q in enumerate(S): off|=((j>>k)&1)<<q
        idx=boff[:,None]|off[None,:]; sh=psi[idx]
        for g in loc:
            if g[0]=="u":
                a,M=g[1],g[2]; i=np.arange((1<<B)//2); i0=((i>>a)<<(a+1))|(i&((1<<a)-1)); i1=i0|(1<<a)
                x,y=sh[:,i0].copy(),sh[:,i1].copy(); sh[:,i0]=M[0,0]*x+M[0,1]*y; sh[:,i1]=M[1,0]*x+M[1,1]*y
            else:
                a,b=g[1],g[2]; lo,hi=min(a,b),max(a,b); i=np.arange((1<<B)//4)
                x=((i>>lo)<<(lo+1))|(i&((1<<lo)-1)); x=((x>>hi)<<(hi+1))|(x&((1<<hi)-1)); i0=x|(1<<a); i1=i0|(1<<b)
                t=sh[:,i0].copy(); sh[:,i0]=sh[:,i1]; sh[:,i1]=t
        psi[idx]=sh
    return psi
class BlockSim:
    def __init__(s):
        s.k=cp.RawModule(code=SRC).get_function("blk")
        try: s.k.max_dynamic_shared_size_bytes=65536  # B=13 en FP32 = 64 Ko
        except Exception as e: print("note: opt-in 64 Ko indisponible (%s), B=12 peut echouer"%e)
    def upload(s,n,passes):
        out=[]
        for S,loc in passes:
            C=[q for q in range(n) if q not in S]; G=np.zeros((max(1,len(loc)),11))
            for r,g in enumerate(loc):
                if g[0]=="u": M=g[2]; G[r]=[0,g[1],0,M[0,0].real,M[0,0].imag,M[0,1].real,M[0,1].imag,M[1,0].real,M[1,0].imag,M[1,1].real,M[1,1].imag]
                else: G[r,:3]=[1,g[1],g[2]]
            j=np.arange(1<<len(S),dtype=np.int64); off=np.zeros_like(j)
            for k,q in enumerate(S): off|=((j>>k)&1)<<q
            out.append((len(S),cp.asarray(off),cp.asarray(np.array(C or [0],np.int32)),len(loc),cp.asarray(G.astype(np.float32))))
        return out
    def run(s,n,dev):
        psi=cp.zeros(1<<n,dtype=cp.complex64); psi[0]=1
        for B,OFF,C,ng,G in dev:
            s.k((1<<(n-B),),(512,),(psi,np.int32(n),np.int32(B),OFF,C,np.int32(ng),G),shared_mem=(1<<B)*8)
        return psi
def main():
    L=[]; P=lambda s:(print(s,flush=True),L.append(s))
    P("="*100); P("DELTA QPU EMULE V24 | simple precision complex64 + blocage de cache | RTX 4080"); P("="*100)
    ops=circuit(12,12,5); prog=compile_ops(12,ops); ref=qiskit_sv(12,ops)
    for B,Lb in ((8,4),(10,5)):
        ps=schedule(12,prog,B,Lb); f=fid(ref,run_numpy_blocked(12,ps))
        P("emulation NumPy n=12 B=%d L=%d : %d ops -> %d passages  fidelite vs Qiskit=%.12f %s"%(B,Lb,len(prog),len(ps),f,"PASS" if f>1-1e-9 else "FAIL"))
    if not GPU: P("CuPy absent : arret apres validation logique."); return
    bs=BlockSim(); v21=GPUSim()
    for n in (16,20,24):
        pr=compile_ops(n,circuit(n,20,n)); a=cp.asnumpy(v21.run(n,pr)).astype(np.complex128)
        b=cp.asnumpy(bs.run(n,bs.upload(n,schedule(n,pr,12,3)))).astype(np.complex128); f=fid(a,b)
        P("validation GPU v24 FP32 vs GPU v21 FP64 n=%d : infidelite=%.2e  %s"%(n,1-f,"PASS" if 1-f<1e-4 else "FAIL")); del a,b
        cp.get_default_memory_pool().free_all_blocks()
    V22={20:10.8,22:42.8,24:171.7,26:738.7,28:3171.2,29:6555.4}
    free=cp.cuda.runtime.memGetInfo()[0]
    for n in (20,22,24,26,28,29,30):
        if (1<<n)*8>free*0.85: P("n=%-2d : VRAM insuffisante, saute"%n); continue
        pr=compile_ops(n,circuit(n,20,n))
        for B in (11,12,13):
            try:
                ps=schedule(n,pr,B,3); dev=bs.upload(n,ps); bs.run(n,bs.upload(n,schedule(n,compile_ops(n,circuit(n,1,1)),B,3))); cp.cuda.Device().synchronize()
                e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(); psi=bs.run(n,dev); e1.record(); e1.synchronize()
                ms=cp.cuda.get_elapsed_time(e0,e1); x=psi.view(cp.float32); K=1<<26; norm=math.sqrt(sum(float(x[i:i+K].dot(x[i:i+K])) for i in range(0,x.size,K))); del x,psi; cp.get_default_memory_pool().free_all_blocks()
                P("n=%-2d B=%-2d : %3d ops -> %3d passages (%4.1f p/pass) : %9.1f ms  |norme-1|=%.1e  ~%4.0f Go/s%s"%(n,B,len(pr),len(ps),len(pr)/len(ps),ms,abs(norm-1),len(ps)*(1<<n)*16/1e9/(ms/1e3),("  vs v22 FP64 x%.1f"%(V22[n]/ms)) if n in V22 else "  (inaccessible en FP64)"))
            except Exception as e:
                P("n=%-2d B=%-2d : ECHEC %s"%(n,B,str(e)[:80])); cp.get_default_memory_pool().free_all_blocks()
    P("="*100)
    f=os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_gpu_v24_results.txt"); open(f,"w").write("\n".join(L)+"\n"); print("JOURNAL="+f)
if __name__=="__main__": main()
