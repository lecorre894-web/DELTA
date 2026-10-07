#!/usr/bin/env python3
"""DELTA QPU EMULE V22 — BLOCAGE DE CACHE : plusieurs portes par passage memoire (RTX 4080).
v21 est au plafond VRAM (~715 Go/s) car CHAQUE porte relit et reecrit tout le vecteur d'etat.
v22 : on regroupe des portes consecutives qui ne touchent qu'un ensemble S de B qubits (S contient toujours
les qubits 0,1,2 pour des acces de 128 o contigus). Chaque bloc CUDA charge 2^B amplitudes en memoire partagee,
y applique TOUTES les portes du groupe, puis reecrit : 1 passage memoire pour ~10 portes au lieu de 10.
Validation : emulation NumPy du decoupage vs Qiskit (n=12), puis GPU v22 vs GPU v21 (n=16, 20)."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_arena_v20 import circuit, qiskit_sv
from delta_qpu_gpu_v21 import compile_ops, fid, GPU
if GPU:
    import cupy as cp
    from delta_qpu_gpu_v21 import GPUSim
SRC=r'''
extern "C" __global__ void blk(double2 *psi, int n, int B, const int *Sg, const int *Cg, int ng, const double *G){
  extern __shared__ double2 sh[];
  __shared__ int S[16], C[48];
  if(threadIdx.x<B) S[threadIdx.x]=Sg[threadIdx.x];
  if(threadIdx.x<n-B) C[threadIdx.x]=Cg[threadIdx.x];
  __syncthreads();
  long long base=blockIdx.x, boff=0;
  for(int k=0;k<n-B;k++) if((base>>k)&1LL) boff|=1LL<<C[k];
  int M=1<<B;
  for(int j=threadIdx.x;j<M;j+=blockDim.x){ long long off=boff; for(int k=0;k<B;k++) if((j>>k)&1) off|=1LL<<S[k]; sh[j]=psi[off]; }
  __syncthreads();
  for(int g=0;g<ng;g++){
    const double *d=G+11*g; int ty=(int)d[0], a=(int)d[1], b=(int)d[2];
    if(ty==0){
      for(int i=threadIdx.x;i<(M>>1);i+=blockDim.x){
        int i0=((i>>a)<<(a+1))|(i&((1<<a)-1)), i1=i0|(1<<a); double2 x=sh[i0], y=sh[i1], r0, r1;
        r0.x=d[3]*x.x-d[4]*x.y+d[5]*y.x-d[6]*y.y; r0.y=d[3]*x.y+d[4]*x.x+d[5]*y.y+d[6]*y.x;
        r1.x=d[7]*x.x-d[8]*x.y+d[9]*y.x-d[10]*y.y; r1.y=d[7]*x.y+d[8]*x.x+d[9]*y.y+d[10]*y.x;
        sh[i0]=r0; sh[i1]=r1; }
    } else {
      int lo=a<b?a:b, hi=a<b?b:a;
      for(int i=threadIdx.x;i<(M>>2);i+=blockDim.x){
        int x=((i>>lo)<<(lo+1))|(i&((1<<lo)-1)); x=((x>>hi)<<(hi+1))|(x&((1<<hi)-1));
        int i0=x|(1<<a), i1=i0|(1<<b); double2 t=sh[i0]; sh[i0]=sh[i1]; sh[i1]=t; }
    }
    __syncthreads();
  }
  for(int j=threadIdx.x;j<M;j+=blockDim.x){ long long off=boff; for(int k=0;k<B;k++) if((j>>k)&1) off|=1LL<<S[k]; psi[off]=sh[j]; }
}
'''
FIXED=(0,1,2)
def schedule(n,prog,B):
    B=min(B,n); passes=[]; cur=set(FIXED[:B]); gs=[]
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
            close(); cur=set(FIXED[:B])|qs; gs=[g]
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
    def __init__(s): s.k=cp.RawModule(code=SRC).get_function("blk")
    def upload(s,n,passes):
        out=[]
        for S,loc in passes:
            C=[q for q in range(n) if q not in S]; G=np.zeros((max(1,len(loc)),11))
            for r,g in enumerate(loc):
                if g[0]=="u": M=g[2]; G[r]=[0,g[1],0,M[0,0].real,M[0,0].imag,M[0,1].real,M[0,1].imag,M[1,0].real,M[1,0].imag,M[1,1].real,M[1,1].imag]
                else: G[r,:3]=[1,g[1],g[2]]
            out.append((len(S),cp.asarray(np.array(S,np.int32)),cp.asarray(np.array(C or [0],np.int32)),len(loc),cp.asarray(G)))
        return out
    def run(s,n,dev):
        psi=cp.zeros(1<<n,dtype=cp.complex128); psi[0]=1
        for B,S,C,ng,G in dev:
            s.k((1<<(n-B),),(256,),(psi,np.int32(n),np.int32(B),S,C,np.int32(ng),G),shared_mem=(1<<B)*16)
        return psi
def main():
    L=[]; P=lambda s:(print(s,flush=True),L.append(s))
    P("="*100); P("DELTA QPU EMULE V22 | blocage de cache : plusieurs portes par passage memoire | RTX 4080"); P("="*100)
    ops=circuit(12,12,5); prog=compile_ops(12,ops); ref=qiskit_sv(12,ops)
    for B in (6,8):
        ps=schedule(12,prog,B); f=fid(ref,run_numpy_blocked(12,ps))
        P("emulation NumPy n=12 B=%d : %d ops -> %d passages  fidelite vs Qiskit=%.12f %s"%(B,len(prog),len(ps),f,"PASS" if f>1-1e-9 else "FAIL"))
    if not GPU: P("CuPy absent : arret apres validation logique."); return
    bs=BlockSim(); v21=GPUSim()
    for n in (16,20):
        pr=compile_ops(n,circuit(n,20,n)); a=cp.asnumpy(v21.run(n,pr)); b=cp.asnumpy(bs.run(n,bs.upload(n,schedule(n,pr,10))))
        f=fid(a,b); P("validation GPU v22 (B=10) vs GPU v21 n=%d : fidelite=%.12f %s"%(n,f,"PASS" if f>1-1e-9 else "FAIL"))
    V21={20:12.6,22:53.8,24:526.2,26:2264.3,28:9750.8,29:20173.4}
    free=cp.cuda.runtime.memGetInfo()[0]
    for n in (20,22,24,26,28,29):
        if (1<<n)*16>free*0.85: P("n=%-2d : VRAM insuffisante, saute"%n); continue
        pr=compile_ops(n,circuit(n,20,n))
        for B in (10,11):
            ps=schedule(n,pr,B); dev=bs.upload(n,ps); bs.run(min(n,14),bs.upload(min(n,14),schedule(min(n,14),compile_ops(min(n,14),circuit(min(n,14),2,1)),min(B,14)))); cp.cuda.Device().synchronize()
            e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(); psi=bs.run(n,dev); e1.record(); e1.synchronize()
            ms=cp.cuda.get_elapsed_time(e0,e1); norm=float(cp.linalg.norm(psi)); del psi; cp.get_default_memory_pool().free_all_blocks()
            P("n=%-2d B=%-2d : %3d ops -> %3d passages (%.1f portes/passage) : %9.1f ms  norme=%.12f  ~%.0f Go/s  vs v21 x%.2f"%(n,B,len(pr),len(ps),len(pr)/len(ps),ms,norm,len(ps)*(1<<n)*32/1e9/(ms/1e3),V21[n]/ms))
    P("="*100)
    f=os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_gpu_v22_results.txt"); open(f,"w").write("\n".join(L)+"\n"); print("JOURNAL="+f)
if __name__=="__main__": main()
