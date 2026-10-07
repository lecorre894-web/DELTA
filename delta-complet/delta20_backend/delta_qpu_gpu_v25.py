#!/usr/bin/env python3
"""DELTA QPU EMULE V25 — synthese de toutes les pistes (RTX 4080).
(1) complex64 (v24) : calcul FP32 ~64x plus rapide que FP64 sur 4080, 30 qubits en 8 Go.
(2) blocage de cache (v22) : B qubits en memoire partagee par bloc CUDA, L qubits bas fixes (paquets contigus).
(3) NOUVEAU ordonnanceur DAG : des portes sur qubits disjoints commutent -> un passage avance sur PLUSIEURS COUCHES
    tant que les portes restent dans S ; 29 q : 63 passages (v24) -> ~16. Or v24 est au plafond VRAM : temps ~ passages.
(4) NOUVEAU fusion 2 qubits dans chaque passage : u(a), u(b), cx(a,b), u... -> une seule matrice 4x4 (moins d'etapes/synchros).
Validation : emulation NumPy vs Qiskit (n=12) ; GPU v25 FP32 vs GPU v21 FP64 (n=16, 20, 24)."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_arena_v20 import circuit, qiskit_sv
from delta_qpu_gpu_v21 import compile_ops, fid, GPU
from delta_qpu_gpu_v24 import schedule as schedule_v24
if GPU:
    import cupy as cp
    from delta_qpu_gpu_v21 import GPUSim
RW=35
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
    const float *d=G+35*g; int ty=(int)d[0], a=(int)d[1], b=(int)d[2];
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
        if(ty==1){ int i0=x|(1<<a), i1=i0|(1<<b); float2 t=sh[i0]; sh[i0]=sh[i1]; sh[i1]=t; }
        else {
          int id[4]={x, x|(1<<a), x|(1<<b), x|(1<<a)|(1<<b)}; float2 v[4], r;
          #pragma unroll
          for(int k=0;k<4;k++) v[k]=sh[id[k]];
          #pragma unroll
          for(int r_=0;r_<4;r_++){ const float *m=d+3+8*r_; r.x=0; r.y=0;
            #pragma unroll
            for(int k=0;k<4;k++){ r.x+=m[2*k]*v[k].x-m[2*k+1]*v[k].y; r.y+=m[2*k]*v[k].y+m[2*k+1]*v[k].x; }
            sh[id[r_]]=r; }
        }
      }
    }
    __syncthreads();
  }
  for(int j=threadIdx.x;j<M;j+=blockDim.x) psi[boff|OFF[j]]=sh[j];
}
'''
CX4=np.eye(4,dtype=np.complex128)[[0,3,2,1]]  # base k = bit_a + 2*bit_b ; cx(a=controle,b=cible)
I2=np.eye(2)
def gq(g): return (g[1],) if g[0]=="u" else (g[1],g[2])
def fuse(gs):
    """fusion 2 qubits d'une liste ordonnee (indices globaux) : ('u',q,M2) / ('u2',a,b,M4)"""
    out=[]; last={}
    for g in gs:
        if g[0]=="u":
            q,M=g[1],g[2]; j=last.get(q)
            if j is not None and out[j] is not None:
                F=out[j]
                if F[0]=="u": out[j]=("u",q,M@F[2])
                else: out[j]=("u2",F[1],F[2],(np.kron(I2,M) if q==F[1] else np.kron(M,I2))@F[3])
            else: out.append(("u",q,M)); last[q]=len(out)-1
        else:
            a,b=g[1],g[2]; ja,jb=last.get(a),last.get(b)
            if ja is not None and ja==jb:
                F=out[ja]; P=CX4 if (F[1],F[2])==(a,b) else CX4[np.ix_([0,2,1,3],[0,2,1,3])]
                out[ja]=("u2",F[1],F[2],P@F[3]); continue
            M=np.eye(4,dtype=np.complex128)
            for q,jq,emb in ((a,ja,lambda m:np.kron(I2,m)),(b,jb,lambda m:np.kron(m,I2))):
                if jq is not None and out[jq][0]=="u": M=emb(out[jq][2])@M; out[jq]=None
            out.append(("u2",a,b,CX4@M)); last[a]=last[b]=len(out)-1
    return [g for g in out if g is not None]
def schedule_dag(n,prog,B,L=3,fz=True):
    B=min(B,n); FIXED=set(range(min(L,B))); Q=[[] for _ in range(n)]
    for i,g in enumerate(prog):
        for q in gq(g): Q[q].append(i)
    ptr=[0]*n; left=len(prog); passes=[]
    def ready(i): return all(Q[q][ptr[q]]==i for q in gq(prog[i]))
    while left:
        S=set(FIXED); gs=[]
        while True:
            moved=True
            while moved:
                moved=False
                for q in sorted(S):
                    while ptr[q]<len(Q[q]):
                        i=Q[q][ptr[q]]
                        if set(gq(prog[i]))<=S and ready(i):
                            gs.append(prog[i]); left-=1; moved=True
                            for r in gq(prog[i]): ptr[r]+=1
                        else: break
            if len(S)>=B: break
            best=None
            for q in range(n):
                if ptr[q]<len(Q[q]):
                    i=Q[q][ptr[q]]
                    if ready(i):
                        new=set(gq(prog[i]))-S
                        if new and len(S)+len(new)<=B and (best is None or (len(new),i)<best[0]): best=((len(new),i),new)
            if best is None: break
            S|=best[1]
        if not gs: raise RuntimeError("blocage ordonnanceur")
        k=0
        while len(S)<B:
            if k not in S: S.add(k)
            k+=1
        S=sorted(S); pos={q:j for j,q in enumerate(S)}; loc=[]
        for g in (fuse(gs) if fz else gs):
            if g[0]=="u": loc.append(("u",pos[g[1]],g[2]))
            elif g[0]=="cx": loc.append(("cx",pos[g[1]],pos[g[2]]))
            else: loc.append(("u2",pos[g[1]],pos[g[2]],g[3]))
        passes.append((S,loc))
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
                x=((i>>lo)<<(lo+1))|(i&((1<<lo)-1)); x=((x>>hi)<<(hi+1))|(x&((1<<hi)-1))
                if g[0]=="cx": i0=x|(1<<a); i1=i0|(1<<b); t=sh[:,i0].copy(); sh[:,i0]=sh[:,i1]; sh[:,i1]=t
                else:
                    ids=[x,x|(1<<a),x|(1<<b),x|(1<<a)|(1<<b)]; v=np.stack([sh[:,k] for k in ids],0); r=np.tensordot(g[3],v,1)
                    for k in range(4): sh[:,ids[k]]=r[k]
        psi[idx]=sh
    return psi
class BlockSim:
    def __init__(s):
        s.k=cp.RawModule(code=SRC).get_function("blk")
        try: s.k.max_dynamic_shared_size_bytes=65536
        except Exception as e: print("note: opt-in 64 Ko indisponible (%s)"%e)
    def upload(s,n,passes):
        out=[]
        for S,loc in passes:
            C=[q for q in range(n) if q not in S]; G=np.zeros((max(1,len(loc)),RW))
            for r,g in enumerate(loc):
                if g[0]=="u": M=g[2]; G[r,:11]=[0,g[1],0,M[0,0].real,M[0,0].imag,M[0,1].real,M[0,1].imag,M[1,0].real,M[1,0].imag,M[1,1].real,M[1,1].imag]
                elif g[0]=="cx": G[r,:3]=[1,g[1],g[2]]
                else: G[r,:3]=[2,g[1],g[2]]; G[r,3:]=np.stack([g[3].real,g[3].imag],-1).reshape(-1)
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
    P("="*100); P("DELTA QPU EMULE V25 | complex64 + blocage + ordonnanceur DAG multi-couches + fusion 2 qubits | RTX 4080"); P("="*100)
    ops=circuit(12,12,5); prog=compile_ops(12,ops); ref=qiskit_sv(12,ops)
    for B,Lb,fz in ((8,3,False),(8,3,True),(10,2,True)):
        ps=schedule_dag(12,prog,B,Lb,fz); f=fid(ref,run_numpy_blocked(12,ps))
        P("emulation NumPy n=12 B=%d L=%d fusion=%d : %d ops -> %d passages, %d etapes  fidelite vs Qiskit=%.12f %s"%(B,Lb,fz,len(prog),len(ps),sum(len(l) for _,l in ps),f,"PASS" if f>1-1e-9 else "FAIL"))
    if not GPU: P("CuPy absent : arret apres validation logique."); return
    bs=BlockSim(); v21=GPUSim()
    for n in (16,20,24):
        pr=compile_ops(n,circuit(n,20,n)); a=cp.asnumpy(v21.run(n,pr)).astype(np.complex128)
        for Lb,fz in ((3,False),(2,True)):
            b=cp.asnumpy(bs.run(n,bs.upload(n,schedule_dag(n,pr,12,Lb,fz)))).astype(np.complex128); f=fid(a,b)
            P("validation GPU v25 FP32 (L=%d fusion=%d) vs GPU v21 FP64 n=%d : infidelite=%.2e  %s"%(Lb,fz,n,1-f,"PASS" if 1-f<1e-4 else "FAIL"))
        del a,b; cp.get_default_memory_pool().free_all_blocks()
    V24={20:1.1,22:4.3,24:21.8,26:100.6,28:433.9,29:924.1,30:1927.0}; V22={20:10.8,22:42.8,24:171.7,26:738.7,28:3171.2,29:6555.4}
    free=cp.cuda.runtime.memGetInfo()[0]; best={}
    for n in (20,22,24,26,28,29,30):
        if (1<<n)*8>free*0.85: P("n=%-2d : VRAM insuffisante, saute"%n); continue
        pr=compile_ops(n,circuit(n,20,n))
        for B,Lb,fz in ((11,3,True),(12,2,False),(12,2,True),(12,3,True),(13,3,True)):
            try:
                ps=schedule_dag(n,pr,B,Lb,fz); dev=bs.upload(n,ps); bs.run(n,bs.upload(n,schedule_dag(n,compile_ops(n,circuit(n,1,1)),B,Lb,fz))); cp.cuda.Device().synchronize()
                ms=1e30
                for _ in range(2):
                    e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(); psi=bs.run(n,dev); e1.record(); e1.synchronize(); ms=min(ms,cp.cuda.get_elapsed_time(e0,e1)); 
                    if _==0: del psi
                x=psi.view(cp.float32); K=1<<26; norm=math.sqrt(sum(float(x[i:i+K].dot(x[i:i+K])) for i in range(0,x.size,K))); del x,psi; cp.get_default_memory_pool().free_all_blocks()
                best[n]=min(best.get(n,1e30),ms)
                P("n=%-2d B=%-2d L=%d fus=%d : %3d ops -> %3d passages (v24 %3d), %4d etapes : %8.1f ms  |norme-1|=%.1e  ~%4.0f Go/s  vs v24 x%4.2f  vs v22 FP64 %s"%(n,B,Lb,fz,len(pr),len(ps),len(schedule_v24(n,pr,B,3)),sum(len(l) for _,l in ps),ms,abs(norm-1),len(ps)*(1<<n)*16/1e9/(ms/1e3),V24[n]/ms,("x%.0f"%(V22[n]/ms)) if n in V22 else "-"))
            except Exception as e:
                P("n=%-2d B=%-2d L=%d fus=%d : ECHEC %s"%(n,B,Lb,fz,str(e)[:80])); cp.get_default_memory_pool().free_all_blocks()
    P("-"*100); P("MEILLEUR V25 : "+"  ".join("%dq %.1f ms (x%.2f v24)"%(n,best[n],V24[n]/best[n]) for n in sorted(best)))
    P("="*100)
    f=os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_gpu_v25_results.txt"); open(f,"w").write("\n".join(L)+"\n"); print("JOURNAL="+f)
if __name__=="__main__": main()
