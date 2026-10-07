#!/usr/bin/env python3
"""DELTA QPU EMULE V26 — v25 + portes appliquees EN REGISTRES (RTX 4080).
Diagnostic v25 : ~26 etapes/passage, chacune = lecture+ecriture memoire partagee + __syncthreads -> goulot (~200 Go/s).
v26 : 2^(B-4) threads par bloc, chaque thread tient 16 amplitudes en registres = 4 qubits locaux R.
Les portes d'un passage sont groupees : tant qu'elles tiennent dans 4 qubits, elles s'appliquent en registres
(aucune synchro, aucun acces partage) ; une seule lecture/ecriture partagee + 1 synchro par GROUPE.
Positions des qubits en registre = constantes de compilation (templates) -> pas de debordement en memoire locale.
Garde : complex64, blocage B, ordonnanceur DAG multi-couches, fusion 2 qubits (v25)."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_arena_v20 import circuit, qiskit_sv
from delta_qpu_gpu_v21 import compile_ops, fid, GPU
from delta_qpu_gpu_v25 import schedule_dag
if GPU:
    import cupy as cp
    from delta_qpu_gpu_v21 import GPUSim
RW=35; GW=10
SRC=r'''
template<int P> __device__ __forceinline__ void g1(float2 *v,const float *d){
  #pragma unroll
  for(int k=0;k<16;k++) if(!(k&(1<<P))){ int k1=k|(1<<P); float2 x=v[k], y=v[k1], r0, r1;
    r0.x=d[3]*x.x-d[4]*x.y+d[5]*y.x-d[6]*y.y; r0.y=d[3]*x.y+d[4]*x.x+d[5]*y.y+d[6]*y.x;
    r1.x=d[7]*x.x-d[8]*x.y+d[9]*y.x-d[10]*y.y; r1.y=d[7]*x.y+d[8]*x.x+d[9]*y.y+d[10]*y.x; v[k]=r0; v[k1]=r1; }
}
template<int P,int Q> __device__ __forceinline__ void cxr(float2 *v){
  #pragma unroll
  for(int k=0;k<16;k++) if(!(k&(1<<P)) && !(k&(1<<Q))){ int i0=k|(1<<P), i1=i0|(1<<Q); float2 t=v[i0]; v[i0]=v[i1]; v[i1]=t; }
}
template<int P,int Q> __device__ __forceinline__ void g2(float2 *v,const float *d){
  #pragma unroll
  for(int k=0;k<16;k++) if(!(k&(1<<P)) && !(k&(1<<Q))){
    int id[4]={k, k|(1<<P), k|(1<<Q), k|(1<<P)|(1<<Q)}; float2 w[4], r[4];
    #pragma unroll
    for(int j=0;j<4;j++) w[j]=v[id[j]];
    #pragma unroll
    for(int i=0;i<4;i++){ const float *m=d+3+8*i; r[i].x=0; r[i].y=0;
      #pragma unroll
      for(int j=0;j<4;j++){ r[i].x+=m[2*j]*w[j].x-m[2*j+1]*w[j].y; r[i].y+=m[2*j]*w[j].y+m[2*j+1]*w[j].x; } }
    #pragma unroll
    for(int j=0;j<4;j++) v[id[j]]=r[j]; }
}
#define C2(P,Q) case (P*4+Q): if(ty==1) cxr<P,Q>(v); else g2<P,Q>(v,d); break;
extern "C" __global__ void blk(float2 *psi, int n, int B, const long long *OFF, const int *Cg, int ngr, const int *GR, const float *G){
  extern __shared__ float2 sh[];
  __shared__ int C[48]; __shared__ long long boff_s;
  if(threadIdx.x<n-B) C[threadIdx.x]=Cg[threadIdx.x];
  __syncthreads();
  if(threadIdx.x==0){ long long base=blockIdx.x, b=0; for(int k=0;k<n-B;k++) if((base>>k)&1LL) b|=1LL<<C[k]; boff_s=b; }
  __syncthreads();
  long long boff=boff_s; int M=1<<B;
  for(int j=threadIdx.x;j<M;j+=blockDim.x) sh[j]=psi[boff|OFF[j]];
  __syncthreads();
  const int t=threadIdx.x;
  for(int g=0;g<ngr;g++){
    const int *h=GR+10*g; int base=t;
    #pragma unroll
    for(int j=0;j<4;j++){ int s=h[4+j]; base=((base>>s)<<(s+1))|(base&((1<<s)-1)); }
    int o0=1<<h[0], o1=1<<h[1], o2=1<<h[2], o3=1<<h[3]; float2 v[16];
    #pragma unroll
    for(int k=0;k<16;k++) v[k]=sh[base|((k&1)?o0:0)|((k&2)?o1:0)|((k&4)?o2:0)|((k&8)?o3:0)];
    for(int q=h[8];q<h[8]+h[9];q++){
      const float *d=G+35*q; int ty=(int)d[0], a=(int)d[1], b=(int)d[2];
      if(ty==0){ switch(a){ case 0: g1<0>(v,d); break; case 1: g1<1>(v,d); break; case 2: g1<2>(v,d); break; default: g1<3>(v,d); } }
      else switch(a*4+b){ C2(0,1) C2(0,2) C2(0,3) C2(1,0) C2(1,2) C2(1,3) C2(2,0) C2(2,1) C2(2,3) C2(3,0) C2(3,1) C2(3,2) }
    }
    #pragma unroll
    for(int k=0;k<16;k++) sh[base|((k&1)?o0:0)|((k&2)?o1:0)|((k&4)?o2:0)|((k&8)?o3:0)]=v[k];
    __syncthreads();
  }
  for(int j=threadIdx.x;j<M;j+=blockDim.x) psi[boff|OFF[j]]=sh[j];
}
'''
def gq(g): return (g[1],) if g[0]=="u" else (g[1],g[2])
def group(loc,B):
    """groupes gloutons de portes (ordre conserve) tenant dans 4 qubits locaux ; R complete a 4 qubits distincts"""
    grs=[]; R=[]; gs=[]
    def close():
        RR=list(R); k=0
        while len(RR)<4:
            if k not in RR: RR.append(k)
            k+=1
        grs.append((RR,list(gs)))
    for g in loc:
        new=[q for q in gq(g) if q not in R]
        if len(R)+len(new)>4: close(); R=[]; gs=[]; new=list(gq(g))
        R+=new; gs.append(g)
    if gs: close()
    return grs
def run_numpy_kernel(n,passes):
    """emulation NumPy fidele du noyau v26 (groupes, positions registre, insertion de bits)"""
    N=1<<n; psi=np.zeros(N,np.complex128); psi[0]=1
    for S,loc in passes:
        B=len(S); C=[q for q in range(n) if q not in S]
        bases=np.arange(1<<(n-B),dtype=np.int64); boff=np.zeros_like(bases)
        for k,q in enumerate(C): boff|=((bases>>k)&1)<<q
        j=np.arange(1<<B,dtype=np.int64); off=np.zeros_like(j)
        for k,q in enumerate(S): off|=((j>>k)&1)<<q
        idx=boff[:,None]|off[None,:]; sh=psi[idx]
        for R,gs in group(loc,B):
            t=np.arange(1<<(B-4)); base=t.copy()
            for s in sorted(R): base=((base>>s)<<(s+1))|(base&((1<<s)-1))
            kk=np.arange(16); ofs=sum(((kk>>j)&1)<<R[j] for j in range(4)); I=base[:,None]|ofs[None,:]
            v=sh[:,I]  # (blocs, threads, 16)
            pos={q:j for j,q in enumerate(R)}
            for g in gs:
                if g[0]=="u":
                    P=pos[g[1]]; M=g[2]; k0=[k for k in range(16) if not k&(1<<P)]; k1=[k|(1<<P) for k in k0]
                    x,y=v[...,k0].copy(),v[...,k1].copy(); v[...,k0]=M[0,0]*x+M[0,1]*y; v[...,k1]=M[1,0]*x+M[1,1]*y
                else:
                    P,Q=pos[g[1]],pos[g[2]]; ks=[k for k in range(16) if not k&(1<<P) and not k&(1<<Q)]
                    if g[0]=="cx":
                        i0=[k|(1<<P) for k in ks]; i1=[k|(1<<P)|(1<<Q) for k in ks]; tt=v[...,i0].copy(); v[...,i0]=v[...,i1]; v[...,i1]=tt
                    else:
                        ids=[[k,k|(1<<P),k|(1<<Q),k|(1<<P)|(1<<Q)] for k in ks]
                        for row in ids:
                            w=v[...,row].copy(); v[...,row]=np.einsum('ij,...j->...i',g[3],w)
            sh[:,I]=v
        psi[idx]=sh
    return psi
class RegSim:
    def __init__(s):
        s.k=cp.RawModule(code=SRC,options=("-std=c++14",),name_expressions=None).get_function("blk")
        try: s.k.max_dynamic_shared_size_bytes=65536
        except Exception as e: print("note: opt-in 64 Ko indisponible (%s)"%e)
    def upload(s,n,passes):
        out=[]
        for S,loc in passes:
            C=[q for q in range(n) if q not in S]; grs=group(loc,len(S)); rows=[]; GRi=[]
            for R,gs in grs:
                pos={q:j for j,q in enumerate(R)}; GRi.append(R+sorted(R)+[len(rows),len(gs)])
                for g in gs:
                    r=np.zeros(RW)
                    if g[0]=="u": M=g[2]; r[:11]=[0,pos[g[1]],0,M[0,0].real,M[0,0].imag,M[0,1].real,M[0,1].imag,M[1,0].real,M[1,0].imag,M[1,1].real,M[1,1].imag]
                    elif g[0]=="cx": r[:3]=[1,pos[g[1]],pos[g[2]]]
                    else: r[:3]=[2,pos[g[1]],pos[g[2]]]; r[3:]=np.stack([g[3].real,g[3].imag],-1).reshape(-1)
                    rows.append(r)
            j=np.arange(1<<len(S),dtype=np.int64); off=np.zeros_like(j)
            for k,q in enumerate(S): off|=((j>>k)&1)<<q
            out.append((len(S),cp.asarray(off),cp.asarray(np.array(C or [0],np.int32)),len(grs),cp.asarray(np.array(GRi,np.int32)),cp.asarray(np.array(rows,np.float32))))
        return out
    def run(s,n,dev):
        psi=cp.zeros(1<<n,dtype=cp.complex64); psi[0]=1
        for B,OFF,C,ngr,GR,G in dev:
            s.k((1<<(n-B),),(1<<(B-4),),(psi,np.int32(n),np.int32(B),OFF,C,np.int32(ngr),GR,G),shared_mem=(1<<B)*8)
        return psi
def main():
    L=[]; P=lambda s:(print(s,flush=True),L.append(s))
    P("="*100); P("DELTA QPU EMULE V26 | v25 + portes en registres (16 amplitudes/thread, 1 synchro par groupe) | RTX 4080"); P("="*100)
    ops=circuit(12,12,5); prog=compile_ops(12,ops); ref=qiskit_sv(12,ops)
    for B,Lb in ((8,3),(10,2)):
        ps=schedule_dag(12,prog,B,Lb,True); f=fid(ref,run_numpy_kernel(12,ps))
        P("emulation NumPy noyau v26 n=12 B=%d L=%d : %d passages, %d etapes, %d groupes  fidelite vs Qiskit=%.12f %s"%(B,Lb,len(ps),sum(len(l) for _,l in ps),sum(len(group(l,B)) for _,l in ps),f,"PASS" if f>1-1e-9 else "FAIL"))
    if not GPU: P("CuPy absent : arret apres validation logique."); return
    rs=RegSim(); v21=GPUSim()
    for n in (16,20,24):
        pr=compile_ops(n,circuit(n,20,n)); a=cp.asnumpy(v21.run(n,pr)).astype(np.complex128)
        b=cp.asnumpy(rs.run(n,rs.upload(n,schedule_dag(n,pr,13,3,True)))).astype(np.complex128); f=fid(a,b)
        P("validation GPU v26 FP32 vs GPU v21 FP64 n=%d : infidelite=%.2e  %s"%(n,1-f,"PASS" if abs(1-f)<1e-4 else "FAIL")); del a,b
        cp.get_default_memory_pool().free_all_blocks()
    V25={20:0.7,22:2.5,24:12.6,26:53.4,28:224.9,29:474.5,30:997.5}; V22={20:10.8,22:42.8,24:171.7,26:738.7,28:3171.2,29:6555.4}
    free=cp.cuda.runtime.memGetInfo()[0]; best={}
    for n in (20,22,24,26,28,29,30):
        if (1<<n)*8>free*0.85: P("n=%-2d : VRAM insuffisante, saute"%n); continue
        pr=compile_ops(n,circuit(n,20,n))
        for B,Lb in ((12,2),(12,3),(13,3)):
            try:
                ps=schedule_dag(n,pr,B,Lb,True); dev=rs.upload(n,ps); rs.run(n,rs.upload(n,schedule_dag(n,compile_ops(n,circuit(n,1,1)),B,Lb,True))); cp.cuda.Device().synchronize()
                ms=1e30
                for it in range(2):
                    e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(); psi=rs.run(n,dev); e1.record(); e1.synchronize(); ms=min(ms,cp.cuda.get_elapsed_time(e0,e1))
                    if it==0: del psi
                x=psi.view(cp.float32); K=1<<26; norm=math.sqrt(sum(float(x[i:i+K].dot(x[i:i+K])) for i in range(0,x.size,K))); del x,psi; cp.get_default_memory_pool().free_all_blocks()
                best[n]=min(best.get(n,1e30),ms); ng=sum(len(group(l,B)) for _,l in ps)
                P("n=%-2d B=%-2d L=%d : %3d passages, %4d etapes -> %4d groupes : %8.1f ms  |norme-1|=%.1e  ~%4.0f Go/s  vs v25 x%4.2f  vs v22 FP64 %s"%(n,B,Lb,len(ps),sum(len(l) for _,l in ps),ng,ms,abs(norm-1),len(ps)*(1<<n)*16/1e9/(ms/1e3),V25[n]/ms,("x%.0f"%(V22[n]/ms)) if n in V22 else "-"))
            except Exception as e:
                P("n=%-2d B=%-2d L=%d : ECHEC %s"%(n,B,Lb,str(e)[:90])); cp.get_default_memory_pool().free_all_blocks()
    P("-"*100); P("MEILLEUR V26 : "+"  ".join("%dq %.1f ms (x%.2f v25)"%(n,best[n],V25[n]/best[n]) for n in sorted(best)))
    P("="*100)
    f=os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_gpu_v26_results.txt"); open(f,"w").write("\n".join(L)+"\n"); print("JOURNAL="+f)
if __name__=="__main__": main()
