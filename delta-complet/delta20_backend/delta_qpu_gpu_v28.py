#!/usr/bin/env python3
"""DELTA QPU EMULE V28 — v26 + ordonnanceur meilleur-de + groupes registre a 4 OU 5 qubits + diagnostic plancher memoire (RTX 4080).
Constat v26 (30 q) : 12 passages x 16 Go = 192 Go -> plancher 322 ms a 596 Go/s ; mesure 886 ms -> le noyau n'est pas au plafond VRAM.
(1) NOUVEAU ordonnanceur 'bo' (meilleur-de) : chaque passage est simule avec 4 strategies, on garde celle qui absorbe
    le plus de portes (gain modeste mesure hors GPU : 30 q B=13 L=2 11->10 passages, B=14 10->9).
(2) NOUVEAU NR=5 : 32 amplitudes par thread en registres (5 qubits) -> groupes plus gros, moins de synchros (NR=4 = v26).
(3) NOUVEAU L=4 teste : paquets contigus de 16 amplitudes = lignes de 128 octets pleines (coalescence).
(4) NOUVEAU diagnostic : meme noyau SANS portes (copie seule) -> temps memoire pur ; ecart = calcul + synchros.
Validation : emulation NumPy fidele du noyau vs Qiskit (n=12) ; GPU v28 FP32 vs GPU v21 FP64 (n=16, 20, 24)."""
import os, sys, math, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_arena_v20 import circuit, qiskit_sv
from delta_qpu_gpu_v21 import compile_ops, fid, GPU
from delta_qpu_gpu_v25 import schedule_dag, fuse, gq
if GPU:
    import cupy as cp
    from delta_qpu_gpu_v21 import GPUSim
RW=35
def gen_src():
    c1=lambda NR:" ".join("case %d: g1<NR,%d>(v,d); break;"%(p,p) for p in range(NR))
    c2=lambda NR:" ".join("case %d: if(ty==1) cxr<NR,%d,%d>(v); else g2<NR,%d,%d>(v,d); break;"%(p*NR+q,p,q,p,q) for p in range(NR) for q in range(NR) if p!=q)
    return r'''
template<int NR,int P> __device__ __forceinline__ void g1(float2 *v,const float *d){
  #pragma unroll
  for(int k=0;k<(1<<NR);k++) if(!(k&(1<<P))){ int k1=k|(1<<P); float2 x=v[k], y=v[k1], r0, r1;
    r0.x=d[3]*x.x-d[4]*x.y+d[5]*y.x-d[6]*y.y; r0.y=d[3]*x.y+d[4]*x.x+d[5]*y.y+d[6]*y.x;
    r1.x=d[7]*x.x-d[8]*x.y+d[9]*y.x-d[10]*y.y; r1.y=d[7]*x.y+d[8]*x.x+d[9]*y.y+d[10]*y.x; v[k]=r0; v[k1]=r1; }
}
template<int NR,int P,int Q> __device__ __forceinline__ void cxr(float2 *v){
  #pragma unroll
  for(int k=0;k<(1<<NR);k++) if(!(k&(1<<P)) && !(k&(1<<Q))){ int i0=k|(1<<P), i1=i0|(1<<Q); float2 t=v[i0]; v[i0]=v[i1]; v[i1]=t; }
}
template<int NR,int P,int Q> __device__ __forceinline__ void g2(float2 *v,const float *d){
  #pragma unroll
  for(int k=0;k<(1<<NR);k++) if(!(k&(1<<P)) && !(k&(1<<Q))){
    int id0=k, id1=k|(1<<P), id2=k|(1<<Q), id3=k|(1<<P)|(1<<Q); float2 w[4]={v[id0],v[id1],v[id2],v[id3]}, r[4];
    #pragma unroll
    for(int i=0;i<4;i++){ const float *m=d+3+8*i; r[i].x=0; r[i].y=0;
      #pragma unroll
      for(int j=0;j<4;j++){ r[i].x+=m[2*j]*w[j].x-m[2*j+1]*w[j].y; r[i].y+=m[2*j]*w[j].y+m[2*j+1]*w[j].x; } }
    v[id0]=r[0]; v[id1]=r[1]; v[id2]=r[2]; v[id3]=r[3]; }
}
template<int NR> __device__ __forceinline__ void apply(float2 *v,const float *d){
  int ty=(int)d[0], a=(int)d[1], b=(int)d[2];
  if constexpr(NR==4){ if(ty==0){ switch(a){ '''+c1(4)+''' } } else { switch(a*NR+b){ '''+c2(4)+''' } } }
  else     { if(ty==0){ switch(a){ '''+c1(5)+''' } } else { switch(a*NR+b){ '''+c2(5)+''' } } }
}
template<int NR> __global__ void blk(float2 *psi, int n, int B, const long long *OFF, const int *Cg, int ngr, const int *GR, const float *G){
  extern __shared__ float2 sh[];
  __shared__ int C[48]; __shared__ long long boff_s;
  if(threadIdx.x<n-B) C[threadIdx.x]=Cg[threadIdx.x];
  __syncthreads();
  if(threadIdx.x==0){ long long base=blockIdx.x, b=0; for(int k=0;k<n-B;k++) if((base>>k)&1LL) b|=1LL<<C[k]; boff_s=b; }
  __syncthreads();
  long long boff=boff_s; int M=1<<B;
  for(int j=threadIdx.x;j<M;j+=blockDim.x) sh[j]=psi[boff|OFF[j]];
  __syncthreads();
  const int t=threadIdx.x, W=2*NR+2;
  for(int g=0;g<ngr;g++){
    const int *h=GR+W*g; int base=t; int o[NR];
    #pragma unroll
    for(int j=0;j<NR;j++){ int s=h[NR+j]; base=((base>>s)<<(s+1))|(base&((1<<s)-1)); o[j]=1<<h[j]; }
    float2 v[1<<NR];
    #pragma unroll
    for(int k=0;k<(1<<NR);k++){ int x=base;
      #pragma unroll
      for(int j=0;j<NR;j++) if(k&(1<<j)) x|=o[j];
      v[k]=sh[x]; }
    for(int q=h[2*NR];q<h[2*NR]+h[2*NR+1];q++) apply<NR>(v,G+35*q);
    #pragma unroll
    for(int k=0;k<(1<<NR);k++){ int x=base;
      #pragma unroll
      for(int j=0;j<NR;j++) if(k&(1<<j)) x|=o[j];
      sh[x]=v[k]; }
    __syncthreads();
  }
  for(int j=threadIdx.x;j<M;j+=blockDim.x) psi[boff|OFF[j]]=sh[j];
}
'''
def absorb(prog,Q,ptr,S,out):
    """absorbe (et consomme dans ptr) toutes les portes pretes dont les qubits sont dans S"""
    moved=True; c=0
    while moved:
        moved=False
        for q in sorted(S):
            while ptr[q]<len(Q[q]):
                i=Q[q][ptr[q]]; g=prog[i]
                if set(gq(g))<=S and all(Q[r][ptr[r]]==i for r in gq(g)):
                    if out is not None: out.append(g)
                    c+=1; moved=True
                    for r in gq(g): ptr[r]+=1
                else: break
    return c
def onepass(prog,Q,ptr,B,FIXED,mode):
    """simule UN passage avec une strategie de croissance de S ; renvoie (S, portes, ptr apres)"""
    ptr=list(ptr); S=set(FIXED); gs=[]; absorb(prog,Q,ptr,S,gs)
    while len(S)<B:
        cands={}
        for q in range(len(Q)):
            if ptr[q]<len(Q[q]):
                i=Q[q][ptr[q]]; g=prog[i]
                if all(Q[r][ptr[r]]==i for r in gq(g)):
                    new=frozenset(gq(g))-S
                    if new and len(S)+len(new)<=B: cands[new]=min(cands.get(new,1<<30),i)
        if not cands: break
        if mode=="dag": new=min(cands.items(),key=lambda t:(len(t[0]),t[1]))[0]
        else:
            sc=[]
            for new,i in cands.items():
                c=absorb(prog,Q,list(ptr),S|new,None)
                sc.append(({"ratio":(c/len(new),-len(new),-i),"tot":(c,-len(new),-i),"old":(-min(ptr[q] for q in new),-len(new),-i)}[mode],new))
            new=max(sc,key=lambda t:t[0])[1]
        S|=new; absorb(prog,Q,ptr,S,gs)
    return S,gs,ptr
def schedule_bo(n,prog,B,L=2,fz=True):
    """ordonnanceur 'meilleur-de' : a chaque passage, 4 strategies sont simulees (dag v26, ratio, total, anciennete)
    et on garde celle qui absorbe le plus de portes"""
    B=min(B,n); FIXED=set(range(min(L,B))); Q=[[] for _ in range(n)]
    for i,g in enumerate(prog):
        for q in gq(g): Q[q].append(i)
    ptr=[0]*n; left=len(prog); passes=[]
    while left:
        S,gs,ptr=max((onepass(prog,Q,ptr,B,FIXED,m) for m in ("dag","ratio","tot","old")),key=lambda t:len(t[1]))
        if not gs: raise RuntimeError("blocage ordonnanceur")
        left-=len(gs); k=0
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
def sched(kind,n,prog,B,L):
    a=schedule_dag(n,prog,B,L,True)
    if kind!="bo": return a
    b=schedule_bo(n,prog,B,L,True); return b if len(b)<len(a) else a  # jamais pire que v26
def group(loc,NR):
    grs=[]; R=[]; gs=[]
    def close():
        RR=list(R); k=0
        while len(RR)<NR:
            if k not in RR: RR.append(k)
            k+=1
        grs.append((RR,list(gs)))
    for g in loc:
        new=[q for q in gq(g) if q not in R]
        if len(R)+len(new)>NR: close(); R=[]; gs=[]; new=list(gq(g))
        R+=new; gs.append(g)
    if gs: close()
    return grs
def run_numpy_kernel(n,passes,NR):
    """emulation NumPy fidele du noyau v28 (groupes NR qubits, positions registre, insertion de bits)"""
    N=1<<n; psi=np.zeros(N,np.complex128); psi[0]=1; K=1<<NR
    for S,loc in passes:
        B=len(S); C=[q for q in range(n) if q not in S]
        bases=np.arange(1<<(n-B),dtype=np.int64); boff=np.zeros_like(bases)
        for k,q in enumerate(C): boff|=((bases>>k)&1)<<q
        j=np.arange(1<<B,dtype=np.int64); off=np.zeros_like(j)
        for k,q in enumerate(S): off|=((j>>k)&1)<<q
        idx=boff[:,None]|off[None,:]; sh=psi[idx]
        for R,gs in group(loc,NR):
            base=np.arange(1<<(B-NR))
            for s in sorted(R): base=((base>>s)<<(s+1))|(base&((1<<s)-1))
            kk=np.arange(K); ofs=sum(((kk>>j)&1)<<R[j] for j in range(NR)); I=base[:,None]|ofs[None,:]
            v=sh[:,I]; pos={q:j for j,q in enumerate(R)}
            for g in gs:
                if g[0]=="u":
                    P=pos[g[1]]; M=g[2]; k0=[k for k in range(K) if not k&(1<<P)]; k1=[k|(1<<P) for k in k0]
                    x,y=v[...,k0].copy(),v[...,k1].copy(); v[...,k0]=M[0,0]*x+M[0,1]*y; v[...,k1]=M[1,0]*x+M[1,1]*y
                else:
                    P,Qb=pos[g[1]],pos[g[2]]; ks=[k for k in range(K) if not k&(1<<P) and not k&(1<<Qb)]
                    if g[0]=="cx":
                        i0=[k|(1<<P) for k in ks]; i1=[k|(1<<P)|(1<<Qb) for k in ks]; tt=v[...,i0].copy(); v[...,i0]=v[...,i1]; v[...,i1]=tt
                    else:
                        for k in ks:
                            row=[k,k|(1<<P),k|(1<<Qb),k|(1<<P)|(1<<Qb)]; w=v[...,row].copy(); v[...,row]=np.einsum('ij,...j->...i',g[3],w)
            sh[:,I]=v
        psi[idx]=sh
    return psi
class RegSim:
    def __init__(s):
        m=cp.RawModule(code=gen_src(),options=("-std=c++17",),name_expressions=["blk<4>","blk<5>"])
        s.k={4:m.get_function("blk<4>"),5:m.get_function("blk<5>")}
        for f in s.k.values():
            try: f.max_dynamic_shared_size_bytes=65536
            except Exception as e: print("note: opt-in 64 Ko indisponible (%s)"%e)
    def upload(s,n,passes,NR):
        out=[]
        for S,loc in passes:
            C=[q for q in range(n) if q not in S]; grs=group(loc,NR); rows=[]; GRi=[]
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
            out.append((len(S),cp.asarray(off),cp.asarray(np.array(C or [0],np.int32)),len(grs),cp.asarray(np.array(GRi,np.int32)),cp.asarray(np.array(rows or [np.zeros(RW)],np.float32))))
        return out
    def run(s,n,dev,NR,copy=False):
        psi=cp.zeros(1<<n,dtype=cp.complex64); psi[0]=1
        for B,OFF,C,ngr,GR,G in dev:
            s.k[NR]((1<<(n-B),),(1<<(B-NR),),(psi,np.int32(n),np.int32(B),OFF,C,np.int32(0 if copy else ngr),GR,G),shared_mem=(1<<B)*8)
        return psi
def tmin(f,it=2):
    ms=1e30; r=None
    for _ in range(it):
        r=None; cp.get_default_memory_pool().free_all_blocks()
        e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(); r=f(); e1.record(); e1.synchronize(); ms=min(ms,cp.cuda.get_elapsed_time(e0,e1))
    return ms,r
def main():
    L=[]; P=lambda s:(print(s,flush=True),L.append(s))
    P("="*104); P("DELTA QPU EMULE V28 | v26 + ordonnanceur meilleur-de + registres NR=4/5 + L=4 + diagnostic plancher memoire | RTX 4080"); P("="*104)
    ops=circuit(12,12,5); prog=compile_ops(12,ops); ref=qiskit_sv(12,ops)
    for kind,B,Lb,NR in (("dag",8,3,4),("bo",8,3,4),("bo",10,2,5),("bo",10,4,5),("bo",9,2,5)):
        ps=sched(kind,12,prog,B,Lb); f=fid(ref,run_numpy_kernel(12,ps,NR))
        P("emulation NumPy noyau v28 n=12 %-3s B=%d L=%d NR=%d : %d passages, %d groupes  fidelite vs Qiskit=%.12f %s"%(kind,B,Lb,NR,len(ps),sum(len(group(l,NR)) for _,l in ps),f,"PASS" if f>1-1e-9 else "FAIL"))
    P("-"*104)
    for n in (24,28,30):
        pr=compile_ops(n,circuit(n,20,n))
        for B,Lb in ((12,2),(13,2),(13,3),(13,4)):
            t=time.perf_counter(); a=len(schedule_dag(n,pr,B,Lb,True)); b=len(sched("bo",n,pr,B,Lb))
            P("passages n=%d B=%d L=%d : dag (v26) %2d -> bo (v28) %2d   [%.1f s ordonnancement]"%(n,B,Lb,a,b,time.perf_counter()-t))
    if not GPU: P("CuPy absent : arret apres validation logique."); return
    rs=RegSim(); v21=GPUSim()
    for n in (16,20,24):
        pr=compile_ops(n,circuit(n,20,n)); a=cp.asnumpy(v21.run(n,pr)).astype(np.complex128)
        for NR in (4,5):
            b=cp.asnumpy(rs.run(n,rs.upload(n,sched("bo",n,pr,13,2),NR),NR)).astype(np.complex128); f=fid(a,b)
            P("validation GPU v28 FP32 NR=%d vs GPU v21 FP64 n=%d : infidelite=%.2e  %s"%(NR,n,1-f,"PASS" if abs(1-f)<1e-4 else "FAIL"))
        del a,b; cp.get_default_memory_pool().free_all_blocks()
    V26={28:200.0,30:886.3}; free=cp.cuda.runtime.memGetInfo()[0]; best={}
    CFG=(("dag",12,2,4),("bo",12,2,4),("bo",13,2,4),("bo",13,4,4),("bo",12,2,5),("bo",13,2,5),("bo",13,4,5))
    for n in (20,24,28,29,30):
        if (1<<n)*8>free*0.85: P("n=%-2d : VRAM insuffisante, saute"%n); continue
        pr=compile_ops(n,circuit(n,20,n))
        for kind,B,Lb,NR in CFG:
            try:
                ps=sched(kind,n,pr,B,Lb); dev=rs.upload(n,ps,NR); rs.run(20,rs.upload(20,sched(kind,20,compile_ops(20,circuit(20,1,1)),B,Lb),NR),NR); cp.cuda.Device().synchronize()
                ms,psi=tmin(lambda:rs.run(n,dev,NR))
                x=psi.view(cp.float32); K=1<<26; norm=math.sqrt(sum(float(x[i:i+K].dot(x[i:i+K])) for i in range(0,x.size,K))); del x,psi
                mc,psi=tmin(lambda:rs.run(n,dev,NR,True)); del psi; cp.get_default_memory_pool().free_all_blocks()
                ng=sum(len(group(l,NR)) for _,l in ps); gbs=len(ps)*(1<<n)*16/1e9; tag="%s B=%d L=%d NR=%d"%(kind,B,Lb,NR)
                if ms<best.get(n,(1e30,""))[0]: best[n]=(ms,tag)
                P("n=%-2d %-20s: %2d passages, %4d groupes : %8.1f ms (copie seule %7.1f ms = %3.0f%%, %4.0f Go/s)  |norme-1|=%.1e  %s"%(n,tag,len(ps),ng,ms,mc,100*mc/ms,gbs/(mc/1e3),abs(norm-1),("vs v26 x%.2f"%(V26[n]/ms)) if n in V26 else ""))
            except Exception as e:
                P("n=%-2d %s B=%d L=%d NR=%d : ECHEC %s"%(n,kind,B,Lb,NR,str(e)[:90])); cp.get_default_memory_pool().free_all_blocks()
    P("-"*104); P("MEILLEUR V28 : "+"  ".join("%dq %.1f ms [%s]"%(n,best[n][0],best[n][1]) for n in sorted(best)))
    P("LECTURE : 'copie seule' = memes acces memoire sans aucune porte ; si proche du total -> goulot VRAM (reduire passages/v29 L2) ; sinon goulot calcul/synchros.")
    P("="*104)
    f=os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_gpu_v28_results.txt"); open(f,"w").write("\n".join(L)+"\n"); print("JOURNAL="+f)
if __name__=="__main__": main()
