#!/usr/bin/env python3
"""DELTA QPU EMULE V31 — v30 (swizzle) + groupes de 9 qubits par warp avec echanges de main a main (shuffle) (RTX 4080).
Lecons v30 : swizzle = record 30 q 843,7 ms ; partie calcul ~403 ms = ~104 groupes de 4 qubits, chacun = aller-retour au guichet + __syncthreads.
v31 : un warp (32 threads x 16 amplitudes = 512) couvre 9 qubits : 4 en registres + 5 en 'lanes'. Une porte sur un qubit-lane
s'applique par __shfl_xor_sync entre threads du meme warp, sans memoire partagee ni synchro. Un groupe tient jusqu'a 9 qubits
au lieu de 4 -> beaucoup moins de groupes, donc moins d'allers-retours au guichet.
Validation : emulation NumPy EXACTE des echanges (lanes simulees) vs Qiskit (n=12) ; GPU v31 vs GPU v21 FP64 (n=16, 20, 24) ;
v30 SW rechronometree dans le meme lancement."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_arena_v20 import circuit, qiskit_sv
from delta_qpu_gpu_v21 import compile_ops, fid, GPU
from delta_qpu_gpu_v25 import CX4, gq
from delta_qpu_gpu_v26 import SRC as SRC26, RW, group as group4
from delta_qpu_gpu_v29 import schedule_rl, to_logical, offs
if GPU:
    import cupy as cp
    from delta_qpu_gpu_v21 import GPUSim
    from delta_qpu_gpu_v30 import Sim30
GW=16
KERN=SRC26[:SRC26.index('extern "C" __global__ void blk')]+r'''
#define SWZ(x) ((x)^(((x)>>4)&15))
#define FULL 0xffffffffu
__device__ __forceinline__ float2 shx(float2 a,unsigned m){ float2 r; r.x=__shfl_xor_sync(FULL,a.x,m); r.y=__shfl_xor_sync(FULL,a.y,m); return r; }
__device__ __forceinline__ void macc(float2 &o,const float *c,float2 a){ o.x+=c[0]*a.x-c[1]*a.y; o.y+=c[0]*a.y+c[1]*a.x; }
#define CM(r,c) (d+3+8*(r)+2*(c))
__device__ __forceinline__ void g1L(float2 *v,const float *d,int j,int lane){
  int b=(lane>>j)&1; const float *A=b?d+9:d+3, *Bc=b?d+7:d+5; unsigned m=1u<<j;
  #pragma unroll
  for(int k=0;k<16;k++){ float2 y=shx(v[k],m), o; o.x=0; o.y=0; macc(o,A,v[k]); macc(o,Bc,y); v[k]=o; }
}
template<int P> __device__ __forceinline__ void g2RL(float2 *v,const float *d,int j,int lane,int regFirst){
  int lb=(lane>>j)&1; unsigned m=1u<<j;
  int i00=regFirst?(0+2*lb):(lb+0), i10=regFirst?(1+2*lb):(lb+2), p00=regFirst?(0+2*(1-lb)):((1-lb)+0), p10=regFirst?(1+2*(1-lb)):((1-lb)+2);
  #pragma unroll
  for(int k=0;k<16;k++) if(!(k&(1<<P))){ int k1=k|(1<<P); float2 a0=v[k], a1=v[k1], q0=shx(a0,m), q1=shx(a1,m), o0, o1; o0.x=o0.y=o1.x=o1.y=0;
    macc(o0,CM(i00,i00),a0); macc(o0,CM(i00,i10),a1); macc(o0,CM(i00,p00),q0); macc(o0,CM(i00,p10),q1);
    macc(o1,CM(i10,i00),a0); macc(o1,CM(i10,i10),a1); macc(o1,CM(i10,p00),q0); macc(o1,CM(i10,p10),q1);
    v[k]=o0; v[k1]=o1; }
}
__device__ __forceinline__ void g2LL(float2 *v,const float *d,int j1,int j2,int lane){
  int l1=(lane>>j1)&1, l2=(lane>>j2)&1; unsigned m1=1u<<j1, m2=1u<<j2;
  int my=l1+2*l2, i1=(1-l1)+2*l2, i2=l1+2*(1-l2), i3=(1-l1)+2*(1-l2);
  #pragma unroll
  for(int k=0;k<16;k++){ float2 a=v[k], p1=shx(a,m1), p2=shx(a,m2), p3=shx(a,m1|m2), o; o.x=0; o.y=0;
    macc(o,CM(my,my),a); macc(o,CM(my,i1),p1); macc(o,CM(my,i2),p2); macc(o,CM(my,i3),p3); v[k]=o; }
}
extern "C" __global__ void blk31(float2 *psi, int n, int B, const long long *OFF, const int *Cg, int ngr, const int *GR, const float *G, const long long *OFF2){
  extern __shared__ float2 sh[];
  __shared__ int C[48]; __shared__ long long boff_s;
  if(threadIdx.x<n-B) C[threadIdx.x]=Cg[threadIdx.x];
  __syncthreads();
  if(threadIdx.x==0){ long long base=blockIdx.x, b=0; for(int k=0;k<n-B;k++) if((base>>k)&1LL) b|=1LL<<C[k]; boff_s=b; }
  __syncthreads();
  const long long boff=boff_s; const int M=1<<B, T=blockDim.x, t=threadIdx.x, lane=t&31, w=t>>5, nO=B-9;
  for(int j=t;j<M;j+=T) sh[SWZ(j)]=psi[boff|OFF[j]];
  __syncthreads();
  for(int g=0;g<ngr;g++){
    const int *h=GR+16*g; int base=0;
    #pragma unroll
    for(int j=0;j<5;j++) if((lane>>j)&1) base|=1<<h[4+j];
    for(int j=0;j<nO;j++) if((w>>j)&1) base|=1<<h[9+j];
    int o0=1<<h[0], o1=1<<h[1], o2=1<<h[2], o3=1<<h[3]; float2 v[16];
    #pragma unroll
    for(int k=0;k<16;k++){ int x=base|((k&1)?o0:0)|((k&2)?o1:0)|((k&4)?o2:0)|((k&8)?o3:0); v[k]=sh[SWZ(x)]; }
    for(int q=h[13];q<h[13]+h[14];q++){
      const float *d=G+35*q; int ty=(int)d[0], a=(int)d[1], b=(int)d[2];
      if(ty==0){ if(a<4){ switch(a){ case 0: g1<0>(v,d); break; case 1: g1<1>(v,d); break; case 2: g1<2>(v,d); break; default: g1<3>(v,d); } } else g1L(v,d,a-4,lane); }
      else if(a<4 && b<4){ switch(a*4+b){ C2(0,1) C2(0,2) C2(0,3) C2(1,0) C2(1,2) C2(1,3) C2(2,0) C2(2,1) C2(2,3) C2(3,0) C2(3,1) C2(3,2) } }
      else if(a<4){ switch(a){ case 0: g2RL<0>(v,d,b-4,lane,1); break; case 1: g2RL<1>(v,d,b-4,lane,1); break; case 2: g2RL<2>(v,d,b-4,lane,1); break; default: g2RL<3>(v,d,b-4,lane,1); } }
      else if(b<4){ switch(b){ case 0: g2RL<0>(v,d,a-4,lane,0); break; case 1: g2RL<1>(v,d,a-4,lane,0); break; case 2: g2RL<2>(v,d,a-4,lane,0); break; default: g2RL<3>(v,d,a-4,lane,0); } }
      else g2LL(v,d,a-4,b-4,lane);
    }
    #pragma unroll
    for(int k=0;k<16;k++){ int x=base|((k&1)?o0:0)|((k&2)?o1:0)|((k&4)?o2:0)|((k&8)?o3:0); sh[SWZ(x)]=v[k]; }
    __syncthreads();
  }
  for(int j=t;j<M;j+=T) psi[boff|OFF2[j]]=sh[SWZ(j)];
}
'''
def group9(loc,B):
    """groupes gloutons de 9 qubits locaux : 4 registres (les plus sollicites) + 5 lanes ; renvoie (R, Lq, others, portes en slots)"""
    out=[]; W=[]; gs=[]
    def close():
        cnt={q:0 for q in W}
        for g in gs:
            for q in gq(g): cnt[q]+=1+(g[0]!="u")
        order=sorted(W,key=lambda q:(-cnt[q],q)); pad=[q for q in range(B) if q not in W]
        full=order+pad[:9-len(order)]; R=full[:4]; Lq=full[4:9]; oth=sorted(q for q in range(B) if q not in full)
        slot={q:i for i,q in enumerate(R)}; slot.update({q:4+j for j,q in enumerate(Lq)}); enc=[]
        for g in gs:
            if g[0]=="u": enc.append(("u",slot[g[1]],g[2]))
            elif g[0]=="cx" and slot[g[1]]<4 and slot[g[2]]<4: enc.append(("cx",slot[g[1]],slot[g[2]]))
            elif g[0]=="cx": enc.append(("u2",slot[g[1]],slot[g[2]],CX4))
            else: enc.append(("u2",slot[g[1]],slot[g[2]],g[3]))
        out.append((R,Lq,oth,enc))
    for g in loc:
        new=[q for q in gq(g) if q not in W]
        if len(W)+len(new)>9: close(); W=[]; gs=[]; new=list(gq(g))
        W+=new; gs.append(g)
    if gs: close()
    return out
def shx(v,m): return v[:,:,np.arange(32)^m,:]
def run_numpy_v31(n,passes):
    """emulation NumPy EXACTE du noyau v31 : warps et lanes simules, shuffles = permutations d'axe"""
    N=1<<n; psi=np.zeros(N,np.complex128); psi[0]=1; lane=np.arange(32)[None,None,:]
    for Sl,loc,P,P2 in passes:
        B=len(P); C=[p for p in range(n) if p not in P]; bases=np.arange(1<<(n-B),dtype=np.int64); boff=np.zeros_like(bases)
        for k,p in enumerate(C): boff|=((bases>>k)&1)<<p
        sh=psi[boff[:,None]|offs(P)[None,:]]; nb=sh.shape[0]; NW=1<<(B-9)
        for R,Lq,oth,enc in group9(loc,B):
            w=np.arange(NW)[:,None]; l=np.arange(32)[None,:]; base=np.zeros((NW,32),np.int64)
            for j,q in enumerate(Lq): base|=((l>>j)&1)<<q
            for j,q in enumerate(oth): base|=((w>>j)&1)<<q
            kk=np.arange(16); ko=sum(((kk>>j)&1)<<R[j] for j in range(4)); I=base[:,:,None]|ko[None,None,:]
            v=sh[:,I]  # (nb, NW, 32, 16)
            for g in enc:
                if g[0]=="u" and g[1]<4:
                    Pp=g[1]; M=g[2]; k0=[k for k in range(16) if not k&(1<<Pp)]; k1=[k|(1<<Pp) for k in k0]
                    x,y=v[...,k0].copy(),v[...,k1].copy(); v[...,k0]=M[0,0]*x+M[0,1]*y; v[...,k1]=M[1,0]*x+M[1,1]*y
                elif g[0]=="u":
                    j=g[1]-4; M=g[2]; b=((lane>>j)&1)[...,None]; y=shx(v,1<<j)
                    v=np.where(b==0,M[0,0]*v+M[0,1]*y,M[1,1]*v+M[1,0]*y)
                elif g[0]=="cx":
                    Pp,Qq=g[1],g[2]; ks=[k for k in range(16) if not k&(1<<Pp) and not k&(1<<Qq)]
                    i0=[k|(1<<Pp) for k in ks]; i1=[k|(1<<Pp)|(1<<Qq) for k in ks]; tt=v[...,i0].copy(); v[...,i0]=v[...,i1]; v[...,i1]=tt
                else:
                    a,b,m=g[1],g[2],g[3]
                    if a<4 and b<4:
                        ks=[k for k in range(16) if not k&(1<<a) and not k&(1<<b)]
                        for k in ks:
                            row=[k,k|(1<<a),k|(1<<b),k|(1<<a)|(1<<b)]; ww=v[...,row].copy(); v[...,row]=np.einsum('ij,...j->...i',m,ww)
                    elif a<4 or b<4:
                        Pp,j,rf=(a,b-4,1) if a<4 else (b,a-4,0); lb=((lane>>j)&1)[...,None]; nv=v.copy()
                        idx=lambda r,lbit:(r+2*lbit) if rf else (lbit+2*r)
                        for k in range(16):
                            if k&(1<<Pp): continue
                            k1=k|(1<<Pp); a0,a1=v[...,k],v[...,k1]; q0,q1=shx(v,1<<j)[...,k],shx(v,1<<j)[...,k1]; L0=lb[...,0]
                            for kk_,r in ((k,0),(k1,1)):
                                acc=0
                                for (val,rr,ll) in ((a0,0,L0),(a1,1,L0),(q0,0,1-L0),(q1,1,1-L0)):
                                    acc=acc+m[idx(r,L0),idx(rr,ll)]*val
                                nv[...,kk_]=acc
                        v=nv
                    else:
                        j1,j2=a-4,b-4; l1=((lane>>j1)&1)[...,None]; l2=((lane>>j2)&1)[...,None]
                        my=l1+2*l2; i1=(1-l1)+2*l2; i2=l1+2*(1-l2); i3=(1-l1)+2*(1-l2)
                        p1,p2,p3=shx(v,1<<j1),shx(v,1<<j2),shx(v,(1<<j1)|(1<<j2))
                        v=m[my,my]*v+m[my,i1]*p1+m[my,i2]*p2+m[my,i3]*p3
            sh[:,I]=v
        out=np.empty_like(psi); out[boff[:,None]|offs(P2)[None,:]]=sh; psi=out
    return psi
class Sim31:
    def __init__(s):
        s.k=cp.RawModule(code=KERN,options=("-std=c++14",)).get_function("blk31")
        try: s.k.max_dynamic_shared_size_bytes=65536
        except Exception as e: print("note: opt-in 64 Ko indisponible (%s)"%e)
    def upload(s,n,passes):
        out=[]
        for Sl,loc,P,P2 in passes:
            B=len(P); C=[p for p in range(n) if p not in P]; grs=group9(loc,B); rows=[]; GRi=[]
            for R,Lq,oth,enc in grs:
                GRi.append(R+Lq+(oth+[0]*4)[:4]+[len(rows),len(enc),0])
                for g in enc:
                    r=np.zeros(RW)
                    if g[0]=="u": M=g[2]; r[:11]=[0,g[1],0,M[0,0].real,M[0,0].imag,M[0,1].real,M[0,1].imag,M[1,0].real,M[1,0].imag,M[1,1].real,M[1,1].imag]
                    elif g[0]=="cx": r[:3]=[1,g[1],g[2]]
                    else: r[:3]=[2,g[1],g[2]]; r[3:]=np.stack([g[3].real,g[3].imag],-1).reshape(-1)
                    rows.append(r)
            out.append((B,cp.asarray(offs(P)),cp.asarray(np.array(C or [0],np.int32)),len(grs),cp.asarray(np.array(GRi,np.int32)),cp.asarray(np.array(rows,np.float32)),cp.asarray(offs(P2))))
        return out
    def run(s,n,dev,copy=False):
        psi=cp.zeros(1<<n,dtype=cp.complex64); psi[0]=1
        for B,OFF,C,ngr,GR,G,OFF2 in dev:
            s.k((1<<(n-B),),(1<<(B-4),),(psi,np.int32(n),np.int32(B),OFF,C,np.int32(0 if copy else ngr),GR,G,OFF2),shared_mem=(1<<B)*8)
        return psi
def tmin(f,it=3):
    ms=1e30; r=None
    for _ in range(it):
        r=None; cp.get_default_memory_pool().free_all_blocks()
        e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(); r=f(); e1.record(); e1.synchronize(); ms=min(ms,cp.cuda.get_elapsed_time(e0,e1))
    return ms,r
def norm1(psi):
    x=psi.view(cp.float32); K=1<<26; return abs(math.sqrt(sum(float(x[i:i+K].dot(x[i:i+K])) for i in range(0,x.size,K)))-1)
def main():
    L=[]; P=lambda s:(print(s,flush=True),L.append(s))
    P("="*104); P("DELTA QPU EMULE V31 | v30 swizzle + groupes 9 qubits/warp (4 registres + 5 lanes en shuffle) | RTX 4080"); P("="*104)
    ops=circuit(12,12,5); prog=compile_ops(12,ops); ref=qiskit_sv(12,ops)
    for B,Lb in ((9,2),(10,3),(11,3)):
        ps,ph=schedule_rl(12,prog,B,Lb); f=fid(ref,to_logical(run_numpy_v31(12,ps),12,ph))
        P("emulation NumPy exacte v31 n=12 B=%d L=%d : %d passages, %d groupes-9 (v30 : %d groupes-4)  fidelite vs Qiskit=%.12f %s"%(B,Lb,len(ps),sum(len(group9(l,B)) for _,l,_,_ in ps),sum(len(group4(l,B)) for _,l,_,_ in ps),f,"PASS" if f>1-1e-9 else "FAIL"))
    for n in (24,28,30):
        pr=compile_ops(n,circuit(n,20,n)); ps,_=schedule_rl(n,pr,12,3)
        P("groupes n=%d B=12 L=3 : v30 %d groupes-4 -> v31 %d groupes-9"%(n,sum(len(group4(l,12)) for _,l,_,_ in ps),sum(len(group9(l,12)) for _,l,_,_ in ps)))
    if not GPU: P("CuPy absent : arret apres validation logique."); return
    s31=Sim31(); s30=Sim30(); v21=GPUSim(); SW=(1,0,0)
    for n in (16,20,24):
        pr=compile_ops(n,circuit(n,20,n)); a=cp.asnumpy(v21.run(n,pr)).astype(np.complex128)
        for B in (12,13):
            ps,ph=schedule_rl(n,pr,B,3); b=to_logical(cp.asnumpy(s31.run(n,s31.upload(n,ps))).astype(np.complex128),n,ph); f=fid(a,b)
            P("validation GPU v31 B=%d vs GPU v21 FP64 n=%d : infidelite=%.2e  %s"%(B,n,1-f,"PASS" if abs(1-f)<1e-4 else "FAIL"))
        del a,b; cp.get_default_memory_pool().free_all_blocks()
    P("-"*104)
    free=cp.cuda.runtime.memGetInfo()[0]; best={}
    for n in (24,28,30):
        if (1<<n)*8>free*0.85: P("n=%-2d : VRAM insuffisante, saute"%n); continue
        pr=compile_ops(n,circuit(n,20,n))
        for B,Lb in ((12,2),(12,3),(13,3)):
            ps,ph=schedule_rl(n,pr,B,Lb); ref=None
            try:
                d30=s30.upload(n,ps); s30.run(20,s30.upload(20,schedule_rl(20,compile_ops(20,circuit(20,1,1)),B,Lb)[0]),SW)
                ref,psi=tmin(lambda:s30.run(n,d30,SW)); del psi,d30; cp.get_default_memory_pool().free_all_blocks()
                P("n=%-2d B=%d L=%d v30 SW (reference)  : %2d passages, %4d groupes-4 : %8.1f ms"%(n,B,Lb,len(ps),sum(len(group4(l,B)) for _,l,_,_ in ps),ref))
            except Exception as e: P("n=%-2d B=%d L=%d v30 : ECHEC %s"%(n,B,Lb,str(e)[:90]))
            try:
                d31=s31.upload(n,ps); s31.run(20,s31.upload(20,schedule_rl(20,compile_ops(20,circuit(20,1,1)),B,Lb)[0])); cp.cuda.Device().synchronize()
                ms,psi=tmin(lambda:s31.run(n,d31)); nm=norm1(psi); del psi
                mc,psi=tmin(lambda:s31.run(n,d31,True),2); del psi; cp.get_default_memory_pool().free_all_blocks()
                tag="B=%d L=%d"%(B,Lb)
                if ms<best.get(n,(1e30,""))[0]: best[n]=(ms,tag)
                P("n=%-2d %-10s v31 shuffle        : %2d passages, %4d groupes-9 : %8.1f ms (copie seule %7.1f ms, calcul %6.1f ms)  |norme-1|=%.1e  %s"%(n,tag,len(ps),sum(len(group9(l,B)) for _,l,_,_ in ps),ms,mc,ms-mc,nm,("vs v30 x%.2f"%(ref/ms)) if ref else ""))
            except Exception as e:
                P("n=%-2d B=%d L=%d v31 : ECHEC %s"%(n,B,Lb,str(e)[:90])); cp.get_default_memory_pool().free_all_blocks()
    P("-"*104); P("MEILLEUR V31 : "+"  ".join("%dq %.1f ms [%s]"%(n,best[n][0],best[n][1]) for n in sorted(best)))
    P("="*104)
    f=os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_gpu_v31_results.txt"); open(f,"w").write("\n".join(L)+"\n"); print("JOURNAL="+f)
if __name__=="__main__": main()
