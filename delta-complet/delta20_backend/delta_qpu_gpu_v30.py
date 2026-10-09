#!/usr/bin/env python3
"""DELTA QPU EMULE V30 — v29 + 3 interrupteurs mesures separement et combines (RTX 4080).
Lecons v29 : ~520 ms a 30 q restent constants quel que soit le nombre de passages = trafic memoire partagee des groupes.
Modele hors GPU : conflits de bancs moyens 3,9 voies -> 1,4 avec un melange d'adresses XOR.
SW (swizzle)    : toute adresse en memoire partagee x devient x^((x>>4)&15) (bijection, meme resultat, moins de files au guichet).
PF (prechargement) : chaque thread lance ses 16 lectures VRAM d'un coup en registres avant d'ecrire au guichet (plus de lectures en vol).
CA (cp.async)   : copie VRAM -> memoire partagee par le moteur asynchrone (sans passer par les registres).
Combinaisons : 000 = v29 exacte (reference rechronometree), puis chaque interrupteur seul et combine.
Validation : chaque combinaison GPU FP32 vs GPU v21 FP64 (n=16, 20) apres remise en ordre logique."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_arena_v20 import circuit
from delta_qpu_gpu_v21 import compile_ops, fid, GPU
from delta_qpu_gpu_v26 import SRC as SRC26, group, RW
from delta_qpu_gpu_v29 import schedule_rl, to_logical, offs
if GPU:
    import cupy as cp
    from delta_qpu_gpu_v21 import GPUSim
KERN=SRC26[:SRC26.index('extern "C" __global__ void blk')]+r'''
#define SWZ(x) (SW ? ((x)^(((x)>>4)&15)) : (x))
template<int SW,int PF,int CA> __global__ void blk(float2 *psi, int n, int B, const long long *OFF, const int *Cg, int ngr, const int *GR, const float *G, const long long *OFF2){
  extern __shared__ float2 sh[];
  __shared__ int C[48]; __shared__ long long boff_s;
  if(threadIdx.x<n-B) C[threadIdx.x]=Cg[threadIdx.x];
  __syncthreads();
  if(threadIdx.x==0){ long long base=blockIdx.x, b=0; for(int k=0;k<n-B;k++) if((base>>k)&1LL) b|=1LL<<C[k]; boff_s=b; }
  __syncthreads();
  const long long boff=boff_s; const int M=1<<B, T=blockDim.x, t=threadIdx.x;
  if(CA){
    for(int j=t;j<M;j+=T){ unsigned s=(unsigned)__cvta_generic_to_shared(&sh[SWZ(j)]); const float2 *g=psi+(boff|OFF[j]);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(g)); }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
  } else if(PF){
    float2 r[16];
    #pragma unroll
    for(int k=0;k<16;k++) r[k]=psi[boff|OFF[t+k*T]];
    #pragma unroll
    for(int k=0;k<16;k++){ int j=t+k*T; sh[SWZ(j)]=r[k]; }
  } else {
    for(int j=t;j<M;j+=T) sh[SWZ(j)]=psi[boff|OFF[j]];
  }
  __syncthreads();
  for(int g=0;g<ngr;g++){
    const int *h=GR+10*g; int base=t;
    #pragma unroll
    for(int j=0;j<4;j++){ int s=h[4+j]; base=((base>>s)<<(s+1))|(base&((1<<s)-1)); }
    int o0=1<<h[0], o1=1<<h[1], o2=1<<h[2], o3=1<<h[3]; float2 v[16];
    #pragma unroll
    for(int k=0;k<16;k++){ int x=base|((k&1)?o0:0)|((k&2)?o1:0)|((k&4)?o2:0)|((k&8)?o3:0); v[k]=sh[SWZ(x)]; }
    for(int q=h[8];q<h[8]+h[9];q++){
      const float *d=G+35*q; int ty=(int)d[0], a=(int)d[1], b=(int)d[2];
      if(ty==0){ switch(a){ case 0: g1<0>(v,d); break; case 1: g1<1>(v,d); break; case 2: g1<2>(v,d); break; default: g1<3>(v,d); } }
      else switch(a*4+b){ C2(0,1) C2(0,2) C2(0,3) C2(1,0) C2(1,2) C2(1,3) C2(2,0) C2(2,1) C2(2,3) C2(3,0) C2(3,1) C2(3,2) }
    }
    #pragma unroll
    for(int k=0;k<16;k++){ int x=base|((k&1)?o0:0)|((k&2)?o1:0)|((k&4)?o2:0)|((k&8)?o3:0); sh[SWZ(x)]=v[k]; }
    __syncthreads();
  }
  if(PF){
    float2 r[16];
    #pragma unroll
    for(int k=0;k<16;k++){ int j=t+k*T; r[k]=sh[SWZ(j)]; }
    #pragma unroll
    for(int k=0;k<16;k++) psi[boff|OFF2[t+k*T]]=r[k];
  } else {
    for(int j=t;j<M;j+=T) psi[boff|OFF2[j]]=sh[SWZ(j)];
  }
}
'''
COMBOS=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(1,1,1)]
NAME=lambda c:"SW=%d PF=%d CA=%d"%c
class Sim30:
    def __init__(s):
        m=cp.RawModule(code=KERN,options=("-std=c++14",),name_expressions=["blk<%d,%d,%d>"%c for c in COMBOS])
        s.k={c:m.get_function("blk<%d,%d,%d>"%c) for c in COMBOS}
        for f in s.k.values():
            try: f.max_dynamic_shared_size_bytes=65536
            except Exception as e: print("note: opt-in 64 Ko indisponible (%s)"%e)
    def upload(s,n,passes):
        out=[]
        for Sl,loc,P,P2 in passes:
            B=len(P); C=[p for p in range(n) if p not in P]; grs=group(loc,B); rows=[]; GRi=[]
            for R,gs in grs:
                pos={q:j for j,q in enumerate(R)}; GRi.append(R+sorted(R)+[len(rows),len(gs)])
                for g in gs:
                    r=np.zeros(RW)
                    if g[0]=="u": M=g[2]; r[:11]=[0,pos[g[1]],0,M[0,0].real,M[0,0].imag,M[0,1].real,M[0,1].imag,M[1,0].real,M[1,0].imag,M[1,1].real,M[1,1].imag]
                    elif g[0]=="cx": r[:3]=[1,pos[g[1]],pos[g[2]]]
                    else: r[:3]=[2,pos[g[1]],pos[g[2]]]; r[3:]=np.stack([g[3].real,g[3].imag],-1).reshape(-1)
                    rows.append(r)
            out.append((B,cp.asarray(offs(P)),cp.asarray(np.array(C or [0],np.int32)),len(grs),cp.asarray(np.array(GRi,np.int32)),cp.asarray(np.array(rows,np.float32)),cp.asarray(offs(P2))))
        return out
    def run(s,n,dev,c,copy=False):
        psi=cp.zeros(1<<n,dtype=cp.complex64); psi[0]=1
        for B,OFF,C,ngr,GR,G,OFF2 in dev:
            s.k[c]((1<<(n-B),),(1<<(B-4),),(psi,np.int32(n),np.int32(B),OFF,C,np.int32(0 if copy else ngr),GR,G,OFF2),shared_mem=(1<<B)*8)
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
    P("="*104); P("DELTA QPU EMULE V30 | v29 + interrupteurs SW (swizzle) / PF (prechargement registres) / CA (cp.async) | RTX 4080"); P("="*104)
    if not GPU: P("CuPy absent : ce banc demande la RTX 4080."); return
    s=Sim30(); v21=GPUSim()
    for n in (16,20):
        pr=compile_ops(n,circuit(n,20,n)); a=cp.asnumpy(v21.run(n,pr)).astype(np.complex128); ps,ph=schedule_rl(n,pr,12,3); dev=s.upload(n,ps)
        for c in COMBOS:
            b=to_logical(cp.asnumpy(s.run(n,dev,c)).astype(np.complex128),n,ph); f=fid(a,b)
            P("validation n=%d %-16s vs v21 FP64 : infidelite=%.2e  %s"%(n,NAME(c),1-f,"PASS" if abs(1-f)<1e-4 else "FAIL"))
        del a,b; cp.get_default_memory_pool().free_all_blocks()
    P("-"*104)
    free=cp.cuda.runtime.memGetInfo()[0]; best={}
    for n in (24,28,30):
        if (1<<n)*8>free*0.85: P("n=%-2d : VRAM insuffisante, saute"%n); continue
        pr=compile_ops(n,circuit(n,20,n))
        for Lb in (2,3):
            ps,ph=schedule_rl(n,pr,12,Lb); dev=s.upload(n,ps); ref=None
            for c in COMBOS:
                try:
                    s.run(20,s.upload(20,schedule_rl(20,compile_ops(20,circuit(20,1,1)),12,Lb)[0]),c); cp.cuda.Device().synchronize()
                    ms,psi=tmin(lambda:s.run(n,dev,c)); nm=norm1(psi); del psi
                    mc,psi=tmin(lambda:s.run(n,dev,c,True),2); del psi; cp.get_default_memory_pool().free_all_blocks()
                    if c==(0,0,0): ref=ms
                    tag="L=%d %s"%(Lb,NAME(c))
                    if ms<best.get(n,(1e30,""))[0]: best[n]=(ms,tag)
                    P("n=%-2d %-24s: %2d passages : %8.1f ms (copie seule %7.1f ms = %3.0f%%)  |norme-1|=%.1e  %s"%(n,tag,len(ps),ms,mc,100*mc/ms,nm,"(= v29)" if c==(0,0,0) else ("vs v29 x%.2f"%(ref/ms) if ref else "")))
                except Exception as e:
                    P("n=%-2d L=%d %s : ECHEC %s"%(n,Lb,NAME(c),str(e)[:90])); cp.get_default_memory_pool().free_all_blocks()
    P("-"*104); P("MEILLEUR V30 : "+"  ".join("%dq %.1f ms [%s]"%(n,best[n][0],best[n][1]) for n in sorted(best)))
    P("LECTURE : si SW seul gagne -> conflits de bancs confirmes ; si PF/CA gagnent surtout sur 'copie seule' -> latence VRAM ; sinon goulot ailleurs (calcul des portes).")
    P("="*104)
    f=os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_gpu_v30_results.txt"); open(f,"w").write("\n".join(L)+"\n"); print("JOURNAL="+f)
if __name__=="__main__": main()
