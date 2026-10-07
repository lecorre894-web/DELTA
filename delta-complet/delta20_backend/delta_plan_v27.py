#!/usr/bin/env python3
"""DELTA PLAN COMPLET V27 — tout DELTA quantique emule sur RTX 4080, avec etiquettes honnetes.
[A] ETALON      : plafond VRAM reel de la 4080 (copie CuPy) -> efficacite de DELTA en % du materiel
[B] EXACT       : vecteur d'etat complet, intrication totale, v26 (registres + DAG + fusion), 20-30 qubits, fidelite vs v21 FP64
[C] STABILISEUR : GHZ (Clifford seulement) via stim, 1 000 -> 1 000 000 qubits, verification tous-0/tous-1
[D] PRODUIT 1T  : 10^12 qubits SANS INTRICATION, L couches RY+RZ par qubit, <Z> moyen ; rien n'est stocke
                  (16 To equivalents), tout est genere en registres ; verifie vs NumPy FP64 sur 10^6 qubits tires au hasard.
usage : python delta_plan_v27.py [N_produit=1e12] [L=4] [n_ghz_max=1e6]"""
import os, sys, math, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_arena_v20 import circuit
from delta_qpu_gpu_v21 import compile_ops, fid, GPU
NP=int(float(sys.argv[1])) if len(sys.argv)>1 else 10**12
LAY=int(sys.argv[2]) if len(sys.argv)>2 else 4
NG=int(float(sys.argv[3])) if len(sys.argv)>3 else 10**6
SEED=0xD317A
SRC=r'''
__device__ __forceinline__ unsigned fmix(unsigned x){ x^=x>>16; x*=0x85EBCA6Bu; x^=x>>13; x*=0xC2B2AE35u; x^=x>>16; return x; }
__device__ __forceinline__ float zq(unsigned long long q, int L, unsigned seed){
  unsigned lo=(unsigned)q, hi=(unsigned)(q>>32); float ar=1.f, ai=0.f, br=0.f, bi=0.f;
  for(int l=0;l<L;l++){
    unsigned h=fmix(lo ^ (hi*0x9E3779B1u) ^ ((unsigned)l*0x85EBCA6Bu) ^ seed), g=fmix(h ^ 0x68E31DA4u);
    float th=(float)h*1.4629180792671596e-09f, ph=(float)g*1.4629180792671596e-09f, s, c, sp, cp;
    __sincosf(0.5f*th,&s,&c);
    float nar=c*ar-s*br, nai=c*ai-s*bi, nbr=s*ar+c*br, nbi=s*ai+c*bi;
    __sincosf(0.5f*ph,&sp,&cp);
    ar=nar*cp+nai*sp; ai=nai*cp-nar*sp; br=nbr*cp-nbi*sp; bi=nbi*cp+nbr*sp;
  }
  return ar*ar+ai*ai-br*br-bi*bi;
}
extern "C" __global__ void prod(unsigned long long q0, unsigned long long n, int L, unsigned seed, double *out){
  float acc=0.f; double d=0.0; int k=0;
  for(unsigned long long i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x; i<n; i+=(unsigned long long)gridDim.x*blockDim.x){
    acc+=zq(q0+i,L,seed); if(++k==256){ d+=acc; acc=0.f; k=0; } }
  d+=acc;
  for(int o=16;o>0;o>>=1) d+=__shfl_down_sync(0xffffffffu,d,o);
  __shared__ double ws[32]; int lane=threadIdx.x&31, w=threadIdx.x>>5;
  if(lane==0) ws[w]=d; __syncthreads();
  if(w==0){ d=(lane<(int)(blockDim.x>>5))?ws[lane]:0.0; for(int o=16;o>0;o>>=1) d+=__shfl_down_sync(0xffffffffu,d,o); if(lane==0) atomicAdd(out,d); }
}
extern "C" __global__ void samp(const unsigned long long *Q, int m, int L, unsigned seed, float *z){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<m) z[i]=zq(Q[i],L,seed); }
'''
def fmix_np(x):
    x=x^(x>>np.uint32(16)); x=x*np.uint32(0x85EBCA6B); x=x^(x>>np.uint32(13)); x=x*np.uint32(0xC2B2AE35); return x^(x>>np.uint32(16))
def zq_np(q,L,seed):
    """reference NumPy FP64 du meme calcul (memes hachages, memes portes)"""
    lo=(q&np.uint64(0xffffffff)).astype(np.uint32); hi=(q>>np.uint64(32)).astype(np.uint32)
    a=np.ones(len(q),np.complex128); b=np.zeros(len(q),np.complex128); k=2*math.pi/2**32
    for l in range(L):
        h=fmix_np(lo^(hi*np.uint32(0x9E3779B1))^np.uint32((l*0x85EBCA6B)&0xffffffff)^np.uint32(seed)); g=fmix_np(h^np.uint32(0x68E31DA4))
        th=h.astype(np.float64)*k; ph=g.astype(np.float64)*k; c,s=np.cos(th/2),np.sin(th/2)
        a,b=c*a-s*b,s*a+c*b; a=a*np.exp(-0.5j*ph); b=b*np.exp(0.5j*ph)
    return np.abs(a)**2-np.abs(b)**2
def main():
    L=[]; P=lambda s:(print(s,flush=True),L.append(s))
    P("="*100); P("DELTA PLAN COMPLET V27 | etalon + exact + stabiliseur + produit 1T | RTX 4080 | etiquettes honnetes"); P("="*100)
    q=np.random.default_rng(1).integers(0,NP,200000,dtype=np.uint64); zr=zq_np(q,LAY,SEED)
    P("[D0] reference NumPy FP64 : %d qubits tires sur [0,%.0e) : <Z> moyen=%.5f  ecart-type=%.4f"%(len(q),NP,zr.mean(),zr.std()))
    if not GPU: P("CuPy absent : arret apres reference."); return
    import cupy as cp
    from delta_qpu_gpu_v21 import GPUSim
    from delta_qpu_gpu_v25 import schedule_dag
    from delta_qpu_gpu_v26 import RegSim
    dev=cp.cuda.Device(); nsm=dev.attributes["MultiProcessorCount"]; tm=lambda:cp.cuda.Event()
    # [A] ETALON
    x=cp.ones(1<<28,cp.float64); y=cp.empty_like(x); y[...]=x; dev.synchronize(); best=0
    for _ in range(5):
        e0,e1=tm(),tm(); e0.record(); y[...]=x; e1.record(); e1.synchronize(); best=max(best,2*x.nbytes/1e9/(cp.cuda.get_elapsed_time(e0,e1)/1e3))
    del x,y; cp.get_default_memory_pool().free_all_blocks(); VR=best
    P("[A] ETALON VRAM (copie 2 Go) : %.0f Go/s  (theorique 4080 ~716 Go/s) -> plafond de reference"%VR)
    # [B] EXACT
    rs=RegSim(); v21=GPUSim(); pr=compile_ops(20,circuit(20,20,20))
    f=fid(cp.asnumpy(v21.run(20,pr)).astype(np.complex128),cp.asnumpy(rs.run(20,rs.upload(20,schedule_dag(20,pr,12,2,True)))).astype(np.complex128))
    P("[B] EXACT validation 20 q v26 FP32 vs v21 FP64 : infidelite=%.1e %s"%(1-f,"PASS" if abs(1-f)<1e-4 else "FAIL"))
    free=cp.cuda.runtime.memGetInfo()[0]
    for n in (24,28,30):
        if (1<<n)*8>free*0.85: P("[B] %d q : VRAM insuffisante"%n); continue
        pr=compile_ops(n,circuit(n,20,n)); ps=schedule_dag(n,pr,12,2,True); d=rs.upload(n,ps); rs.run(n,rs.upload(n,schedule_dag(n,compile_ops(n,circuit(n,1,1)),12,2,True))); dev.synchronize()
        e0,e1=tm(),tm(); e0.record(); psi=rs.run(n,d); e1.record(); e1.synchronize(); ms=cp.cuda.get_elapsed_time(e0,e1)
        xx=psi.view(cp.float32); K=1<<26; nm=math.sqrt(sum(float(xx[i:i+K].dot(xx[i:i+K])) for i in range(0,xx.size,K))); del xx,psi; cp.get_default_memory_pool().free_all_blocks()
        bw=len(ps)*(1<<n)*16/1e9/(ms/1e3)
        P("[B] EXACT %d q (2^%d=%.1e amplitudes, intrication totale) : %d portes en %.1f ms  |norme-1|=%.0e  ~%.0f Go/s = %.0f%% du plafond VRAM"%(n,n,2.0**n,len(pr),ms,abs(nm-1),bw,100*bw/VR))
    # [C] STABILISEUR
    try:
        import stim
        n=1000
        while n<=NG:
            t=time.time(); c=stim.Circuit(); c.append("H",[0]); a=np.empty(2*(n-1),np.uint32); a[0::2]=np.arange(n-1); a[1::2]=np.arange(1,n)
            c.append("CX",a); c.append("M",np.arange(n,dtype=np.uint32)); r=c.compile_sampler(skip_reference_sample=True).sample(16)
            ok=all(row.all() or (~row).all() for row in r); P("[C] STABILISEUR GHZ %9d qubits (Clifford, CPU, stim) : 16 tirs tous-0/tous-1=%s (%d x tous-1) en %.2f s"%(n,"PASS" if ok else "FAIL",int(r[:,0].sum()),time.time()-t)); n*=10
    except ImportError: P("[C] stim absent -> .venv/bin/pip install stim puis relancer")
    # [D] PRODUIT 1T
    m=cp.RawModule(code=SRC); kp,ks=m.get_function("prod"),m.get_function("samp")
    Q=cp.asarray(q); z=cp.empty(len(q),cp.float32); ks(((len(q)+255)//256,),(256,),(Q,np.int32(len(q)),np.int32(LAY),np.uint32(SEED),z))
    err=float(np.abs(cp.asnumpy(z)-zr).max()); P("[D] PRODUIT verification GPU FP32 vs NumPy FP64 sur %d qubits : ecart max=%.1e %s"%(len(q),err,"PASS" if err<1e-3 else "FAIL"))
    out=cp.zeros(1,cp.float64); CH=1<<33; grid=(nsm*32,)
    kp(grid,(256,),(np.uint64(0),np.uint64(1<<24),np.int32(LAY),np.uint32(SEED),out)); dev.synchronize(); out[...]=0
    t=time.time(); e0,e1=tm(),tm(); e0.record(); s=0
    while s<NP:
        c=min(CH,NP-s); kp(grid,(256,),(np.uint64(s),np.uint64(c),np.int32(LAY),np.uint32(SEED),out)); s+=c
    e1.record(); e1.synchronize(); ms=cp.cuda.get_elapsed_time(e0,e1); mean=float(out[0])/NP
    sig=zr.std()/math.sqrt(len(q)); dz=abs(mean-zr.mean())
    P("[D] PRODUIT %.0e qubits x %d couches (RY+RZ) = %.1e portes : %.2f s  -> %.2e qubits/s, %.2e portes/s"%(NP,LAY,2.0*NP*LAY,ms/1e3,NP/(ms/1e3),2*NP*LAY/(ms/1e3)))
    P("[D] PRODUIT <Z> moyen GPU=%.6f vs echantillon FP64=%.6f : ecart %.1f sigma %s | memoire equivalente si stocke : %.0f To (rien stocke)"%(mean,zr.mean(),dz/sig,"PASS" if dz<5*sig else "FAIL",NP*16/1e12))
    P("-"*100)
    P("LECTURE HONNETE : [B] = vrai calcul quantique exact (intrication totale), plafond 30 q en 8 Go ;")
    P("[C] = circuits Clifford seulement (simulables classiquement en temps polynomial, Gottesman-Knill) ;")
    P("[D] = qubits independants, sans intrication : debit de portes, pas une puissance quantique. Seul [B] est dur classiquement.")
    P("="*100)
    fn=os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_plan_v27_results.txt"); open(fn,"w").write("\n".join(L)+"\n"); print("JOURNAL="+fn)
if __name__=="__main__": main()
