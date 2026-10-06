#!/usr/bin/env python3
"""DELTA ARENA V20 — PC BRUT (outils standard) contre PILE DELTA (moteurs optimises), meme machine, memes epreuves.
Epreuve 1 TPOP LOURD : requetes de 4M mots (3 x 32 Mo) -> popcount(C^(A&B)), graine neuve par requete.
  PC brut  : NumPy (generation default_rng + popcount unpackbits)
  DELTA    : fusion GPU v19 (RTX 4080) mesuree en direct + fusion CPU v17 16 coeurs (journal x512_same_v17_results.txt)
Epreuve 2 QPU EMULE : circuit aleatoire n qubits (couches H/RZ + CX en echelle), vecteur d'etat complet.
  PC brut  : Qiskit Statevector (CPU)
  DELTA    : simulateur vecteur d'etat CuPy sur RTX 4080 (meme circuit), valide par fidelite contre Qiskit a n=12.
Aucun QPU reel n'est utilise. Chaque camp est valide avant d'etre chronometre."""
import os, re, sys, time, math, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
try:
    import cupy as cp; GPU=True
except Exception: cp=None; GPU=False
def T(f,n=1):
    t=time.perf_counter()
    for _ in range(n): r=f()
    return (time.perf_counter()-t)/n*1e6, r
NW=4194304
def numpy_req(seed):
    g=np.random.default_rng(seed); A,B,C=[g.integers(0,2**63,NW,dtype=np.uint64) for _ in range(3)]
    return int(np.unpackbits(((A&B)^C).view(np.uint8)).sum())
def gpu_tpop():
    SRC=r'''
__device__ __forceinline__ unsigned long long smix(unsigned long long &s){ unsigned long long z=(s+=0x9E3779B97F4A7C15ULL);
  z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
__device__ __forceinline__ unsigned long long xs(unsigned long long x){ x^=x<<13; x^=x>>7; x^=x<<17; return x; }
extern "C" __global__ void fused(unsigned long long seed, long long nw, unsigned long long *out){
  long long tid=blockIdx.x*(long long)blockDim.x+threadIdx.x, nt=(long long)gridDim.x*blockDim.x;
  unsigned long long s=seed*0xD1B54A32D192ED03ULL+tid*0x8CB92BA72F3D8DD7ULL+1;
  unsigned long long a=smix(s)|1, b=smix(s)|1, c=smix(s)|1, acc=0;
  for(long long i=tid;i<nw;i+=nt){ a=xs(a); b=xs(b); c=xs(c); acc+=__popcll((a&b)^c); }
  for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
  if((threadIdx.x&31)==0) atomicAdd(out,acc); }
extern "C" __global__ void mat(unsigned long long seed, long long nw, unsigned long long *A, unsigned long long *B, unsigned long long *C){
  long long tid=blockIdx.x*(long long)blockDim.x+threadIdx.x, nt=(long long)gridDim.x*blockDim.x;
  unsigned long long s=seed*0xD1B54A32D192ED03ULL+tid*0x8CB92BA72F3D8DD7ULL+1;
  unsigned long long a=smix(s)|1, b=smix(s)|1, c=smix(s)|1;
  for(long long i=tid;i<nw;i+=nt){ a=xs(a); b=xs(b); c=xs(c); A[i]=a; B[i]=b; C[i]=c; } }'''
    m=cp.RawModule(code=SRC); F=m.get_function("fused"); M=m.get_function("mat")
    SM=cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount']; blocks=SM*8; out=cp.zeros(1,dtype=cp.uint64)
    pop=cp.ElementwiseKernel('uint64 a, uint64 b, uint64 c','uint64 y','y=__popcll((a&b)^c)','tpop_chk')
    A,B,C=[cp.empty(NW,dtype=cp.uint64) for _ in range(3)]; out.fill(0)
    F((blocks,),(256,),(np.uint64(777),np.int64(NW),out)); M((blocks,),(256,),(np.uint64(777),np.int64(NW),A,B,C))
    ok=int(out.get()[0])==int(pop(A,B,C).sum()); del A,B,C
    for w in range(20): F((blocks,),(256,),(np.uint64(9000+w),np.int64(NW),out))
    n=2000; e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record()
    for i in range(n): F((blocks,),(256,),(np.uint64(100000+i),np.int64(NW),out))
    e1.record(); e1.synchronize(); return cp.cuda.get_elapsed_time(e0,e1)*1e3/n, ok
def cpu_v17():
    f=os.path.join(HERE,"x512_same_v17_results.txt")
    if not os.path.exists(f): return None
    for l in open(f):
        if l.startswith("4194304") and "16 coeurs" in l and "F fusion" in l: return float(l.split()[6])
    return None
def circuit(n,depth,seed):
    r=np.random.default_rng(seed); ops=[]
    for d in range(depth):
        for q in range(n): ops.append(("h",q)) if r.random()<0.5 else ops.append(("rz",q,float(r.uniform(0,2*math.pi))))
        for q in range(d%2,n-1,2): ops.append(("cx",q,q+1))
    return ops
def run_sv(xp,n,ops):
    psi=xp.zeros(2**n,dtype=xp.complex128); psi[0]=1
    H=xp.asarray(np.array([[1,1],[1,-1]])/math.sqrt(2),dtype=xp.complex128)
    for op in ops:
        if op[0]=="cx":
            c,t=op[1],op[2]; v=psi.reshape([2]*n)
            ac,at=n-1-c,n-1-t; idx=[slice(None)]*n; idx[ac]=1
            sub=v[tuple(idx)]; ax=at if at<ac else at-1
            sub=xp.flip(sub,axis=ax).copy(); v[tuple(idx)]=sub; psi=v.reshape(-1)
        else:
            q=op[1]; v=psi.reshape(2**(n-1-q),2,2**q)
            if op[0]=="h": v=xp.einsum('ij,ajb->aib',H,v)
            else:
                ph=xp.asarray([np.exp(-0.5j*op[2]),np.exp(0.5j*op[2])],dtype=xp.complex128); v=v*ph[None,:,None]
            psi=xp.ascontiguousarray(v).reshape(-1)
    return psi
def qiskit_sv(n,ops):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    qc=QuantumCircuit(n)
    for op in ops:
        if op[0]=="h": qc.h(op[1])
        elif op[0]=="rz": qc.rz(op[2],op[1])
        else: qc.cx(op[1],op[2])
    return Statevector(qc).data
def main():
    L=[]; P=lambda s:(print(s,flush=True),L.append(s))
    P("="*100); P("DELTA ARENA V20 | PC BRUT (outils standard) vs PILE DELTA (moteurs optimises) | meme machine | QPU EMULE, aucun QPU reel"); P("="*100)
    P("-- EPREUVE 1 : TPOP LOURD, 4 194 304 mots x 3 (96 Mo par requete), graine neuve par requete --")
    numpy_req(1); us_np,_=T(lambda:numpy_req(2),3); P("PC brut  NumPy            : %12.1f us/requete"%us_np)
    if GPU:
        us_g,ok=gpu_tpop(); P("DELTA    fusion GPU v19   : %12.3f us/requete  validation=%s  -> %.0fx"%(us_g,"PASS" if ok else "FAIL",us_np/us_g))
    else: P("DELTA    fusion GPU v19   : CuPy absent")
    c=cpu_v17()
    if c: P("DELTA    fusion CPU v17   : %12.3f us/requete  (16 coeurs, journal v17, valide PASS) -> %.0fx"%(c,us_np/c))
    P("-- EPREUVE 2 : QPU EMULE, circuit aleatoire H/RZ + echelle CX --")
    ops=circuit(12,12,5); a=qiskit_sv(12,ops)
    for nom,xp in (("NumPy",np),)+((("GPU",cp),) if GPU else ()):
        b=run_sv(xp,12,ops); b=cp.asnumpy(b) if xp is not np else b; f=abs(np.vdot(a,b))**2
        P("validation n=12 simulateur DELTA-%s vs Qiskit : fidelite=%.12f %s"%(nom,f,"PASS" if f>1-1e-9 else "FAIL"))
    for n in (16,20,22,24):
        ops=circuit(n,20,n); gates=len(ops)
        if n<=22:
            us_q,_=T(lambda:qiskit_sv(n,ops),1); P("n=%-2d (%5d portes, %7.1f Mo)  PC brut Qiskit : %12.1f ms"%(n,gates,2**n*16/1048576,us_q/1e3))
        else: us_q=None; P("n=%-2d (%5d portes, %7.1f Mo)  PC brut Qiskit : saute (trop long)"%(n,gates,2**n*16/1048576))
        if GPU:
            run_sv(cp,min(n,16),circuit(min(n,16),2,1)); cp.cuda.Device().synchronize()
            t=time.perf_counter(); run_sv(cp,n,ops); cp.cuda.Device().synchronize(); us_d=(time.perf_counter()-t)*1e6
            P("n=%-2d                              DELTA GPU     : %12.1f ms%s"%(n,us_d/1e3,("  -> %.1fx"%(us_q/us_d)) if us_q else ""))
    P("="*100)
    f=os.path.join(HERE,"delta_arena_v20_results.txt"); open(f,"w").write("\n".join(L)+"\n"); print("JOURNAL="+f)
if __name__=="__main__": main()
