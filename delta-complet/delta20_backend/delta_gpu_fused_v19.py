#!/usr/bin/env python3
"""DELTA X512 V19 — FUSION GENERATION+TPOP SUR RTX 4080 (CuPy RawKernel)
Meme travail que v17/v18 : requete = graine neuve -> A,B,C de NW mots -> popcount(C ^ (A & B)).
Chaque thread CUDA possede ses 3 flux xorshift64 (amorces par splitmix64(graine, thread)), genere et consomme en registres.
Validation : un noyau jumeau ECRIT les memes A,B,C en VRAM, puis CuPy recalcule popcount((A&B)^C) -> doit egaler la fusion.
Le generateur differe de celui du CPU (v14-v18) : le travail est identique en taille, pas en valeurs."""
import time, numpy as np, cupy as cp
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
  for(long long i=tid;i<nw;i+=nt){ a=xs(a); b=xs(b); c=xs(c); A[i]=a; B[i]=b; C[i]=c; } }
'''
mod=cp.RawModule(code=SRC); F=mod.get_function("fused"); M=mod.get_function("mat")
dev=cp.cuda.Device(); props=cp.cuda.runtime.getDeviceProperties(0); SM=props['multiProcessorCount']
pop=cp.ElementwiseKernel('uint64 a, uint64 b, uint64 c','uint64 y','y=__popcll((a&b)^c)','tpop_chk')
print("DELTA X512 V19 FUSION GPU | %s | %d SM | requete = graine neuve -> 3 x NW mots -> popcount(C^(A&B))"%(props['name'].decode(),SM))
print("%-10s %-8s %-10s %12s %12s %12s %8s"%("NW_mots","Mo_eq","GRILLE","US/REQ","REQ/S","Gbit/s","VALID"))
out=cp.zeros(1,dtype=cp.uint64); TPB=256
for NW in (65536,262144,1048576,4194304,16777216,67108864):
    blocks=min(SM*8, max(1,(NW+TPB-1)//TPB))
    A,B,C=[cp.empty(NW,dtype=cp.uint64) for _ in range(3)]; ok=True
    for sd in (777,778,779):
        out.fill(0); F((blocks,),(TPB,),(np.uint64(sd),np.int64(NW),out)); r=int(out.get()[0])
        M((blocks,),(TPB,),(np.uint64(sd),np.int64(NW),A,B,C)); ok&=(r==int(pop(A,B,C).sum()))
    del A,B,C
    for w in range(20): F((blocks,),(TPB,),(np.uint64(9000+w),np.int64(NW),out))
    dev.synchronize(); n=max(50,min(20000,int(2e5*65536/NW))); best=1e30
    for rep in range(3):
        e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record()
        for i in range(n): F((blocks,),(TPB,),(np.uint64(100000+rep*n+i),np.int64(NW),out))
        e1.record(); e1.synchronize(); us=cp.cuda.get_elapsed_time(e0,e1)*1e3/n; best=min(best,us)
    print("%-10d %-8.1f %-10s %12.3f %12.0f %12.1f %8s"%(NW,3*NW*8/1048576,"%dx%d"%(blocks,TPB),best,1e6/best,3*NW*64/(best*1e3),"PASS" if ok else "FAIL"),flush=True)
print("Note : US/REQ = temps GPU par requete, lancements en file (sans aller-retour Python par requete).")
print("Reference CPU (v17, fusion 16 coeurs) : 65536=1.212 us | 262144=3.084 | 1048576=10.681 | 4194304=40.964")
