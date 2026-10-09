#!/usr/bin/env python3
"""DELTA QPU EMULE V32 — test court : d'ou viennent les ~394 ms de calcul a 30 q ? (RTX 4080).
Calcul : 30 q avec fusion = 290 portes 4x4 = ~9300 flops/amplitude x 2^30 = 1,0e13 flops -> ~200 ms au plafond FP32 (~50 TFLOPS).
On mesure ~394 ms = ~50 % du plafond. Deux interrupteurs :
CM (memoire constante) : les coefficients des portes sont lus dans le cache constant (diffusion a tout le warp) au lieu de la memoire globale.
FZ (fusion) : FZ=1 = portes fusionnees 4x4 (v25-v31) ; FZ=0 = portes simples 2x2 + CX (echange sans calcul) = 12 % de flops en moins.
Base : noyau v30 swizzle (record). Validation GPU vs v21 FP64 (n=16, 20)."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_arena_v20 import circuit
from delta_qpu_gpu_v21 import compile_ops, fid, GPU
from delta_qpu_gpu_v29 import schedule_rl, to_logical
from delta_qpu_gpu_v30 import KERN as K30, Sim30
if GPU:
    import cupy as cp
    from delta_qpu_gpu_v21 import GPUSim
NC=16000
KERN="__constant__ float Gc[%d];\n"%NC+K30.replace("template<int SW,int PF,int CA> __global__","template<int SW,int PF,int CA,int CMM> __global__",1).replace("const float *d=G+35*q;","const float *d=CMM?(Gc+35*q):(G+35*q);",1)
assert "CMM?(Gc" in KERN
CF=[(1,0,0,0),(1,0,0,1)]
class Sim32(Sim30):
    def __init__(s):
        m=cp.RawModule(code=KERN,options=("-std=c++14",),name_expressions=["blk<%d,%d,%d,%d>"%c for c in CF])
        s.k={c:m.get_function("blk<%d,%d,%d,%d>"%c) for c in CF}
        for f in s.k.values():
            try: f.max_dynamic_shared_size_bytes=65536
            except Exception as e: print("note: opt-in 64 Ko indisponible (%s)"%e)
        s.gc=cp.ndarray((NC,),cp.float32,m.get_global("Gc"))
    def run(s,n,dev,c,copy=False):
        psi=cp.zeros(1<<n,dtype=cp.complex64); psi[0]=1
        for B,OFF,C,ngr,GR,G,OFF2 in dev:
            if c[3]:
                assert G.size<=NC, "trop de portes pour la memoire constante"
                s.gc[:G.size]=G.reshape(-1)
            s.k[c]((1<<(n-B),),(1<<(B-4),),(psi,np.int32(n),np.int32(B),OFF,C,np.int32(0 if copy else ngr),GR,G,OFF2),shared_mem=(1<<B)*8)
        return psi
def tmin(f,it=3):
    ms=1e30; r=None
    for _ in range(it):
        r=None; cp.get_default_memory_pool().free_all_blocks()
        e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(); r=f(); e1.record(); e1.synchronize(); ms=min(ms,cp.cuda.get_elapsed_time(e0,e1))
    return ms,r
def main():
    L=[]; P=lambda s:(print(s,flush=True),L.append(s))
    P("="*104); P("DELTA QPU EMULE V32 | v30 swizzle + CM (coefficients en memoire constante) x FZ (fusion on/off) | RTX 4080"); P("="*104)
    if not GPU: P("CuPy absent : ce banc demande la RTX 4080."); return
    s=Sim32(); v21=GPUSim()
    for n in (16,20):
        pr=compile_ops(n,circuit(n,20,n)); a=cp.asnumpy(v21.run(n,pr)).astype(np.complex128)
        for fz in (1,0):
            ps,ph=schedule_rl(n,pr,12,3,bool(fz)); dev=s.upload(n,ps)
            for c in CF:
                b=to_logical(cp.asnumpy(s.run(n,dev,c)).astype(np.complex128),n,ph); f=fid(a,b)
                P("validation n=%d FZ=%d CM=%d vs v21 FP64 : infidelite=%.2e  %s"%(n,fz,c[3],1-f,"PASS" if abs(1-f)<1e-4 else "FAIL"))
        cp.get_default_memory_pool().free_all_blocks()
    P("-"*104); best={}
    for n in (24,28,30):
        pr=compile_ops(n,circuit(n,20,n)); ref=None
        for fz in (1,0):
            ps,ph=schedule_rl(n,pr,12,3,bool(fz)); dev=s.upload(n,ps); s.run(20,s.upload(20,schedule_rl(20,compile_ops(20,circuit(20,1,1)),12,3,bool(fz))[0]),CF[1])
            for c in CF:
                try:
                    ms,psi=tmin(lambda:s.run(n,dev,c)); del psi
                    mc,psi=tmin(lambda:s.run(n,dev,c,True),2); del psi; cp.get_default_memory_pool().free_all_blocks()
                    if fz==1 and c[3]==0: ref=ms
                    tag="FZ=%d CM=%d"%(fz,c[3])
                    if ms<best.get(n,(1e30,""))[0]: best[n]=(ms,tag)
                    P("n=%-2d %-10s : %2d passages, %4d portes : %8.1f ms (copie %7.1f ms, calcul %6.1f ms)  %s"%(n,tag,len(ps),sum(len(l) for _,l,_,_ in ps),ms,mc,ms-mc,"(= v30 SW)" if (fz==1 and c[3]==0) else ("vs v30 x%.2f"%(ref/ms) if ref else "")))
                except Exception as e: P("n=%-2d FZ=%d CM=%d : ECHEC %s"%(n,fz,c[3],str(e)[:90]))
    P("-"*104); P("MEILLEUR V32 : "+"  ".join("%dq %.1f ms [%s]"%(n,best[n][0],best[n][1]) for n in sorted(best)))
    P("="*104)
    f=os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_gpu_v32_results.txt"); open(f,"w").write("\n".join(L)+"\n"); print("JOURNAL="+f)
if __name__=="__main__": main()
