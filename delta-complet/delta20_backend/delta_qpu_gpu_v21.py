#!/usr/bin/env python3
"""DELTA QPU EMULE V21 — simulateur vecteur d'etat en VRAIS noyaux CUDA (CuPy RawKernel), RTX 4080.
Par rapport a v20 (une operation NumPy + copies par porte) :
  - portes appliquees EN PLACE (aucune copie du vecteur d'etat)
  - fusion des portes 1 qubit consecutives (H, RZ) en une seule matrice 2x2 par qubit, videe avant chaque CX
  - CX = echange d'amplitudes en place, N/4 threads
Meme circuit que v20 (circuit() importe de delta_arena_v20.py). Validation : fidelite vs Qiskit (n=12) et vs v20 GPU (n=16).
Ordre des qubits : petit-boutiste (qubit q = bit q de l'indice), comme Qiskit."""
import os, sys, time, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_arena_v20 import circuit, run_sv, qiskit_sv
try:
    import cupy as cp; GPU=True
except Exception: cp=None; GPU=False
SRC=r'''
extern "C" __global__ void g1(double2 *psi, long long half, int q,
  double m00r,double m00i,double m01r,double m01i,double m10r,double m10i,double m11r,double m11i){
  long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x; if(i>=half) return;
  long long lo=i&((1LL<<q)-1), i0=((i>>q)<<(q+1))|lo, i1=i0|(1LL<<q);
  double2 a=psi[i0], b=psi[i1], r0, r1;
  r0.x=m00r*a.x-m00i*a.y+m01r*b.x-m01i*b.y; r0.y=m00r*a.y+m00i*a.x+m01r*b.y+m01i*b.x;
  r1.x=m10r*a.x-m10i*a.y+m11r*b.x-m11i*b.y; r1.y=m10r*a.y+m10i*a.x+m11r*b.y+m11i*b.x;
  psi[i0]=r0; psi[i1]=r1; }
extern "C" __global__ void cx(double2 *psi, long long quarter, int c, int t){
  long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x; if(i>=quarter) return;
  int lo=c<t?c:t, hi=c<t?t:c;
  long long x=((i>>lo)<<(lo+1))|(i&((1LL<<lo)-1));
  x=((x>>hi)<<(hi+1))|(x&((1LL<<hi)-1));
  long long i0=x|(1LL<<c), i1=i0|(1LL<<t);
  double2 tmp=psi[i0]; psi[i0]=psi[i1]; psi[i1]=tmp; }
'''
H=np.array([[1,1],[1,-1]],dtype=np.complex128)/math.sqrt(2)
def rz(th): return np.diag([np.exp(-0.5j*th),np.exp(0.5j*th)]).astype(np.complex128)
def compile_ops(n,ops):
    pend={}; out=[]
    def flush(q):
        if q in pend: out.append(("u",q,pend.pop(q)))
    for op in ops:
        if op[0]=="cx": flush(op[1]); flush(op[2]); out.append(op)
        else:
            M=H if op[0]=="h" else rz(op[2]); q=op[1]; pend[q]=M@pend.get(q,np.eye(2,dtype=np.complex128))
    for q in sorted(list(pend)): flush(q)
    return out
def run_numpy_emul(n,prog):
    N=1<<n; psi=np.zeros(N,np.complex128); psi[0]=1
    for g in prog:
        if g[0]=="u":
            q,M=g[1],g[2]; i=np.arange(N//2,dtype=np.int64); lo=i&((1<<q)-1); i0=((i>>q)<<(q+1))|lo; i1=i0|(1<<q)
            a,b=psi[i0].copy(),psi[i1].copy(); psi[i0]=M[0,0]*a+M[0,1]*b; psi[i1]=M[1,0]*a+M[1,1]*b
        else:
            c,t=g[1],g[2]; i=np.arange(N//4,dtype=np.int64); lo,hi=min(c,t),max(c,t)
            x=((i>>lo)<<(lo+1))|(i&((1<<lo)-1)); x=((x>>hi)<<(hi+1))|(x&((1<<hi)-1))
            i0=x|(1<<c); i1=i0|(1<<t); tmp=psi[i0].copy(); psi[i0]=psi[i1]; psi[i1]=tmp
    return psi
class GPUSim:
    def __init__(s):
        m=cp.RawModule(code=SRC)
        s.g1=m.get_function("g1"); s.cx=m.get_function("cx")
    def run(s,n,prog):
        N=1<<n; psi=cp.zeros(N,dtype=cp.complex128); psi[0]=1; T=256
        bh=((N//2)+T-1)//T; bq=((N//4)+T-1)//T
        for g in prog:
            if g[0]=="u":
                M=g[2]; s.g1((bh,),(T,),(psi,np.int64(N//2),np.int32(g[1]),
                    np.float64(M[0,0].real),np.float64(M[0,0].imag),np.float64(M[0,1].real),np.float64(M[0,1].imag),
                    np.float64(M[1,0].real),np.float64(M[1,0].imag),np.float64(M[1,1].real),np.float64(M[1,1].imag)))
            else: s.cx((bq,),(T,),(psi,np.int64(N//4),np.int32(g[1]),np.int32(g[2])))
        return psi
def fid(a,b): return abs(np.vdot(a,b))**2
def main():
    L=[]; P=lambda s:(print(s,flush=True),L.append(s))
    P("="*96); P("DELTA QPU EMULE V21 | vrais noyaux CUDA en place + fusion des portes 1 qubit | RTX 4080"); P("="*96)
    ops=circuit(12,12,5); prog=compile_ops(12,ops); ref=qiskit_sv(12,ops)
    P("n=12 : %d portes -> %d operations apres fusion"%(len(ops),len(prog)))
    P("validation emulation NumPy des noyaux vs Qiskit : fidelite=%.12f %s"%(fid(ref,run_numpy_emul(12,prog)),"PASS" if fid(ref,run_numpy_emul(12,prog))>1-1e-9 else "FAIL"))
    if not GPU: P("CuPy absent : arret apres validation logique."); return
    sim=GPUSim(); g=cp.asnumpy(sim.run(12,prog)); f=fid(ref,g)
    P("validation GPU v21 vs Qiskit n=12 : fidelite=%.12f %s"%(f,"PASS" if f>1-1e-9 else "FAIL"))
    ops16=circuit(16,20,16); a=cp.asnumpy(run_sv(cp,16,ops16)); b=cp.asnumpy(sim.run(16,compile_ops(16,ops16)))
    P("validation GPU v21 vs GPU v20 n=16 : fidelite=%.12f %s"%(fid(a,b),"PASS" if fid(a,b)>1-1e-9 else "FAIL"))
    V20={16:29.6,20:377.4,22:1912.0,24:7107.7}; QK={16:47.0,20:2560.8,22:12573.5}
    free=cp.cuda.runtime.memGetInfo()[0]
    for n in (16,20,22,24,26,28,29):
        need=(1<<n)*16
        if need>free*0.85: P("n=%-2d : %.1f Go necessaires > VRAM libre, saute"%(n,need/2**30)); continue
        ops=circuit(n,20,n); prog=compile_ops(n,ops)
        sim.run(min(n,14),compile_ops(min(n,14),circuit(min(n,14),2,1))); cp.cuda.Device().synchronize()
        e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(); psi=sim.run(n,prog); e1.record(); e1.synchronize()
        ms=cp.cuda.get_elapsed_time(e0,e1); norm=float(cp.linalg.norm(psi)); del psi; cp.get_default_memory_pool().free_all_blocks()
        gb=len(prog)*(1<<n)*16*2/1e9
        extra=("  vs v20 x%.1f"%(V20[n]/ms) if n in V20 else "")+("  vs Qiskit x%.0f"%(QK[n]/ms) if n in QK else "")
        P("n=%-2d (%3d portes -> %3d ops, %6.2f Go etat) : %10.1f ms  norme=%.12f  ~%.0f Go/s VRAM%s"%(n,len(ops),len(prog),need/2**30,ms,norm,gb/(ms/1e3),extra))
    P("="*96)
    f=os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_gpu_v21_results.txt"); open(f,"w").write("\n".join(L)+"\n"); print("JOURNAL="+f)
if __name__=="__main__": main()
