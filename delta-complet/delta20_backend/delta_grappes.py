import os
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
import sys,time,multiprocessing as mp,numpy as np,scipy.linalg as sl
N=int(os.environ.get("GR_N","2000"));REP=int(os.environ.get("GR_REP","3"))
def node(args):
    gid,start=args
    while time.time()<start:time.sleep(0.0005)
    rng=np.random.default_rng(gid);A=rng.random((N,N))-0.5;b=rng.random(N)-0.5;fl=0.0;t0=time.perf_counter();okn=True
    for _ in range(REP):
        x=sl.lu_solve(sl.lu_factor(A,check_finite=False),b,check_finite=False)
        r=np.linalg.norm(A@x-b,np.inf)/(np.finfo(float).eps*(np.linalg.norm(A,np.inf)*np.linalg.norm(x,np.inf)+np.linalg.norm(b,np.inf))*N);okn&=r<16;fl+=2/3*N**3+2*N**2
    return fl,time.perf_counter()-t0,okn
if __name__=="__main__":
    cpu=os.cpu_count();XM=int(os.environ.get("GR_MAX",str(2*cpu)));rows=[];ok=True;base=None
    print("=== DELTA GRAPPES 1..%d : noeuds = processus independants, 1 thread BLAS chacun (style cluster TOP500) ==="%XM)
    print("CPU_LOGIQUES=%d N=%d par noeud, %d resolutions HPL chacun, residu verifie"%(cpu,N,REP))
    for X in range(1,XM+1):
        with mp.Pool(X) as p:
            st=time.time()+0.3;t0=time.perf_counter();res=p.map(node,[(g,st) for g in range(X)]);wall=time.perf_counter()-t0-0.3
        fl=sum(r[0] for r in res);ok&=all(r[2] for r in res);g=fl/max(r[1] for r in res)/1e9
        if base is None:base=g
        eff=g/(base*X)*100;rows.append((X,g))
        print("GRAPPES=%2d GFLOPS_AGREGES=%7.2f PAR_NOEUD=%6.2f EFFICACITE=%5.1f%% %s"%(X,g,g/X,eff,"(surcharge : plus de noeuds que de CPU)" if X>cpu else ""),flush=True)
    bx,bg=max(rows,key=lambda r:r[1])
    print("--- CLASSEMENT STYLE TOP500 (non homologue) ---")
    print("MEILLEURE_CONFIG=%d grappes RMAX_AGREGE=%.2f GFLOPS | entree TOP500 2026 = 2.66 PFLOPS -> x%.0f"%(bx,bg,2.66e6/bg))
    print("NOTE: agregat de HPL independants, pas un HPL distribue unique (le TOP500 exige un seul run sur tout le systeme)")
    print("GRAPPES_VALIDATION=%s"%("OK" if ok else "FAIL"))
