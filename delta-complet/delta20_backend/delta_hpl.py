import time,os,sys,platform,numpy as np
TOP1_PF=2198.40;TOP500_PF=2.66;CM5_1993_GF=59.7
def cpu():
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):return l.split(":",1)[1].strip()
    except Exception:pass
    return platform.processor()
def hpl(n,rng):
    A=rng.random((n,n))-0.5;b=rng.random(n)-0.5
    t0=time.perf_counter();x=np.linalg.solve(A,b);t=time.perf_counter()-t0
    r=np.linalg.norm(A@x-b,np.inf)/(np.finfo(float).eps*(np.linalg.norm(A,np.inf)*np.linalg.norm(x,np.inf)+np.linalg.norm(b,np.inf))*n)
    return t,(2/3*n**3+2*n**2)/t/1e9,r
print("=== DELTA STRESS HPL (methode LINPACK du TOP500, FP64) ===")
print("CPU=%s LOGICAL_CPUS=%d NUMPY=%s"%(cpu(),os.cpu_count(),np.__version__))
rng=np.random.default_rng(42);best=0;ok=True;Q={"HPL":0.0,"DGEMM":0.0,"JOBS":0.0};T0=time.perf_counter()
def avail_mb():
    try:
        import psutil;return psutil.virtual_memory().available/2**20
    except ImportError:pass
    if os.environ.get("HPL_MEM_MB"):return int(os.environ["HPL_MEM_MB"])
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):return int(l.split()[1])//1024
    return 2048
AV=avail_mb();BUDGET=0.35*AV*2**20;NMAX=int((BUDGET/(8*3))**0.5)//500*500
NS=[int(x) for x in os.environ["HPL_NS"].split(",")] if os.environ.get("HPL_NS") else [n for n in (1000,2000,4000,6000,8000,12000,16000) if n<=NMAX] or [max(500,NMAX)]
print("MEM_AVAILABLE=%d MB BUDGET_35%%=%d MB N_MAX_SUR=%d (DELTA sans memoire dediee : bench adaptatif)"%(AV,BUDGET/2**20,NMAX))
for n in NS:
    ts=[];gs=[]
    for _ in range(3 if n<=4000 else 1):
        t,g,r=hpl(n,rng);ts.append(t);gs.append(g);ok&=r<16;Q["HPL"]+=2/3*n**3+2*n**2
    g=max(gs);best=max(best,g)
    print("HPL N=%5d MEM=%7.0f MB WALL=%8.3f s GFLOPS=%7.2f RESIDU=%.4f %s"%(n,8*n*n/2**20,min(ts),g,r,"PASSED" if r<16 else "FAILED"),flush=True)
n=min(4096,max(512,NMAX));A=rng.random((n,n));B=rng.random((n,n));A@B;t0=time.perf_counter();A@B;t=time.perf_counter()-t0;Q["DGEMM"]+=2*2*n**3
print("DGEMM N=%d WALL=%.3f s GFLOPS=%.2f (pic pratique matmul)"%(n,t,2*n**3/t/1e9))
k=int(min(100000,max(1000,BUDGET/(8*32*32*3))));S=rng.random((k,32,32))+32*np.eye(32);v=rng.random((k,32));np.linalg.solve(S[:100],v[:100,:,None]);t0=time.perf_counter();X=np.linalg.solve(S,v[:,:,None]);t=time.perf_counter()-t0;Q["JOBS"]+=(k+100)*(2/3*32**3+2*32**2)
print("PETITS_JOBS %d systemes 32x32 WALL=%.3f s US_PAR_JOB=%.2f JOBS_PAR_S=%.0f"%(k,t,t/k*1e6,k/t))
print("--- CLASSEMENT (Rmax = meilleur HPL mesure) ---")
print("RMAX_DELTA=%.2f GFLOPS"%best)
print("TOP500 #500 juin 2026 = %.2f PFLOPS -> facteur x%.0f, il faudrait ~%.0f machines identiques a celle-ci SANS perte reseau [PROJECTION]"%(TOP500_PF,TOP500_PF*1e6/best,TOP500_PF*1e6/best))
print("TOP500 #1 juin 2026 = %.0f PFLOPS -> facteur x%.2e [PROJECTION]"%(TOP1_PF,TOP1_PF*1e6/best))
f=2/3*NS[-1]**3
print("DELAI THEORIQUE du #1 pour notre plus gros HPL (N=%d) = %.3f us a Rmax plein [PROJECTION, petit probleme = machine sous-utilisee]"%(NS[-1],f/(TOP1_PF*1e15)*1e6))
print("HISTOIRE: #1 mondial juin 1993 (CM-5) = %.1f GFLOPS -> DELTA %s"%(CM5_1993_GF,"l'aurait battu" if best>CM5_1993_GF else "x%.2f de ce niveau"%(best/CM5_1993_GF)))
tot=sum(Q.values());W=time.perf_counter()-T0
print("--- QUANTITE DE CALCUL REELLEMENT EXECUTEE ---")
for kk,vv in Q.items():print("QTE_%s=%.3e operations (%.2f GFLOP)"%(kk,vv,vv/1e9))
print("QTE_TOTALE=%.3e operations flottantes FP64 en %.1f s de run = %.1f milliards d'operations, resultats HPL verifies (residu<16)"%(tot,W,tot/1e9))
print("EQUIVALENT_HUMAIN=%.0f ans a 1 operation/seconde"%(tot/31557600))
print("HPL_VALIDATION=%s"%("OK" if ok else "FAIL"))
