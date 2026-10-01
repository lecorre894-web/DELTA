import os,sys,time,json,platform,numpy as np,scipy.linalg as sl
TOP500_PF=2.66;CM5_GF=59.7
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):return int(l.split()[1])*1024
def cpu():
    for l in open("/proc/cpuinfo"):
        if l.startswith("model name"):return l.split(":",1)[1].strip()
def resid(A,x,b,n):return np.linalg.norm(A@x-b,np.inf)/(np.finfo(float).eps*(np.linalg.norm(A,np.inf)*np.linalg.norm(x,np.inf)+np.linalg.norm(b,np.inf))*n)
def hpl(accel):
    AV=int(os.environ.get("MEM_MB","0"))*2**20 or avail();NMAX=int((0.35*AV/(8*4))**0.5)//500*500
    NS=[n for n in (2000,4000,6000,8000,10000) if n<=NMAX] or [1000];rng=np.random.default_rng(5);R={"fp64":0,"mxp":0};ok=True;Q=0.0
    print("MEM_AVAILABLE=%d MB N_MAX=%d"%(AV>>20,NMAX))
    for n in NS:
        A=rng.random((n,n))-0.5;b=rng.random(n)-0.5;fl=2/3*n**3+2*n**2
        t0=time.perf_counter();x=sl.lu_solve(sl.lu_factor(A,check_finite=False),b,check_finite=False);t=time.perf_counter()-t0;r=resid(A,x,b,n);ok&=r<16;Q+=fl
        R["fp64"]=max(R["fp64"],fl/t/1e9);line="HPL N=%5d FP64 %.3f s %6.1f GFLOPS RES=%.3f"%(n,t,fl/t/1e9,r)
        if accel in("mxp","all"):
            t0=time.perf_counter();lu=sl.lu_factor(A.astype(np.float32),check_finite=False);x=sl.lu_solve(lu,b.astype(np.float32),check_finite=False).astype(np.float64);it=0
            while resid(A,x,b,n)>=1 and it<20:x+=sl.lu_solve(lu,(b-A@x).astype(np.float32),check_finite=False).astype(np.float64);it+=1
            tm=time.perf_counter()-t0;rm=resid(A,x,b,n);ok&=rm<16;Q+=fl;R["mxp"]=max(R["mxp"],fl/tm/1e9)
            line+=" | MxP %.3f s %6.1f GFLOPS-eq RES=%.3f ITER=%d x%.2f"%(tm,fl/tm/1e9,rm,it,t/tm)
        print(line,flush=True)
    return R,ok,Q
def qpu_accel(physical=True):
    from delta_qpu_cache import DeltaQPUCache,ghz
    D=DeltaQPUCache();qs=[ghz(n) for n in (4,5,6,8)];o,st=D.run_batch(qs,1000,physical=physical)
    for q,x in zip(qs,o):
        c=x["counts"];n=q.num_qubits;pop=(c.get("0"*n,0)+c.get("1"*n,0))/sum(c.values())
        print("QPU GHZ%d NIVEAU=%-13s POP=%.3f QPU=%s JOB=%s"%(n,x["level"],pop,x["origin"].get("backend","-"),x["origin"].get("job_id","-")))
    st["shots_total"]=1000*len(qs);return st
if __name__=="__main__":
    acc=os.environ.get("ACCEL",sys.argv[1] if len(sys.argv)>1 else "none")
    print("=== DELTA TOP500-STYLE (non homologue) ACCELERATEUR=%s ==="%acc)
    R,ok,Q=hpl(acc);Q_s=None
    if acc in("qpu","all"):
        st=qpu_accel(physical=os.environ.get("QPU_SIM","0")!="1")
        print("QPU_BATCH circuits=%d depuis_cache=%d calcules=%d JOBS=%d (au lieu de %d) WALL=%.1f s BACKEND=%s JOB=%s"%(st["circuits"],st["hits"],st["computed"],st["jobs"],st["computed"],st["wall"],st["backend"],st["job_id"]))
        Q_s=st
    print("--- FICHE STYLE TOP500 (non homologuee) ---")
    print("SYSTEME=DELTA / GitHub Codespaces | CPU=%s | COEURS_LOGIQUES=%d"%(cpu(),os.cpu_count()))
    print("ACCELERATEUR=%s"%{"none":"aucun","mxp":"precision mixte FP32 (logiciel)","qpu":"QPU IBM 156 qubits (lots + cache)","all":"precision mixte FP32 + QPU IBM 156 qubits"}.get(acc,acc))
    print("RMAX_HPL_FP64=%.2f GFLOPS  (seul chiffre comparable au TOP500 ; entree 2026 = %.2f PFLOPS, facteur x%.0f)"%(R["fp64"],TOP500_PF,TOP500_PF*1e6/R["fp64"]))
    if R["mxp"]:print("RMAX_HPL_MxP=%.2f GFLOPS-eq  (style liste HPL-MxP, gain x%.2f)"%(R["mxp"],R["mxp"]/R["fp64"]))
    if Q_s:print("QPU_DEBIT=%d shots en %d job(s), %d circuit(s) servis du cache  (non compte en FLOPS : un QPU ne fait pas de HPL)"%(Q_s["shots_total"],Q_s["jobs"],Q_s["hits"]))
    print("QUANTITE_CALCUL_CLASSIQUE=%.3e operations FP64 verifiees | 1993 #1 CM-5 %s"%(Q,"battu" if R["fp64"]>CM5_GF else "non battu"))
    print("TOP500_VALIDATION=%s"%("OK" if ok else "FAIL"))
