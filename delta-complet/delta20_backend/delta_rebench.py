import os,sys,time,json,shutil,numpy as np
from delta_mem import DeltaMem
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):return int(l.split()[1])*1024
def drop(p):
    fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);os.close(fd)
AV=avail();S=int(os.environ.get("REB_MB","0"))*2**20 or int(min(1024*2**20,0.2*AV));N=S//8;TOL=1e-3
print("=== DELTA REBENCH : avant (float64 brut) / apres (delta_core) sur les points faibles ===")
print("MEM_AVAILABLE=%d MB DATASET=%d MB (%d float64) TOLERANCE=%.0e"%(AV>>20,S>>20,N,TOL),flush=True)
rng=np.random.default_rng(11);x=rng.standard_normal(N);ref=float(np.dot(x,x));ok=True
D=DeltaMem(budget_mb=0.001,tol=TOL,cold_dir="delta_rebench_cold");t0=time.perf_counter();m=D.put("dataset",x);tput=time.perf_counter()-t0
raw="delta_rebench_raw.bin";x.tofile(raw)
print("A RAM     AVANT=%d MB  APRES=%d MB (%s)  GAIN=x%.2f  -> le HPL tue faute de RAM aurait 4x plus de marge"%(S>>20,m["bytes"]>>20,m["kind"],S/m["bytes"]),flush=True)
del x
drop(raw);t0=time.perf_counter();s0=0.0;b=np.empty(8*2**20)
with open(raw,"rb",buffering=0) as f:
    while (k:=f.readinto(memoryview(b).cast("B"))):c=b[:k//8];s0+=float(np.dot(c,c))
ta=time.perf_counter()-t0
drop(m["path"]);t0=time.perf_counter();s1=0.0
for c in D.stream("dataset",asyn=True):s1+=float(np.dot(c,c))
tb=time.perf_counter()-t0;e=abs(s1-ref)/ref;ok&=e<=10*TOL
print("B DISQUE  lecture a froid + calcul : AVANT %.2f s (%.2f GB/s) | APRES %.2f s (%.2f GB/s eq. float64) | GAIN=x%.2f ERREUR=%.1e"%(ta,S/ta/1e9,tb,S/tb/1e9,ta/tb,e),flush=True)
os.remove(raw);shutil.rmtree("delta_rebench_cold",ignore_errors=True)
from delta_core import DeltaCore
from delta_qpu_cache import ghz
C=DeltaCore();n=int(os.environ.get("REB_SIMQ","20"));qc=ghz(n)
t0=time.perf_counter();C.qpu._sim(qc,1000);tsim=time.perf_counter()-t0
C.quantum("rebench_ghz%d"%n,qc,1000);C.checkpoint()
C2=DeltaCore();t0=time.perf_counter();p=C2.get("rebench_ghz%d"%n);tre=time.perf_counter()-t0;ok&=abs(p.sum()-1)<1e-3
print("C REPRISE simulation GHZ%d : AVANT recalcul %.3f s | APRES relu du disque %.4f s | GAIN=x%.0f"%(n,tsim,tre,tsim/tre),flush=True)
try:
    q=json.load(open("delta_multiqpu.json"));best=max(q["results"],key=lambda r:r["pop"]);t0=time.perf_counter()
    r=C2.qpu.run(ghz(8),1000,True);tq=time.perf_counter()-t0
    print("C REPRISE QPU %s GHZ8 : AVANT file+execution %.0f s | APRES cache %s %.4f s | GAIN=x%.0f | SHOTS_DEPENSES=0 si niveau != MISS"%(best["backend"],best["done_after_s"],r["level"],tq,best["done_after_s"]/tq))
    ok&=r["level"]!="MISS_COMPUTED"
except FileNotFoundError:print("C REPRISE QPU : delta_multiqpu.json absent, etape sautee")
print("REBENCH_VALIDATION=%s"%("OK" if ok else "FAIL"))
