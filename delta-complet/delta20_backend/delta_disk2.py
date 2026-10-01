import os,time,shutil,numpy as np
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):return int(l.split()[1])*1024
def drop(p):
    fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);os.close(fd)
AV=avail();FREE=shutil.disk_usage(".").free;CH=8*2**20
S=int(os.environ.get("D2_MB","0"))*2**20 or int(min(1.3*AV,0.35*FREE,8*2**30))//(CH*8)*(CH*8)
print("=== DELTA DISK v2 : fichier > RAM libre, calcul direct en float32 ===")
print("MEM_LIBRE=%d MB FICHIER_FLOAT64=%d MB (x%.2f la RAM libre) FICHIER_FP16=%d MB"%(AV>>20,S>>20,S/AV,(S//4)>>20),flush=True)
rng=np.random.default_rng(9);ref=0.0;t0=time.perf_counter()
with open("d2_f64.bin","wb") as f64,open("d2_f16.bin","wb") as f16:
    for _ in range(S//(CH*8)):
        x=rng.standard_normal(CH);ref+=float(np.dot(x,x));f64.write(x.tobytes());f16.write(x.astype(np.float16).tobytes())
print("ECRITURE %.0f s"%(time.perf_counter()-t0),flush=True)
drop("d2_f64.bin");drop("d2_f16.bin")
t0=time.perf_counter();s64=0.0;b=np.empty(CH)
with open("d2_f64.bin","rb",buffering=0) as f:
    while (k:=f.readinto(memoryview(b).cast("B"))):c=b[:k//8];s64+=float(np.dot(c,c))
ta=time.perf_counter()-t0
t0=time.perf_counter();s16=0.0;b=np.empty(CH,np.float16)
with open("d2_f16.bin","rb",buffering=0) as f:
    while (k:=f.readinto(memoryview(b).cast("B"))):c=b[:k//2].astype(np.float32);s16+=float(np.dot(c,c))
tb=time.perf_counter()-t0;e=abs(s16-ref)/ref
os.remove("d2_f64.bin");os.remove("d2_f16.bin")
print("AVANT float64 brut   : %.1f s (%.3f GB/s)"%(ta,S/ta/1e9))
print("APRES fp16 -> float32: %.1f s (%.3f GB/s eq. float64) GAIN=x%.2f ERREUR=%.1e"%(tb,S/tb/1e9,ta/tb,e))
print("DISK2_VALIDATION=%s"%("OK" if e<1e-3 else "FAIL"))
