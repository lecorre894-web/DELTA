import os,sys,time,shutil,threading,queue,numpy as np
def meminfo(k):
    for l in open("/proc/meminfo"):
        if l.startswith(k):return int(l.split()[1])*1024
AV=int(os.environ.get("MEM_MB","0"))*2**20 or meminfo("MemAvailable");FREE_DISK=shutil.disk_usage(".").free
FILE=os.environ.get("MEM_FILE","delta_mem_stream.bin");CH=64*2**20
SIZE=int(os.environ.get("FILE_MB","0"))*2**20 or int(min(1.25*AV,0.5*FREE_DISK,6*2**30))//CH*CH
print("=== DELTA MEMORY BENCH (streaming, asynchrone, types) ===")
print("MEM_AVAILABLE=%d MB DISK_FREE=%d MB FILE=%d MB (%.2fx la RAM libre) CHUNK=%d MB"%(AV>>20,FREE_DISK>>20,SIZE>>20,SIZE/AV,CH>>20))
t0=time.perf_counter();buf=np.arange(CH//8,dtype=np.float64)
with open(FILE,"wb",buffering=0) as f:
    for i in range(SIZE//CH):buf+=1.0;f.write(memoryview(buf).cast("B"))
    os.fsync(f.fileno())
w=time.perf_counter()-t0;print("E1 WRITE  %d MB en %.1f s = %.2f GB/s"%(SIZE>>20,w,SIZE/w/1e9),flush=True)
def stream(fn):
    b=np.empty(CH//8);s=0.0;t=time.perf_counter()
    with open(FILE,"rb",buffering=0) as f:
        while f.readinto(memoryview(b).cast("B")):s+=fn(b)
    return time.perf_counter()-t,s
r,ref=stream(lambda b:float(b.sum()))
print("E1 READ   fichier (> RAM libre si ratio>1) %.2f GB/s"%(SIZE/r/1e9))
n=int(min(0.25*AV,512*2**20))//8;a=np.ones(n);a.sum();t=time.perf_counter();[a.sum() for _ in range(5)];ram=5*n*8/(time.perf_counter()-t)
print("E1 RAM    tableau %d MB en RAM %.2f GB/s -> la RAM est x%.1f plus rapide que le flux fichier"%(n*8>>20,ram/1e9,ram/(SIZE/r)),flush=True);del a
K=int(os.environ.get("COMPUTE_PASSES","0"))
def comp(b,k):
    x=b.copy()
    for _ in range(k):np.multiply(x,1.0000001,out=x);np.add(x,1e-9,out=x)
    return float(x.sum())
if not K:
    b=np.ones(CH//8);t=time.perf_counter();comp(b,4);tc=(time.perf_counter()-t)/4;K=max(1,int((r/(SIZE//CH))/tc))
print("E2 COMPUTE_PASSES=%d (calcul ~= lecture, cas ou l'asynchrone peut le plus)"%K,flush=True)
ts,ss=stream(lambda b:comp(b,K))
def asyn():
    q=queue.Queue(2);pool=[np.empty(CH//8) for _ in range(4)];s=0.0
    def rd():
        i=0
        with open(FILE,"rb",buffering=0) as f:
            while True:
                b=pool[i%4];k=f.readinto(memoryview(b).cast("B"))
                if not k:q.put(None);return
                q.put(b);i+=1
    th=threading.Thread(target=rd);t=time.perf_counter();th.start()
    while (b:=q.get()) is not None:s+=comp(b,K)
    th.join();return time.perf_counter()-t,s
ta,sa=asyn()
best=ts/max(r,ts-r)
print("E2 SYNC   %.2f s | ASYNC double tampon %.2f s | GAIN x%.2f (max theorique x%.2f) | CHECKSUM %s"%(ts,ta,ts/ta,best,"MATCH" if abs(ss-sa)<=1e-9*abs(ss) else "DIFF"),flush=True)
os.remove(FILE)
N=int(min(0.08*AV,256*2**20))//8;rng=np.random.default_rng(7);x=rng.standard_normal(N);y=rng.standard_normal(N);ref_dot=float(np.dot(x,y))
print("E3 TYPES  N=%d elements, oracle = float64"%N)
def q8(v):sc=127/4.0;return np.clip(np.rint(v*sc),-127,127).astype(np.int8),sc
for name in ("float64","float32","float16","int8"):
    if name=="int8":xs,sc=q8(x);ys,_=q8(y);back=lambda v:v.astype(np.float64)/sc
    else:xs=x.astype(name);ys=y.astype(name);back=lambda v:v.astype(np.float64)
    dst=np.empty_like(xs);np.copyto(dst,xs);t=time.perf_counter()
    for _ in range(5):np.copyto(dst,xs)
    tt=(time.perf_counter()-t)/5
    rms=float(np.sqrt(np.mean((back(xs)-x)**2)));dot=float(np.dot(back(xs),back(ys)))
    print("E3 %-7s OCTETS/ELEM=%d MEM=%5d MB DEBIT=%7.1f M elem/s ERREUR_RMS=%.1e ERREUR_DOT_REL=%.1e"%(name,xs.itemsize,xs.nbytes>>20,N/tt/1e6,rms,abs(dot-ref_dot)/abs(ref_dot)),flush=True)
print("MEMORY_VALIDATION=%s"%("OK" if abs(ss-sa)<=1e-9*abs(ss) else "FAIL"))
