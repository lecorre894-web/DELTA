import numpy as np,time,json,os,sqlite3,hashlib,sys
def mem_guard(min_mb=1500):
    av=int([l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0])//1024
    print("MEM_GUARD disponible %d Mo (seuil %d) -> %s"%(av,min_mb,"OK" if av>=min_mb else "REFUS: redemarrer le Codespace"));return av>=min_mb
def rtx_mm(A,B,mode):
    N=A.shape[0];C=np.zeros((N,N),A.dtype);t=time.perf_counter()
    if mode=="lots":
        g=np.arange(N*N);r,c=g//N,g%N;C[r,c]=np.einsum("ij,ji->i",A[r],B[:,c])
    else:
        T=32
        for i in range(0,N,T):
            for j in range(0,N,T):C[i:i+T,j:j+T]=A[i:i+T]@B[:,j:j+T]
    return time.perf_counter()-t,C
def ryzen_mm(A,B):
    N=A.shape[0];C=np.empty((N,N),A.dtype);T=max(1,N//32);t=time.perf_counter()
    for lo in range(0,N,T):C[lo:lo+T]=A[lo:lo+T]@B
    return time.perf_counter()-t,C
class DiskCache:
    def __init__(s,p="delta_cloud_cache.db"):
        s.c=sqlite3.connect(p);s.c.execute("create table if not exists k(key text primary key,val text,t real)")
    def get(s,k):r=s.c.execute("select val from k where key=?",(k,)).fetchone();return r and r[0]
    def put(s,k,v):s.c.execute("insert or replace into k values(?,?,?)",(k,v,time.time()));s.c.commit()
def compute(cache,prec,n,seed):
    key=hashlib.sha256(("%s|%d|%d"%(prec,n,seed)).encode()).hexdigest();v=cache.get(key)
    if v:return v,True
    r=np.random.default_rng(seed);a=r.random((n,n)).astype(np.float64 if prec=="FP64" else np.float32);v=hashlib.sha256((a@a).tobytes()).hexdigest()[:16];cache.put(key,v);return v,False
if __name__=="__main__":
    ok=True;print("=== DELTA UPGRADES ===");mem_guard()
    rng=np.random.default_rng(7);N=256;A=rng.random((N,N),dtype=np.float32);B=rng.random((N,N),dtype=np.float32);ref=A.astype(np.float64)@B
    tl,Cl=rtx_mm(A,B,"lots");tt,Ct=rtx_mm(A,B,"tuiles");tr,Cr=ryzen_mm(A,B)
    for nom,t,C in (("RTX lots (avant)",tl,Cl),("RTX tuiles 32x32",tt,Ct),("Ryzen tranches 32",tr,Cr)):
        e=float(np.abs(C-ref).max()/np.abs(ref).max());ok&=e<1e-5;print("MATMUL %-18s %.4f s err %.0e"%(nom,t,e))
    print("GAIN TUILES vs LOTS = x%.1f"%(tl/tt))
    c=DiskCache();t=time.perf_counter();res=[compute(c,"FP32",256,s) for s in range(20)];w=time.perf_counter()-t
    print("CACHE DISQUE session %s : 20 demandes, %d servies du disque, %.3f s"%(sys.argv[1] if len(sys.argv)>1 else "1",sum(h for _,h in res),w))
    print("UPGRADES_VALIDATION=%s"%("OK" if ok else "FAIL"))
