import os
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np,time,hashlib,multiprocessing as mp
N=128
def _mat(seed):return np.random.default_rng(int(seed)).random((N,N),dtype=np.float32)
def _mm(A,T=32):
    C=np.empty((N,N),np.float32)
    for i in range(0,N,T):
        for j in range(0,N,T):C[i:i+T,j:j+T]=A[i:i+T]@A[:,j:j+T]
    return C
def _worker(wid,inq,outq):
    try:os.sched_setaffinity(0,{wid%os.cpu_count()})
    except Exception:pass
    cache={};calc=0
    while True:
        job=inq.get()
        if job is None:break
        out=[]
        for pos,key,seed in job:
            r=cache.get(key)
            if r is None:r=hashlib.sha256(_mm(_mat(seed)).tobytes()).hexdigest()[:16];cache[key]=r;calc+=1
            out.append((pos,r))
        outq.put((wid,out,calc))
class DeltaMT:
    """DELTA multi-threads : 1 partition de cles par worker, index unifie, resultat ordonne."""
    def __init__(s,n=None):
        s.n=n or os.cpu_count();s.outq=mp.Queue();s.inq=[mp.Queue() for _ in range(s.n)]
        s.P=[mp.Process(target=_worker,args=(i,s.inq[i],s.outq),daemon=True) for i in range(s.n)];[p.start() for p in s.P]
    def route(s,key):return int(key[:8],16)%s.n
    def compute_batch(s,reqs):
        parts=[[] for _ in range(s.n)]
        for pos,(key,seed) in enumerate(reqs):parts[s.route(key)].append((pos,key,seed))
        for i in range(s.n):s.inq[i].put(parts[i])
        res=[None]*len(reqs);calc=0
        for _ in range(s.n):
            wid,out,c=s.outq.get();calc+=c
            for pos,r in out:res[pos]=r
        return res,calc
    def close(s):
        for q in s.inq:q.put(None)
        [p.join() for p in s.P]
if __name__=="__main__":
    M=3000;rng=np.random.default_rng(1);uniq=300;ids=rng.integers(0,uniq,M);ids[:uniq]=np.arange(uniq);rng.shuffle(ids)
    keys={s:hashlib.sha256(_mat(s).tobytes()).hexdigest() for s in range(uniq)}
    reqs=[(keys[int(s)],int(s)) for s in ids]
    print("=== DELTA MULTI-THREADS : partition par cle, %d threads visibles ==="%os.cpu_count());base=None;sigs=set()
    for n in sorted({1,2,os.cpu_count()}):
        D=DeltaMT(n);t=time.perf_counter();res,calc=D.compute_batch(reqs);w=time.perf_counter()-t;D.close()
        sig=hashlib.sha256("".join(res).encode()).hexdigest()[:12];sigs.add(sig);base=base or w
        print("workers %2d | %6.1f ms | %4d calculs reels | gain x%.2f | sig %s"%(n,w*1e3,calc,base/w,sig))
    print("DELTA_MT_VALIDATION=%s"%("OK (resultats identiques quel que soit le nombre de workers)" if len(sigs)==1 else "FAIL"))
