import os
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np,time,hashlib,ctypes,multiprocessing as mp,subprocess
H=os.path.dirname(os.path.abspath(__file__));LIB=os.path.join(H,"libdelta_accel.so")
VEC=4096;IT=2000;B=1.000001;C=0.0000001
def build():
    if not os.path.exists(LIB):subprocess.run(["gcc","-O3","-march=native","-fno-fast-math","-shared","-fPIC",os.path.join(H,"delta_accel.c"),"-o",LIB],check=True)
def detect():
    try:
        import torch
        if torch.cuda.is_available():return "cuda:"+torch.cuda.get_device_name(0)
    except Exception:pass
    return "avx512" if "avx512f" in open("/proc/cpuinfo").read() else "scalaire"
def _vec(seed):return np.random.default_rng(int(seed)).random(VEC,dtype=np.float32)+0.5
def _worker(wid,inq,outq,backend):
    try:os.sched_setaffinity(0,{wid%os.cpu_count()})
    except Exception:pass
    L=ctypes.CDLL(LIB);fn=L.fma_avx512 if backend!="scalaire" else L.fma_scalar
    fn.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float]
    cache={};calc=0
    while True:
        job=inq.get()
        if job is None:break
        out=[]
        for pos,key,seed in job:
            r=cache.get(key)
            if r is None:
                d=_vec(seed);fn(d.ctypes.data,VEC,IT,B,C);r=hashlib.sha256(d.tobytes()).hexdigest()[:16];cache[key]=r;calc+=1
            out.append((pos,r))
        outq.put((wid,out,calc))
class DeltaMTAccel:
    """DELTA multi-threads + accelerateur : index unifie par cle, 1 worker par thread, noyau natif detecte."""
    def __init__(s,n=None,backend=None):
        build();s.backend=backend or detect();kb="avx512" if s.backend.startswith("cuda") else s.backend
        s.n=n or os.cpu_count();s.outq=mp.Queue();s.inq=[mp.Queue() for _ in range(s.n)]
        s.P=[mp.Process(target=_worker,args=(i,s.inq[i],s.outq,kb),daemon=True) for i in range(s.n)];[p.start() for p in s.P]
    def compute_batch(s,reqs):
        parts=[[] for _ in range(s.n)]
        for pos,(key,seed) in enumerate(reqs):parts[int(key[:8],16)%s.n].append((pos,key,seed))
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
    keys={s:hashlib.sha256(_vec(s).tobytes()).hexdigest() for s in range(uniq)};reqs=[(keys[int(s)],int(s)) for s in ids]
    fl=uniq*VEC*IT*2;print("=== DELTA MT + ACCELERATEUR | detecte : %s | %d threads ==="%(detect(),os.cpu_count()));sigs=set();base=None
    for be,n in (("scalaire",1),("scalaire",os.cpu_count()),("avx512",1),("avx512",os.cpu_count())):
        D=DeltaMTAccel(n,be);t=time.perf_counter();res,calc=D.compute_batch(reqs);w=time.perf_counter()-t;D.close()
        sig=hashlib.sha256("".join(res).encode()).hexdigest()[:12];sigs.add(sig);base=base or w
        print("%-9s workers %2d | %8.1f ms | %6.1f GF reels | %d calculs | gain x%6.1f | sig %s"%(be,n,w*1e3,fl/w/1e9,calc,base/w,sig))
    print("DELTA_MT_ACCEL_VALIDATION=%s"%("OK (resultats identiques sur tous les moteurs)" if len(sigs)==1 else "FAIL "+str(sigs)))
