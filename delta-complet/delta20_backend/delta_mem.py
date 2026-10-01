import os,sys,json,time,threading,queue,numpy as np
def _avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):return int(l.split()[1])*1024
class DeltaMem:
    def __init__(s,budget_mb=None,tol=1e-3,cold_dir="delta_mem_cold"):
        s.budget=int(budget_mb*2**20) if budget_mb else int(0.25*_avail());s.tol=tol;s.dir=cold_dir;os.makedirs(cold_dir,exist_ok=True);s.hot={};s.meta={};s.used=0
    @staticmethod
    def _encode(x,kind):
        if kind=="int8i":return x.astype(np.int8),1.0,0.0
        if kind=="uint8i":return x.astype(np.uint8),1.0,0.0
        if kind=="int8q":
            mn,mx=float(x.min()),float(x.max());sc=(mx-mn)/254 or 1.0
            return (np.rint((x-mn)/sc)-127).astype(np.int8),sc,mn+127*sc
        return x.astype(kind),1.0,0.0
    @staticmethod
    def _decode(q,sc,off):return q.astype(np.float64)*sc+off
    def put(s,name,x,tol=None):
        x=np.asarray(x,dtype=np.float64);tol=s.tol if tol is None else tol;rx=float(np.sqrt(np.mean(x*x))) or 1.0
        isint=bool(np.all(x==np.rint(x)));cands=[]
        if isint and x.min()>=-128 and x.max()<=127:cands.append("int8i")
        if isint and x.min()>=0 and x.max()<=255:cands.append("uint8i")
        cands+=["int8q","float16","float32","float64"]
        for k in cands:
            q,sc,off=s._encode(x,k);err=float(np.sqrt(np.mean((s._decode(q,sc,off)-x)**2)))/rx
            if err<=tol:break
        m={"kind":k,"dtype":str(q.dtype),"shape":list(x.shape),"scale":sc,"offset":off,"err":err,"bytes":q.nbytes,"raw_bytes":x.nbytes}
        if s.used+q.nbytes<=s.budget:s.hot[name]=q;s.used+=q.nbytes;m["tier"]="HOT_RAM"
        else:
            p=os.path.join(s.dir,name+".bin");q.tofile(p);m["tier"]="COLD_DISK";m["path"]=p
        s.meta[name]=m;json.dump(s.meta,open(os.path.join(s.dir,"index.json"),"w"),indent=1);return m
    def get(s,name):
        m=s.meta[name]
        q=s.hot[name] if m["tier"]=="HOT_RAM" else np.fromfile(m["path"],dtype=m["dtype"]).reshape(m["shape"])
        return s._decode(q,m["scale"],m["offset"])
    def stream(s,name,chunk_elems=8*2**20,asyn=True):
        m=s.meta[name];dt=np.dtype(m["dtype"])
        if m["tier"]=="HOT_RAM":
            q=s.hot[name].reshape(-1)
            for i in range(0,q.size,chunk_elems):yield s._decode(q[i:i+chunk_elems],m["scale"],m["offset"])
            return
        if not asyn:
            with open(m["path"],"rb",buffering=0) as f:
                b=np.empty(chunk_elems,dt)
                while (k:=f.readinto(memoryview(b).cast("B"))):yield s._decode(b[:k//dt.itemsize],m["scale"],m["offset"])
            return
        Q=queue.Queue(2);pool=[np.empty(chunk_elems,dt) for _ in range(4)]
        def rd():
            i=0
            with open(m["path"],"rb",buffering=0) as f:
                while True:
                    b=pool[i%4];k=f.readinto(memoryview(b).cast("B"))
                    if not k:Q.put(None);return
                    Q.put((b,k//dt.itemsize));i+=1
        th=threading.Thread(target=rd,daemon=True);th.start()
        while (it:=Q.get()) is not None:yield s._decode(it[0][:it[1]],m["scale"],m["offset"])
        th.join()
    def report(s):
        raw=sum(m["raw_bytes"] for m in s.meta.values());st=sum(m["bytes"] for m in s.meta.values())
        return {"objects":len(s.meta),"raw_float64_MB":raw/2**20,"stored_MB":st/2**20,"gain":raw/max(st,1),"hot_MB":s.used/2**20,"budget_MB":s.budget/2**20}
if __name__=="__main__":
    tol=float(os.environ.get("MEM_TOL","1e-3"));N=int(os.environ.get("MEM_N",str(8*2**20)))
    D=DeltaMem(budget_mb=float(os.environ.get("MEM_BUDGET_MB","80")),tol=tol);rng=np.random.default_rng(3)
    print("=== DELTA_MEM : stockage typé par oracle, RAM chaude / disque froid, flux asynchrone ===")
    print("TOLERANCE_RMS_REL=%.0e BUDGET_RAM=%.0f MB N=%d par tableau"%(tol,D.budget/2**20,N))
    data={"pixels_0_255":rng.integers(0,256,N).astype(np.float64),"compteurs_signes":rng.integers(-100,100,N).astype(np.float64),
          "signal_lisse":np.sin(np.linspace(0,60,N))+0.001*rng.standard_normal(N),"gaussien":rng.standard_normal(N),
          "horner_exact":np.cumsum(rng.standard_normal(N))*1e-3}
    ok=True
    for name,x in data.items():
        t=tol if name!="horner_exact" else 1e-12
        m=D.put(name,x,tol=t);back=D.get(name);e=float(np.sqrt(np.mean((back-x)**2)))/(float(np.sqrt(np.mean(x*x))) or 1);ok&=e<=t
        print("PUT %-16s TYPE=%-7s %4d->%4d MB x%.1f ERR=%.1e<=%.0e %s %s"%(name,m["kind"],m["raw_bytes"]>>20,m["bytes"]>>20,m["raw_bytes"]/m["bytes"],e,t,"OK" if e<=t else "FAIL",m["tier"]),flush=True)
    cold=[n for n,m in D.meta.items() if m["tier"]=="COLD_DISK"]
    if cold:
        n=cold[0];D.get(n)
        for mode in (False,True):
            t0=time.perf_counter();s=0.0
            for c in D.stream(n,asyn=mode):s+=float(np.sum(np.tanh(c)**2))
            w=time.perf_counter()-t0;print("STREAM %-16s %-5s %.3f s SUM=%.10e"%(n,"ASYNC" if mode else "SYNC",w,s),flush=True)
            if mode:ok&=abs(s-s0)<=1e-9*abs(s0)
            s0=s
    r=D.report();print("REPORT objets=%d float64_brut=%.0f MB stocke=%.0f MB GAIN_MEMOIRE=x%.2f RAM_chaude=%.0f/%.0f MB"%(r["objects"],r["raw_float64_MB"],r["stored_MB"],r["gain"],r["hot_MB"],r["budget_MB"]))
    print("DELTA_MEM_VALIDATION=%s"%("OK" if ok else "FAIL"))
