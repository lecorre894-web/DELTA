"""DELTA x X512 : un coeur X512 par worker DELTA, chaque worker epingle sur son thread Xeon (socle propre).
Index par cle partage en partitions (cle -> worker), cache par worker, resultats ordonnes. Programmes X512 traduits en AVX-512 natif."""
import os,sys,time,hashlib,ctypes as ct,subprocess,multiprocessing as mp,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
XDIR=os.environ.get("X512_DIR",os.path.expanduser("~/turnip/avx2048/x512"))
LIB=os.path.join(HERE,"libdelta_x512.so")
CSRC=r'''
#include <stdlib.h>
#include "p_direct.h"
#include "p_micro.h"
typedef struct{__m512i X[32],P[16];uint64_t R[16];Micro M[32];}Ctx;
void* x512_new(void){Ctx*c;if(posix_memalign((void**)&c,64,sizeof(Ctx)))return 0;return c;}
void x512_free(void*c){free(c);}
uint64_t x512_run(void*vc,int prog,const uint64_t*A,const uint64_t*B,const uint64_t*C,long nv,long*ni){Ctx*c=vc;
 for(int i=0;i<16;i++){c->R[i]=0;c->P[i]=_mm512_setzero_si512();}
 for(int i=0;i<32;i++){c->M[i].m1=c->M[i].m2=c->M[i].m4=c->M[i].m8=_mm512_setzero_si512();c->M[i].S=0;c->M[i].n=0;}
 *ni=0;if(prog==0){p_direct(c->X,c->R,c->P,c->M,A,B,C,ni,nv);return c->R[0];}p_micro(c->X,c->R,c->P,c->M,A,B,C,ni,nv);return c->R[1];}
'''
def build():
    xc=os.path.join(XDIR,"x512c.py")
    if not os.path.exists(xc):raise SystemExit("x512c.py introuvable dans %s (X512_DIR)"%XDIR)
    if os.path.exists(LIB) and os.path.getmtime(LIB)>os.path.getmtime(xc):return
    d=os.path.join(HERE,"_x512_build");os.makedirs(d,exist_ok=True)
    h=subprocess.check_output([sys.executable,xc,os.path.join(XDIR,"direct.x512"),"p_direct"]).decode()
    m=subprocess.check_output([sys.executable,xc,os.path.join(XDIR,"micro.x512"),"p_micro"]).decode()
    m=m[m.index("static inline uint64_t mread"):];m=m[m.index("\n",m.index("return s;}"))+1:]
    open(os.path.join(d,"p_direct.h"),"w").write(h);open(os.path.join(d,"p_micro.h"),"w").write(m);open(os.path.join(d,"x512_lib.c"),"w").write(CSRC)
    subprocess.check_call(["gcc","-O3","-march=native","-fPIC","-shared","-I",d,"-o",LIB,os.path.join(d,"x512_lib.c")])
def aligne(x):
    """copie alignee sur 64 octets (exigence des chargements AVX-512 alignes)"""
    x=np.asarray(x,np.uint64);buf=np.empty(len(x)+8,np.uint64);o=(-buf.ctypes.data%64)//8;v=buf[o:o+len(x)];v[:]=x;return v
class X512Core:
    """un coeur X512 = un contexte de registres (X, R, micro-registres) dans ce processus"""
    def __init__(s):
        build();s.L=ct.CDLL(LIB);P=ct.c_void_p;s.L.x512_new.restype=P;s.L.x512_free.argtypes=[P]
        s.L.x512_run.restype=ct.c_uint64;s.L.x512_run.argtypes=[P,ct.c_int,P,P,P,ct.c_long,P];s.c=s.L.x512_new();s.ni=np.zeros(1,np.int64)
    def tpop(s,A,B,C,prog="micro"):
        n=len(A);pad=(-n)%16
        if pad:A,B,C=[np.concatenate([x,np.zeros(pad,np.uint64)]) for x in (A,B,C)]
        A,B,C=[x if (isinstance(x,np.ndarray) and x.dtype==np.uint64 and x.flags.c_contiguous and x.ctypes.data%64==0) else aligne(x) for x in (A,B,C)]  # pas de copie si deja aligne
        r=s.L.x512_run(s.c,0 if prog=="direct" else 1,A.ctypes.data,B.ctypes.data,C.ctypes.data,len(A)//8,s.ni.ctypes.data);return int(r),int(s.ni[0])
def vecs(seed,nw):
    g=np.random.default_rng(seed);return [g.integers(0,2**63,nw,dtype=np.uint64)*np.uint64(2)+g.integers(0,2,nw,dtype=np.uint64) for _ in range(3)]
def _worker(wid,inq,outq,prog):
    try:os.sched_setaffinity(0,{wid%os.cpu_count()})
    except Exception:pass
    core=X512Core();cache={};calc=0;ni=0;tc=0.0
    while True:
        job=inq.get()
        if job is None:break
        if isinstance(job,tuple) and job[0]=="INVALIDE":  # formule de Rene : seules les cles marquees sont effacees
            n=sum(1 for k in job[1] if cache.pop(k,None) is not None);outq.put((wid,[],calc,ni,tc,n));continue
        out=[]
        for pos,key,seed,nw in job:
            r=cache.get(key)
            if r is None:
                V=[aligne(v) for v in vecs(seed,nw)];t0=time.perf_counter();r,k=core.tpop(*V,prog=prog);tc+=time.perf_counter()-t0;cache[key]=r;calc+=1;ni+=k
            out.append((pos,r))
        outq.put((wid,out,calc,ni,tc,0))
class DeltaX512:
    """DELTA multi-coeurs X512 : un coeur X512 par worker, un worker par thread Xeon"""
    def __init__(s,n=None,prog="micro"):
        build()  # compile dans le parent : une erreur remonte ici au lieu de bloquer les workers
        ctx=mp.get_context("fork")  # fork explicite : les workers ne re-importent pas le script appelant
        s.n=n or os.cpu_count();s.outq=ctx.Queue();s.inq=[ctx.Queue() for _ in range(s.n)]
        s.P=[ctx.Process(target=_worker,args=(i,s.inq[i],s.outq,prog),daemon=True) for i in range(s.n)];[p.start() for p in s.P]
    def compute(s,reqs):
        parts=[[] for _ in range(s.n)]
        for pos,(key,seed,nw) in enumerate(reqs):parts[int(key[:8],16)%s.n].append((pos,key,seed,nw))
        for i in range(s.n):s.inq[i].put(parts[i])
        res=[None]*len(reqs);calc=0;ni=0;tc=0.0
        for _ in range(s.n):
            w,out,c,k,t,_=s.outq.get();calc+=c;ni+=k;tc=max(tc,t)
            for pos,r in out:res[pos]=r
        return res,calc,ni,tc
    def invalider(s,cles):
        """marque : n'efface que les cles modifiees, chacune dans le cache de son worker ; renvoie le nombre effacees"""
        parts=[[] for _ in range(s.n)]
        for k in cles:parts[int(k[:8],16)%s.n].append(k)
        for i in range(s.n):s.inq[i].put(("INVALIDE",parts[i]))
        return sum(s.outq.get()[5] for _ in range(s.n))
    def close(s):
        for q in s.inq:q.put(None)
        [p.join() for p in s.P]
if __name__=="__main__":
    build();print("=== DELTA x X512 : un coeur X512 par worker DELTA (%d threads Xeon) ==="%os.cpu_count());ok=True
    c=X512Core()
    for nw in (8,1000,16384):
        A,B,C=vecs(nw,nw);ref=int(np.unpackbits(((A&B)^C).view(np.uint8)).sum())
        for p in ("direct","micro"):
            r,_=c.tpop(A,B,C,p);ok&=r==ref
    print("coeur X512 seul : direct et micro-registres = reference numpy sur 8, 1000, 16384 mots : %s"%("IDENTIQUE" if ok else "FAIL"))
    M=600;uniq=60;rng=np.random.default_rng(3);ids=rng.integers(0,uniq,M);ids[:uniq]=np.arange(uniq);NW=16384*4
    reqs=[(hashlib.sha256(b"x512-%d"%s).hexdigest(),int(s),NW) for s in ids];sigs=set();base=None
    for n in sorted({1,os.cpu_count()}):
        for prog in ("direct","micro"):
            D=DeltaX512(n,prog);t=time.perf_counter();res,calc,ni,tc=D.compute(reqs);w=time.perf_counter()-t;D.close()
            sig=hashlib.sha256(str(res).encode()).hexdigest()[:12];sigs.add(sig);base=base or w
            print("workers %d | %-6s | total %6.1f ms | coeurs X512 actifs %5.2f ms -> %.2f G instr X512/s, %4.0f Gbit-op/s | %d calculs / %d demandes | gain total x%.2f | sig %s"%(n,prog,w*1e3,tc*1e3,ni/tc/1e9,3.0*calc*NW*64/tc/1e9,calc,M,base/w,sig))
    ok&=len(sigs)==1
    print("NOTE : temps incluant demarrage des workers et generation des vecteurs ; demandes repetees servies par le cache (pas comptees en calcul)")
    print("DELTA_X512_VALIDATION=%s"%("OK" if ok else "FAIL"))
