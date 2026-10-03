"""DELTA HYBRIDE : moteur booleen popcount((A_i ET B_j) XOR C), aiguilleur plages | multiplicateur 4x4 AVX-512 | repli scalaire."""
import os,ctypes as ct,numpy as np,subprocess,time,hashlib
HERE=os.path.dirname(os.path.abspath(__file__));SRC=os.path.join(HERE,"delta_hybrid.c");LIB=os.path.join(HERE,"libdelta_hybrid.so")
def build():
    if not os.path.exists(LIB) or os.path.getmtime(LIB)<os.path.getmtime(SRC):
        subprocess.check_call(["gcc","-O3","-march=x86-64-v2","-mpopcnt","-fPIC","-shared","-o",LIB,SRC])
    L=ct.CDLL(LIB);P=ct.c_void_p
    L.hy_prepare.restype=P;L.hy_prepare.argtypes=[P,ct.c_int,P,ct.c_int,P,ct.c_long]
    L.hy_run.argtypes=[P,P,ct.c_int];L.hy_stats.argtypes=[P,P];L.hy_free.argtypes=[P];return L
_L=None
def _lib():
    global _L
    if _L is None:_L=build()
    return _L
def _pad4(M):
    n=M.shape[0];r=(-n)%4
    return np.ascontiguousarray(M if r==0 else np.vstack([M,np.zeros((r,M.shape[1]),np.uint64)])),n
class Hybride:
    """A:(na,nw) B:(nb,nw) C:(nw,) en uint64. Preparation (aiguillage + compression) une fois, puis run() autant que voulu."""
    def __init__(s,A,B,C):
        L=_lib();s.A,s.na=_pad4(np.asarray(A,np.uint64));s.B,s.nb=_pad4(np.asarray(B,np.uint64));s.C=np.ascontiguousarray(C,np.uint64)
        s.h=L.hy_prepare(s.A.ctypes.data,s.A.shape[0],s.B.ctypes.data,s.B.shape[0],s.C.ctypes.data,s.C.shape[0])
        st=np.zeros(4,np.int64);L.hy_stats(s.h,st.ctypes.data);s.stats={"tuiles":int(st[0]),"tuiles_plages":int(st[1]),"octets_plages":int(st[2]),"avx512":bool(st[3])}
    def run(s,mode="auto"):
        out=np.zeros((s.A.shape[0],s.B.shape[0]),np.uint64);_lib().hy_run(s.h,out.ctypes.data,{"auto":0,"dense":1,"standard":2}[mode]);return out[:s.na,:s.nb]
    def __del__(s):
        try:_lib().hy_free(s.h)
        except Exception:pass
def compute_bool(A,B,C,mode="auto"):return Hybride(A,B,C).run(mode)
def zz(bits):
    """bits:(nq,shots) 0/1 -> correlations <Z_i Z_j> via le moteur hybride (popcount(i XOR j)=pc(i)+pc(j)-2pc(i ET j))."""
    nq,sh=bits.shape;P=np.packbits(bits.astype(np.uint8),axis=1,bitorder="little");P=np.ascontiguousarray(np.pad(P,((0,0),(0,(-P.shape[1])%8))));W=P.view(np.uint64)
    a=compute_bool(W,W,np.zeros(W.shape[1],np.uint64)).astype(np.int64);pc=np.diag(a);x=pc[:,None]+pc[None,:]-2*a;return 1-2*x/sh
def _ref(A,B,C):
    return np.array([[int(np.unpackbits(((A[i]&B[j])^C).view(np.uint8)).sum()) for j in range(len(B))] for i in range(len(A))],np.uint64)
def _gen(nw,struct,rng):
    if not struct:return rng.integers(0,2**63,nw,dtype=np.uint64)*np.uint64(2)+rng.integers(0,2,nw,dtype=np.uint64)
    bits=np.zeros(nw*64,np.uint8);p=0;b=int(rng.integers(0,2))
    while p<nw*64:r=1+int(rng.exponential(100000));bits[p:p+r]=b;p+=r;b^=1
    return np.packbits(bits,bitorder="little").view(np.uint64)
if __name__=="__main__":
    rng=np.random.default_rng(7);ok=True;L=_lib()
    print("=== DELTA HYBRIDE : validation ===")
    for na,nb,nw,mix in((3,5,1000,"aleatoire"),(4,4,16384*3+77,"mixte"),(16,16,16384*2,"structure")):
        tiles=[];T=(nw+16383)//16384
        def vec():
            parts=[_gen(min(16384,nw-t*16384),(mix=="structure") or (mix=="mixte" and t%2==0),rng) for t in range(T)];return np.concatenate(parts)
        A=np.array([vec() for _ in range(na)]);B=np.array([vec() for _ in range(nb)]);C=vec();H=Hybride(A,B,C);ref=_ref(A,B,C)
        r={m:H.run(m) for m in("auto","dense","standard")};good=all(np.array_equal(r[m],ref) for m in r);ok&=good
        print("%2dx%2d %7d mots %-9s | tuiles %d dont plages %d | auto/dense/standard = reference : %s"%(na,nb,nw,mix,H.stats["tuiles"],H.stats["tuiles_plages"],"OUI" if good else "NON"))
    b=rng.integers(0,2,(1,4000));bits=np.vstack([b,b,1-b,rng.integers(0,2,(1,4000))]);z=zz(bits)
    zok=abs(z[0,1]-1)<1e-12 and abs(z[0,2]+1)<1e-12 and abs(z[0,3])<0.1;ok&=zok;print("ZZ qubits : copie %.3f | inverse %.3f | independant %.3f -> %s"%(z[0,1],z[0,2],z[0,3],"OK" if zok else "FAIL"))
    print("=== DELTA HYBRIDE : banc 16x16, 64 tuiles mixtes ===");nw=16384*64
    A=np.array([np.concatenate([_gen(16384,t%2==0,rng) for t in range(64)]) for _ in range(16)]);B=np.array([np.concatenate([_gen(16384,t%2==0,rng) for t in range(64)]) for _ in range(16)]);C=np.concatenate([_gen(16384,t%2==0,rng) for t in range(64)])
    t=time.perf_counter();H=Hybride(A,B,C);tp=time.perf_counter()-t;res={}
    for m in("standard","dense","auto"):
        H.run(m);best=9e9
        for _ in range(5):t=time.perf_counter();o=H.run(m);best=min(best,time.perf_counter()-t)
        res[m]=(best,hashlib.sha256(o.tobytes()).hexdigest()[:12])
    ops=16*16*nw*64*3
    for m,(t,sg) in res.items():print("%-8s %8.2f ms | %6.0f Gbit-op/s%s | x%.1f vs standard | sig %s"%(m,t*1e3,ops/t/1e9," logiques" if m=="auto" else "",res["standard"][0]/t,sg))
    print("preparation (aiguillage + compression) %.0f ms, une seule fois | AVX-512 %s | memoire plages %.1f Ko"%(tp*1e3,H.stats["avx512"],H.stats["octets_plages"]/1024))
    ok&=len({v[1] for v in res.values()})==1
    print("DELTA_HYBRIDE_VALIDATION=%s"%("OK" if ok else "FAIL"))
