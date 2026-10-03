"""DELTA MARQUE : formule de Rene appliquee a tout DELTA.
Principe : toute ecriture pose une marque (1 octet) sur sa tuile ; au calcul, on lit les marques
et on ne recalcule QUE les tuiles marquees, les autres reutilisent leur resultat memorise.
Mesure sur telephone (moto g75) : jamais de perte (x1.00 a 100 % change), x13.8 a 1 %, jusqu'a x92 si calcul lourd.
Trois usages :
  TuilesMarquees(A,B,C,T)      -> vecteurs de mots, noyau quelconque fn(a,b,c)->int (popcount, X512, ...)
  HybrideMarque(A,B,C)          -> matrice du moteur hybride : seules les lignes de A modifiees sont recalculees
  marquer_cache(dict)           -> cache par cle (workers DELTA x X512) : invalidation ciblee au lieu de tout vider
"""
import os,time,numpy as np

def popcount_noyau(a,b,c):
    """noyau de reference DELTA : popcount((A ET B) XOR C)"""
    return int(np.unpackbits(((a&b)^c).view(np.uint8)).sum())

class TuilesMarquees:
    def __init__(s,A,B,C,T=256,noyau=popcount_noyau):
        s.A,s.B,s.C=[np.ascontiguousarray(x,np.uint64) for x in (A,B,C)];s.N=len(s.A);s.T=T;s.noyau=noyau
        s.nt=(s.N+T-1)//T;s.marque=np.ones(s.nt,np.uint8);s.cache=np.zeros(s.nt,np.int64);s.recalc_total=0
    def _t(s,i):return i//s.T
    def ecrire(s,i,a=None,b=None,c=None):
        """ecrit le mot i (une ou plusieurs entrees) et marque sa tuile"""
        if a is not None:s.A[i]=a
        if b is not None:s.B[i]=b
        if c is not None:s.C[i]=c
        s.marque[s._t(i)]=1
    def ecrire_bloc(s,deb,a=None,b=None,c=None):
        """ecrit un bloc contigu a partir de deb et marque toutes les tuiles touchees"""
        n=len(next(x for x in (a,b,c) if x is not None))
        for src,dst in ((a,s.A),(b,s.B),(c,s.C)):
            if src is not None:dst[deb:deb+n]=src
        s.marque[s._t(deb):s._t(deb+n-1)+1]=1
    def marquer_tout(s):s.marque[:]=1
    def evaluer(s):
        """total exact ; ne recalcule que les tuiles marquees. Renvoie (total, tuiles recalculees)"""
        idx=np.flatnonzero(s.marque)
        for t in idx:
            o=t*s.T;f=min(o+s.T,s.N);s.cache[t]=s.noyau(s.A[o:f],s.B[o:f],s.C[o:f])
        s.marque[idx]=0;s.recalc_total+=len(idx);return int(s.cache.sum()),len(idx)
    def complet(s):
        """calcul complet sans marque (reference)"""
        return sum(s.noyau(s.A[o:min(o+s.T,s.N)],s.B[o:min(o+s.T,s.N)],s.C[o:min(o+s.T,s.N)]) for o in range(0,s.N,s.T))

class HybrideMarque:
    """moteur hybride DELTA (aiguilleur plages | multiplicateur 4x4) avec marques par ligne de A et de B"""
    def __init__(s,A,B,C):
        from delta_hybrid import Hybride
        s.H=Hybride;s.A=np.array(A,np.uint64);s.B=np.array(B,np.uint64);s.C=np.ascontiguousarray(C,np.uint64)
        s.out=s.H(s.A,s.B,s.C).run();s.mA=np.zeros(len(s.A),np.uint8);s.mB=np.zeros(len(s.B),np.uint8)
    def ecrire_A(s,i,ligne):s.A[i]=ligne;s.mA[i]=1
    def ecrire_B(s,j,ligne):s.B[j]=ligne;s.mB[j]=1
    def run(s):
        """matrice exacte ; recalcule seulement les lignes/colonnes marquees. Renvoie (matrice, cellules recalculees)"""
        ia=np.flatnonzero(s.mA);jb=np.flatnonzero(s.mB);cel=0
        if len(ia):s.out[ia,:]=s.H(s.A[ia],s.B,s.C).run();cel+=len(ia)*len(s.B)
        if len(jb):s.out[:,jb]=s.H(s.A,s.B[jb],s.C).run();cel+=len(jb)*len(s.A)
        s.mA[:]=0;s.mB[:]=0;return s.out,cel

def marquer_cache(cache,cles_modifiees):
    """cache par cle des workers DELTA : on n'efface que les cles marquees (au lieu de vider tout le cache)"""
    n=0
    for k in cles_modifiees:
        if cache.pop(k,None) is not None:n+=1
    return n

if __name__=="__main__":
    ok=True;g=np.random.default_rng(7);N=1<<19;T=int(os.environ.get("TUILE",256))
    print("=== DELTA MARQUE : formule de Rene sur tout DELTA (tuile %d mots, %d tuiles, %.1f Mo) ==="%(T,(N+T-1)//T,N*24/1048576))
    A,B,C=[g.integers(0,2**63,N,dtype=np.uint64) for _ in range(3)];tm=TuilesMarquees(A,B,C,T);tm.evaluer()
    for p in (1.0,0.10,0.01,0.0):
        k=int(p*tm.nt+0.5);ti=g.choice(tm.nt,k,replace=False) if k<tm.nt else np.arange(tm.nt)
        for t in ti:tm.ecrire(int(t)*T+int(g.integers(0,T)),a=int(g.integers(0,2**63)))
        t0=time.perf_counter();ref=tm.complet();tc=time.perf_counter()-t0
        tm.marque[ti]=1;t0=time.perf_counter();v,rc=tm.evaluer();tv=time.perf_counter()-t0;bon=(v==ref);ok&=bon
        print("vecteurs  change %5.1f %% | complet %8.2f ms | marque %8.2f ms (%5d tuiles) | x%7.1f | %s"%(p*100,tc*1e3,tv*1e3,rc,tc/max(tv,1e-9),"exact" if bon else "FAUX"))
    try:
        A=g.integers(0,2**63,(256,64),dtype=np.uint64);B=g.integers(0,2**63,(256,64),dtype=np.uint64);C=g.integers(0,2**63,64,dtype=np.uint64)
        from delta_hybrid import Hybride
        hm=HybrideMarque(A,B,C)
        for nl in (64,8,1):
            for i in g.choice(256,nl,replace=False):hm.ecrire_A(int(i),g.integers(0,2**63,64,dtype=np.uint64))
            t0=time.perf_counter();ref=Hybride(hm.A,hm.B,hm.C).run();tc=time.perf_counter()-t0
            t0=time.perf_counter();m,cel=hm.run();tv=time.perf_counter()-t0;bon=bool((m==ref).all());ok&=bon
            print("hybride   %3d lignes A modifiees /256 | complet %7.2f ms | marque %7.2f ms (%6d cellules) | x%6.1f | %s"%(nl,tc*1e3,tv*1e3,cel,tc/max(tv,1e-9),"exact" if bon else "FAUX"))
    except ImportError:print("hybride   delta_hybrid absent : partie sautee")
    c={"k%d"%i:i for i in range(1000)};n=marquer_cache(c,["k1","k2","zz"]);b=(n==2 and len(c)==998);ok&=b
    print("cache     invalidation ciblee : %d cles effacees, %d gardees | %s"%(n,len(c),"exact" if b else "FAUX"))
    print("note : marque = travail evite (resultat identique au calcul complet), pas des Gbit-op/s\nDELTA_MARQUE=%s"%("OK" if ok else "FAIL"))
