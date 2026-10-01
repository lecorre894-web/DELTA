import os,sys,time,json,sqlite3,numpy as np
from delta_pack import unpack
from delta_qpu_cache import DB
DOMAIN=10**12;M=int(os.environ.get("VREQ","4000000"))
def mix(x):
    x=(x+np.uint64(0x9E3779B97F4A7C15));x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9);x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB);return x^(x>>np.uint64(31))
rows=sqlite3.connect(DB).execute("select k,mode,counts,origin from r order by created").fetchall()
if not rows:sys.exit("CACHE VIDE : lance d'abord delta_qpu_cache.py")
R=[];shots_real=0;qpu=0
for k,mode,c,o in rows:
    c=unpack(c);o=json.loads(o);keys=list(c);p=np.array([c[x] for x in keys],float);shots_real+=int(p.sum());qpu+=mode=="qpu"
    R.append({"name":o.get("backend",o.get("engine"))+"/%dq"%len(keys[0]),"mode":mode,"keys":keys,"cdf":np.cumsum(p/p.sum()),"p":p/p.sum(),"job":o.get("job_id","-")})
K=len(R)
print("=== DELTA VIRTUAL BENCH : domaine logique 1T adosse aux resultats reels du cache ===")
print("RESULTATS_REELS=%d (dont QPU physiques=%d) SHOTS_REELS_TOTAL=%d DOMAINE_VIRTUEL=%.0e adresses"%(K,qpu,shots_real,DOMAIN))
rng=np.random.default_rng(42);w=[];np.seterr(over="ignore")
for rep in range(3):
    t0=time.perf_counter();addr=rng.integers(0,DOMAIN,M,dtype=np.uint64);h=mix(addr);idx=(h%np.uint64(K)).astype(np.int64);u=rng.random(M);out=np.empty(M,np.int32)
    for j in range(K):
        sel=idx==j;out[sel]=np.minimum(np.searchsorted(R[j]["cdf"],u[sel]),len(R[j]["cdf"])-1)
    w.append(time.perf_counter()-t0)
t=sorted(w)[1];rate=M/t
print("VIRTUEL requetes=%d WALL_MED=%.3f s DEBIT=%.1f M reponses/s LATENCE=%.1f ns/reponse"%(M,t,rate/1e6,t/M*1e9))
tv=[]
for j in range(K):
    sel=idx==j;n=int(sel.sum())
    if n>1000:
        f=np.bincount(out[sel],minlength=len(R[j]["p"]))/n;tv.append(0.5*np.abs(f-R[j]["p"]).sum())
        print("FIDELITE %-28s requetes=%8d distance_TV=%.4f JOB=%s"%(R[j]["name"],n,tv[-1],R[j]["job"]))
print("AMPLIFICATION=%.0fx reponses virtuelles par shot reel (re-echantillonnage : AUCUNE information quantique nouvelle)"%(M/shots_real))
print("PROJECTION servir les 1T adresses sur 1 coeur = %.1f h [PROJECTION, NON EXECUTEE]"%(DOMAIN/rate/3600))
print("VIRTUAL_VALIDATION=%s (distance_TV max %.4f < 0.01 = couche virtuelle fidele au reel)"%("OK" if tv and max(tv)<0.01 else "FAIL",max(tv) if tv else -1))
