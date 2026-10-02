import numpy as np,time,json,os,hashlib
from delta_core import DeltaCore
H=os.path.dirname(os.path.abspath(__file__));S=json.load(open(os.path.join(H,"delta_station_route.json")))
EMU={"cpu":("RYZEN_9950X3D_EMU",16,32,lambda i:(i%32)//2),"cuda":("RTX_4080_EMU",76,76,lambda i:i%76)}
DT={"FP64":(np.float64,1e-12),"FP32":(np.float32,1e-5),"AMP":(np.float16,2e-3)}
def platform(task,N,prec,seed=7):
    dev=S["route"][prec];name,U,par,unit=EMU[dev];dt,tol=DT[prec];T=max(1,N//par)
    r=np.random.default_rng(seed);A=r.random((N,N)).astype(dt);B=r.random((N,N)).astype(dt)
    cdt=np.float32 if prec=="AMP" else dt;Ac=A.astype(cdt);Bc=B.astype(cdt);C=np.empty((N,N),cdt);load=np.zeros(U)
    t=time.perf_counter()
    for i,lo in enumerate(range(0,N,T)):hi=min(N,lo+T);C[lo:hi]=Ac[lo:hi]@Bc;load[unit(i)]+=hi-lo
    w=time.perf_counter()-t;ref=A.astype(np.float64)@B.astype(np.float64);err=float(np.abs(C-ref).max()/np.abs(ref).max())
    tm=2*N**3/(S["mesures"][prec][dev]*1e9)
    return {"engine":"DELTA_PLATFORM","task":task,"precision":prec,"route":dev,"emulated":name,"host":"XEON_PHYSICAL_MEASURED","exact":err<tol,"err":err,"wall_host_s":w,"real_s":tm,"real_source":"MESURE STATION 2 OCT","gap":w/tm,"units_used":int((load>0).sum()),"units":U,"sig":hashlib.sha256(np.ascontiguousarray(C).tobytes()).hexdigest()[:16]}
DeltaCore.compute_platform=lambda self,task,N,prec:platform(task,N,prec)
if __name__=="__main__":
    D=DeltaCore();ok=True;out=[]
    print("=== DELTA PLATEFORME VIRTUELLE INTRA-NOEUD : D.compute_platform() ===")
    for p in ("FP64","FP32","AMP"):
        r=D.compute_platform("matmul",512,p);ok&=r["exact"];out.append(r)
        print("%-4s -> %-17s %s err=%.1e | unites %d/%d | Xeon %.4f s | vrai %.2e s [%s] | ecart x%.0f | sig %s"%(p,r["emulated"],"EXACT" if r["exact"] else "FAUX",r["err"],r["units_used"],r["units"],r["wall_host_s"],r["real_s"],r["real_source"],r["gap"],r["sig"]))
    json.dump(out,open("delta_platform_result.json","w"),indent=1);print("DELTA_PLATFORM_VALIDATION=%s"%("OK" if ok else "FAIL"))
