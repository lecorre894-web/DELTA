import numpy as np,time,json,hashlib,platform,os
def cpu():
    try:return [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except:return platform.processor()
def best(f,rep=5):
    f();b=9e9
    for _ in range(rep):t=time.perf_counter();f();b=min(b,time.perf_counter()-t)
    return b
N=1024;rng=np.random.default_rng(7);R={}
A64=rng.random((N,N));B64=rng.random((N,N));A32=A64.astype(np.float32);B32=B64.astype(np.float32)
STD={"ryzen_fp64":765.9,"ryzen_fp32":1674.9,"rtx_fp32":37523.9,"rtx_amp":83952.7,"rtx_fp64":750.5}
print("=== COUCHE 0 : SOCLE XEON (natif) ===")
h={"cpu":cpu(),"threads":os.cpu_count()}
for p,(A,B) in (("fp64",(A64,B64)),("fp32",(A32,B32))):h[p]=2*N**3/best(lambda:A@B)/1e9
v=rng.random(1<<24,dtype=np.float32);w=rng.random(1<<24,dtype=np.float32);h["bw_GBs"]=12*(1<<24)/best(lambda:v+w)/1e9
print("HOTE %s | %d threads | FP64 %.1f GF | FP32 %.1f GF | bande passante %.1f Go/s"%(h["cpu"],h["threads"],h["fp64"],h["fp32"],h["bw_GBs"]));R["socle"]=h
print("\n=== COUCHE 1 : SUPPORTS EMULES (meme calcul, decoupe comme la vraie puce) ===")
def emu(A,B,par,U,unit):
    T=max(1,N//par);C=np.empty((N,N),A.dtype);load=np.zeros(U)
    def run():
        for i,lo in enumerate(range(0,N,T)):hi=min(N,lo+T);C[lo:hi]=A[lo:hi]@B;load[unit(i)]+=1
    return best(run,3),C,int((load>0).sum())
E={}
for nom,par,U,unit,p,(A,B),std in (("RYZEN_9950X3D_EMU",32,16,lambda i:(i%32)//2,"fp64",(A64,B64),STD["ryzen_fp64"]),("RYZEN_9950X3D_EMU",32,16,lambda i:(i%32)//2,"fp32",(A32,B32),STD["ryzen_fp32"]),("RTX_4080_EMU",76,76,lambda i:i%76,"fp32",(A32,B32),STD["rtx_fp32"])):
    t,C,u=emu(A,B,par,U,unit);ref=A64@B64;err=float(np.abs(C-ref).max()/np.abs(ref).max());gf=2*N**3/t/1e9
    k="%s_%s"%(nom,p);E[k]={"GF":gf,"overhead_vs_socle":h[p]/gf,"vs_standard":std/gf,"err":err,"units":u}
    print("%-17s %s | %.1f GF | surcout emulation x%.2f vs socle | %d unites | err %.0e | standard reel %.0f GF -> x%.0f"%(nom,p.upper(),gf,h[p]/gf,u,err,std,std/gf))
R["emules"]=E
print("\n=== COUCHE 2 : NUAGE DELTA (routage + cache + signature) ===")
ROUTE={"FP64":"RYZEN_EMU","FP32":"RTX_EMU","QPU":"IBM"};cache={}
def compute(task,prec,n=256,seed=1):
    key=hashlib.sha256(("%s|%s|%d|%d"%(task,prec,n,seed)).encode()).hexdigest()
    if key in cache:return cache[key],True
    r=np.random.default_rng(seed);dt=np.float64 if prec=="FP64" else np.float32;a=r.random((n,n)).astype(dt);c=a@a
    out={"route":ROUTE[prec],"sig":hashlib.sha256(c.tobytes()).hexdigest()[:12]};cache[key]=out;return out,False
t0=time.perf_counter();compute("mm","FP32");miss=time.perf_counter()-t0
t0=time.perf_counter()
for _ in range(10000):compute("mm","FP32")
hitt=(time.perf_counter()-t0)/10000
t0=time.perf_counter()
for _ in range(10000):ROUTE["FP32"]
rt=(time.perf_counter()-t0)/10000
det=compute("mm","FP64",seed=9)[0]["sig"]==compute("mm","FP64",seed=9)[0]["sig"]
R["nuage"]={"calcul_miss_s":miss,"cache_hit_s":hitt,"routage_s":rt,"gain_cache":miss/hitt,"deterministe":det}
print("calcul (miss) %.2e s | cache (hit) %.2e s -> gain x%.0f | decision de routage %.0f ns | signature deterministe %s"%(miss,hitt,miss/hitt,rt*1e9,det))
print("\n=== COUCHE 3 : PLACEMENT DU QPU ===")
def ghz_prep(n):return "\n".join(["h q[0];"]+["cx q[%d],q[%d];"%(i,i+1) for i in range(n-1)])
Q={}
for nom,cout in (("pilote par le socle Xeon",1.0),("pilote par Ryzen emule",E["RYZEN_9950X3D_EMU_fp64"]["overhead_vs_socle"])):
    t=best(lambda:[ghz_prep(n) for n in (5,10,20,50,100)])*cout;Q[nom]=t;print("%-26s preparation 5 circuits GHZ : %.1f us"%(nom,t*1e6))
qpu_job=11.5;Q["part_pilotage"]=Q["pilote par Ryzen emule"]/qpu_job
print("job QPU reel (mesure 1 oct) ~%.1f s -> pilotage emule = %.1e du job : placement NEUTRE [cout emule = estimation via surcout couche 1]"%(qpu_job,Q["part_pilotage"]));R["qpu"]=Q
print("\n=== ETENDUE DU NUAGE : debit effectif selon le taux de repetition ===")
n=256;fl=2*n**3;X=[]
for rep in (0.0,0.5,0.9,0.99,0.999):
    cache.clear();r=np.random.default_rng(3);M=2000;uniq=max(1,int(round(M*(1-rep))));ids=r.integers(0,uniq,M);ids[:uniq]=np.arange(uniq)
    t0=time.perf_counter()
    for s in ids:compute("mm","FP32",n=n,seed=int(s))
    w=time.perf_counter()-t0;eff=M*fl/w/1e9;X.append({"repetition":rep,"uniques":uniq,"GF_effectif":eff,"x_socle":eff/h["fp32"]})
    print("repetition %5.1f%% | %4d demandes, %4d vrais calculs | debit effectif %8.1f GF | x%.1f le socle %s"%(rep*100,M,uniq,eff,eff/h["fp32"],"| depasse le vrai Ryzen FP32" if eff>STD["ryzen_fp32"] else ""))
R["etendue_nuage"]=X;json.dump(R,open("delta_stack_sim_result.json","w"),indent=1)
print("NOTE: debit effectif = calculs SERVIS / temps ; seuls les vrais calculs sont executes, le reste vient du cache signe.")
print("STACK_SIM_VALIDATION=%s"%("OK" if all(e["err"]<1e-5 for e in E.values()) and det else "FAIL"))
