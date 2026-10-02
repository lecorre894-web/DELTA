import json,time,platform,torch
CUDA=torch.cuda.is_available();DEVS=["cpu"]+(["cuda"] if CUDA else []);torch.set_num_threads(16)
def sync(d):
    if d=="cuda":torch.cuda.synchronize()
def gemm(d,dt,n=4096,amp=False):
    a=torch.randn(n,n,device=d,dtype=torch.float32 if amp else dt);b=torch.randn_like(a)
    def f():
        if amp:
            with torch.autocast(device_type=d,dtype=torch.float16 if d=="cuda" else torch.bfloat16):return a@b
        return a@b
    f();sync(d);best=9e9
    for _ in range(3):
        t=time.perf_counter();f();sync(d);best=min(best,time.perf_counter()-t)
    return 2*n**3/best/1e9
def h(s,q):
    v=s.view(-1,2,2**q);a=v[:,0].clone();b=v[:,1];r=0.7071067811865476;v[:,0]=(a+b)*r;v[:,1]=(a-b)*r
def cx(s,n,c,t):
    if c>t:v=s.view(2**(n-c-1),2,2**(c-t-1),2,2**t);x=v[:,1,:,0,:].clone();v[:,1,:,0,:]=v[:,1,:,1,:];v[:,1,:,1,:]=x
    else:v=s.view(2**(n-t-1),2,2**(t-c-1),2,2**c);x=v[:,0,:,1,:].clone();v[:,0,:,1,:]=v[:,1,:,1,:];v[:,1,:,1,:]=x
def ghz(d,n):
    s=torch.zeros(2**n,dtype=torch.complex64,device=d);s[0]=1;sync(d);t=time.perf_counter();h(s,0)
    for i in range(n-1):cx(s,n,i,i+1)
    sync(d);w=time.perf_counter()-t;p0=abs(s[0].item())**2;p1=abs(s[-1].item())**2;del s
    if d=="cuda":torch.cuda.empty_cache()
    return w,p0,p1
R={"date":time.strftime("%Y-%m-%d %H:%M"),"cpu":platform.processor(),"gpu":torch.cuda.get_device_name(0) if CUDA else None,"mesures":{},"route":{}}
print("=== DELTA STATION : routage mesure CPU/GPU | %s | %s ==="%(R["cpu"],R["gpu"]))
for name,dt,amp in (("FP64",torch.float64,False),("FP32",torch.float32,False),("AMP",None,True)):
    m={d:gemm(d,dt,amp=amp) for d in DEVS};R["mesures"][name]=m;best=max(m,key=m.get);R["route"][name]=best
    print("TACHE %-4s "%name+"  ".join("%s=%9.1f GF"%(d,v) for d,v in m.items())+"  -> ROUTE %s"%best.upper())
lim={}
for d in DEVS:
    top=0
    for n in ([20,24,26,28] if d=="cpu" else [20,24,26,28,29,30]):
        try:w,p0,p1=ghz(d,n)
        except RuntimeError:print("SIM GHZ %-4s n=%2d HORS MEMOIRE (plafond atteint)"%(d,n));break
        ok=abs(p0-.5)<1e-3 and abs(p1-.5)<1e-3;print("SIM GHZ %-4s n=%2d %8.3f s  P(0..0)=%.4f P(1..1)=%.4f %s"%(d,n,w,p0,p1,"OK" if ok else "FAIL"))
        if not ok:break
        top=n;R["mesures"].setdefault("SIM_"+d,{})[n]=w
    lim[d]=top
c=[(R["mesures"]["SIM_"+d][26],d) for d in DEVS if 26 in R["mesures"].get("SIM_"+d,{})]
if c:R["route"]["SIM"]=min(c)[1];print("SIM 26 qubits -> ROUTE %s (%s)"%(min(c)[1].upper(),"  ".join("%s=%.3f s"%(d,w) for w,d in c)))
R["route"]["QPU"]="IBM cloud (inchange)";R["plafond_qubits"]=lim;print("PLAFOND SIMULATEUR EXACT (qubits) :",lim,"| Codespace = 26")
json.dump(R,open("delta_station_route.json","w"),indent=1)
print("ROUTAGE ECRIT -> delta_station_route.json");print("STATION_VALIDATION=%s"%("OK" if len(R["route"])>=4 and max(lim.values())>=26 else "FAIL"))
