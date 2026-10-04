import os,sys,time,json,math
from qiskit import QuantumCircuit,transpile
SH=1000;NS=[5,10,20]
def ghz(n,x):
    q=QuantumCircuit(n);q.h(0)
    for i in range(n-1):q.cx(i,i+1)
    if x:q.h(range(n))
    q.measure_all();return q
def chsh(ta,tb):
    q=QuantumCircuit(2);q.h(0);q.cx(0,1);q.ry(-ta,0);q.ry(-tb,1);q.measure_all();return q
def ro(n,one):
    q=QuantumCircuit(n)
    if one:q.x(range(n))
    q.measure_all();return q
A=[0,math.pi/2];B=[math.pi/4,-math.pi/4]
C=[ghz(n,x) for n in NS for x in (0,1)]+[chsh(a,b) for a in A for b in B]+[ro(5,0),ro(5,1)]
def analyse(cs):
    r={};k=0
    for n in NS:
        z,x=cs[k],cs[k+1];k+=2
        p=(z.get("0"*n,0)+z.get("1"*n,0))/SH
        par=sum(v*(-1)**s.count("1") for s,v in x.items())/SH
        f=(p+par)/2;r["GHZ%d"%n]={"pop":round(p,3),"parite":round(par,3),"F":round(f,3),"intrique":f>0.5}
    E=[sum(v*(1 if s[0]==s[1] else -1) for s,v in c.items())/SH for c in cs[k:k+4]];k+=4
    r["CHSH_S"]=round(E[0]+E[1]+E[2]-E[3],3)
    e0=1-cs[k].get("0"*5,0)/SH;e1=1-cs[k+1].get("1"*5,0)/SH
    r["lecture_err"]={"0":round(e0,3),"1":round(e1,3)};return r
def show(name,r,extra=""):
    g=" ".join("G%s F=%.2f%s"%(k[3:],v["F"],"✓" if v["intrique"] else "✗") for k,v in r.items() if k.startswith("GHZ"))
    print("%s | %s | S=%.3f%s | lect %.3f/%.3f %s"%(name,g,r["CHSH_S"],"✓" if r["CHSH_S"]>2 else "✗",r["lecture_err"]["0"],r["lecture_err"]["1"],extra))
if __name__=="__main__":
    if "--sim" in sys.argv:
        from qiskit.primitives import StatevectorSampler
        res=StatevectorSampler().run(C,shots=SH).result()
        r=analyse([p.data.meas.get_counts() for p in res]);show("SIMULATEUR",r);sys.exit()
    from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2
    svc=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
    bks=svc.backends(operational=True,simulator=False);jobs={}
    for b in bks:jobs[b.name]=(SamplerV2(mode=b).run(transpile(C,b,optimization_level=3),shots=SH),time.time())
    ok=0
    for name,(j,t0) in jobs.items():
        res=j.result();dt=time.time()-t0;r=analyse([p.data.meas.get_counts() for p in res])
        try:qs=j.usage()
        except Exception:qs=None
        r.update({"date":time.strftime("%Y-%m-%d %H:%M"),"backend":name,"shots":SH,"qpu_s":qs,"total_s":round(dt,1),"job":j.job_id(),"statut":"mesure"})
        show(name,r,"| QPU %ss | %s"%(qs,j.job_id()));open("delta_qpu_full.jsonl","a").write(json.dumps(r,ensure_ascii=False)+"\n")
        ok+=r["CHSH_S"]>2 and r["GHZ5"]["intrique"]
    print("VERDICT=%s %d/%d QPU (Bell S>2 et GHZ5 intrique)"%("OK" if ok==len(jobs) else "PARTIEL",ok,len(jobs)))
