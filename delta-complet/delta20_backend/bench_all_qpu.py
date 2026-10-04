import os,time,json,sys
from qiskit import QuantumCircuit,transpile
from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2
N=int(sys.argv[1]) if len(sys.argv)>1 else 5;SH=1000
svc=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
bks=[b for b in svc.backends(operational=True,simulator=False)]
print("QPU a portee :",", ".join("%s(%dq,file %s)"%(b.name,b.num_qubits,b.status().pending_jobs) for b in bks))
qc=QuantumCircuit(N);qc.h(0)
for i in range(N-1):qc.cx(i,i+1)
qc.measure_all()
jobs={}
for b in bks:
    t=transpile(qc,b,optimization_level=3);jobs[b.name]=(SamplerV2(mode=b).run([t],shots=SH),time.time(),t.depth())
ok=0
for name,(j,t0,d) in jobs.items():
    c=j.result()[0].data.meas.get_counts();dt=time.time()-t0
    p=(c.get("0"*N,0)+c.get("1"*N,0))/SH
    try:qs=j.metrics()["usage"]["quantum_seconds"]
    except Exception:qs=None
    r={"date":time.strftime("%Y-%m-%d %H:%M"),"backend":name,"ghz":N,"shots":SH,"depth":d,"population":round(p,3),"quantum_s":qs,"total_s":round(dt,1),"job":j.job_id(),"statut":"mesure"}
    print("%s | GHZ%d | pop %.3f | QPU %ss | total %.0fs | job %s"%(name,N,p,qs,dt,j.job_id()))
    open("delta_all_qpu.jsonl","a").write(json.dumps(r)+"\n");ok+=p>0.5
print("VERDICT=%s %d/%d QPU pop>0.5"%("OK" if ok==len(jobs) else "PARTIEL",ok,len(jobs)))
