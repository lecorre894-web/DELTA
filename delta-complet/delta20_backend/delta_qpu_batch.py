import os,json,time,sqlite3,hashlib,warnings;warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit,qasm2
SIM=os.environ.get("QPU_SIM")=="1";SH=1024
def ghz(n):
    q=QuantumCircuit(n);q.h(0)
    for i in range(n-1):q.cx(i,i+1)
    q.measure_all();return q
C={"BELL":ghz(2),"GHZ3":ghz(3),"GHZ4":ghz(4)}
db=sqlite3.connect("delta_qpu_batch_cache.db");db.execute("create table if not exists r(key text primary key,backend text,job text,counts text,t real)")
if SIM:
    from qiskit.primitives import StatevectorSampler as S;bk="local_statevector";run=lambda qs:S(seed=7).run(qs,shots=SH)
else:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.executor_sampler import Sampler
    s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"]);b=s.least_busy(operational=True,simulator=False);bk=b.name
    pm=generate_preset_pass_manager(optimization_level=1,backend=b);run=lambda qs:Sampler(mode=b).run([pm.run(q) for q in qs],shots=SH)
key=lambda q:hashlib.sha256((qasm2.dumps(q)+"|"+bk+"|%d"%SH).encode()).hexdigest()
res={};miss=[]
for n,q in C.items():
    r=db.execute("select job,counts from r where key=?",(key(q),)).fetchone()
    if r:res[n]=(json.loads(r[1]),r[0],True)
    else:miss.append(n)
print("=== DELTA QPU BATCH sur %s : %d circuits, %d dans le cache, %d a envoyer en UN job ==="%(bk,len(C),len(C)-len(miss),len(miss)))
if miss:
    t=time.perf_counter();job=run([C[n] for n in miss]);jid="sim" if SIM else job.job_id();out=job.result()
    for n,r in zip(miss,out):
        d=r.data;c=[getattr(d,a).get_counts() for a in dir(d) if not a.startswith("_") and hasattr(getattr(d,a),"get_counts")][0]
        db.execute("insert or replace into r values(?,?,?,?,?)",(key(C[n]),bk,jid,json.dumps(c),time.time()));res[n]=(c,jid,False)
    db.commit();print("UN SEUL JOB %s pour %d circuits en %.1f s (au lieu de %d files d'attente)"%(jid,len(miss),time.perf_counter()-t,len(miss)))
ok=True
for n,(c,j,h) in res.items():
    k=len(next(iter(c)));p=(c.get("0"*k,0)+c.get("1"*k,0))/sum(c.values());ok&=p>0.7
    print("%-5s %s | population GHZ %.3f | job %s"%(n,"CACHE" if h else "QPU  ",p,j))
print("QPU_BATCH_VALIDATION=%s"%("OK" if ok else "FAIL"))
