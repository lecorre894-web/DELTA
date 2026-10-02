import os,json,time,warnings;warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_sampler import Sampler
assert os.environ.get("IQP_API_TOKEN") and os.environ.get("IQP_INSTANCE_CRN"),"SECRETS IBM ABSENTS"
s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"]);lay=None;bk=None
if os.path.exists("delta_qpairs_best.json"):d=json.load(open("delta_qpairs_best.json"));bk=d.get("backend");lay=d["pairs"][0]
b=s.backend(bk) if bk else s.least_busy(operational=True,simulator=False)
q=QuantumCircuit(2);q.h(0);q.cx(0,1);q.measure_all()
t=generate_preset_pass_manager(optimization_level=1,backend=b,initial_layout=lay).run(q)
job=Sampler(mode=b).run([t],shots=1024);print("SOUMIS %s paire=%s JOB=%s (file d'attente possible)"%(b.name,lay,job.job_id()),flush=True)
c=job.result()[0].data.meas.get_counts();f=(c.get("00",0)+c.get("11",0))/sum(c.values())
print("COUNTS",c);print("FIDELITE_BELL (00+11) = %.3f"%f)
open("delta_exec_proof.jsonl","a").write(json.dumps({"date":time.strftime("%Y-%m-%d %H:%M"),"backend":b.name,"job_id":job.job_id(),"pair":lay,"counts":c,"p00_11":f,"sampler":"executor_sampler.Sampler"})+"\n")
print("EXEC_SAMPLER_QPU_VALIDATION=%s"%("OK" if f>0.85 else "FAIL"))
