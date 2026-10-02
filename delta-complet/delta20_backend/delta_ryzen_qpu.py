import os,json,time,subprocess,warnings;warnings.filterwarnings("ignore")
from qiskit import qasm2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_sampler import Sampler
from delta_upgrades import mem_guard
assert mem_guard(),"MEMOIRE INSUFFISANTE"
G=os.path.expanduser("~/delta_guest")
cmd=["sudo","timeout","120","qemu-system-x86_64","-enable-kvm","-cpu","host,vendor=AuthenticAMD,model-id=AMD Ryzen 9 9950X3D 16-Core Processor","-smp","2","-m","512","-mem-prealloc","-kernel",G+"/vmlinuz","-initrd",G+"/initrd_qpu.gz","-append","console=ttyS0 quiet panic=-1","-nographic","-no-reboot"]
t=time.perf_counter();out=subprocess.run(cmd,stdin=subprocess.DEVNULL,capture_output=True,text=True,errors="ignore").stdout.replace("\r","");tp=time.perf_counter()-t
import re,hashlib;prep=re.findall(r"QPU_PREP .*?FIN_PREP",out);print(prep[0] if prep else "QPU_PREP absent");print("ETAPE 1 socle Ryzen emule : circuit prepare en %.1f s (demarrage VM compris)"%tp)
m=re.search(r"QASM:(OPENQASM.*?):SIG:([0-9a-f]{16}):FIN_QASM",out);q=m.group(1);assert hashlib.sha256(q.encode()).hexdigest()[:16]==m.group(2),"SIGNATURE INVALIDE";print("SIGNATURE VALIDE",m.group(2));qc=qasm2.loads(q);print("ETAPE 2 circuit recu du socle :",qc.num_qubits,"qubits,",qc.size(),"portes")
assert os.environ.get("IQP_API_TOKEN") and os.environ.get("IQP_INSTANCE_CRN"),"SECRETS IBM ABSENTS"
s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"]);bk=None;lay=None
if os.path.exists("delta_qpairs_best.json"):d=json.load(open("delta_qpairs_best.json"));bk=d.get("backend");lay=d["pairs"][0]
b=s.backend(bk) if bk else s.least_busy(operational=True,simulator=False)
t=time.perf_counter();job=Sampler(mode=b).run([generate_preset_pass_manager(optimization_level=1,backend=b,initial_layout=lay).run(qc)],shots=1024)
print("ETAPE 3 Codespace -> IBM %s paire=%s JOB=%s"%(b.name,lay,job.job_id()),flush=True)
d=job.result()[0].data;c={}
for n in dir(d):
    if not n.startswith("_") and hasattr(getattr(d,n),"get_counts"):c=getattr(d,n).get_counts()
tq=time.perf_counter()-t;f=(c.get("00",0)+c.get("11",0))/sum(c.values())
print("ETAPE 4 resultat QPU en %.1f s : %s | FIDELITE_BELL=%.3f | part du pilotage Ryzen = %.1f%% du temps total"%(tq,c,f,100*tp/(tp+tq)))
open("delta_ryzen_qpu.jsonl","a").write(json.dumps({"date":time.strftime("%Y-%m-%d %H:%M"),"socle":prep[0] if prep else None,"prep_s":tp,"backend":b.name,"job_id":job.job_id(),"pair":lay,"counts":c,"fidelite":f,"qpu_s":tq})+"\n")
print("RYZEN_QPU_VALIDATION=%s"%("OK" if f>0.85 else "FAIL"))
