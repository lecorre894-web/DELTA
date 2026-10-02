import os,sys,math,json,time,warnings
warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit
SH=2048;PHYS=[152,153];SIM=os.environ.get("INT_SIM","0")=="1"
def chsh(cx):
    qs=[]
    for a in (0,math.pi/2):
        for b in (math.pi/4,-math.pi/4):
            q=QuantumCircuit(2);q.h(0)
            if cx:q.cx(0,1)
            q.ry(-a,0);q.ry(-b,1);q.measure_all();qs.append(q)
    return qs
qs=chsh(True)+chsh(False)
if SIM:
    from delta_qpu_cache import DeltaQPUCache;D=DeltaQPUCache(":memory:");C=[D._sim(q,SH)[0] for q in qs];bk="local";jid="sim"
else:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2;from qiskit_ibm_runtime.executor_sampler import Sampler as _ExecSampler
    s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
    b=s.backend(os.environ.get("INT_BACKEND","ibm_marrakesh"));pm=generate_preset_pass_manager(optimization_level=1,backend=b,initial_layout=PHYS)
    job=_ExecSampler(mode=b).run([pm.run(q) for q in qs],shots=SH);print("SOUMIS %s JOB=%s"%(b.name,job.job_id()),flush=True);C=[]
    for r in job.result():
        d=r.data
        for n in dir(d):
            if not n.startswith("_") and hasattr(getattr(d,n),"get_counts"):C.append(getattr(d,n).get_counts())
    bk=b.name;jid=job.job_id()
def S(cs):
    E=[(c.get("00",0)+c.get("11",0)-c.get("01",0)-c.get("10",0))/sum(c.values()) for c in cs];v=E[0]+E[1]+E[2]-E[3]
    return v,(v-2)/math.sqrt(sum((1-e*e)/SH for e in E))
s1,g1=S(C[:4]);s0,g0=S(C[4:])
print("=== TEST D'INTRICATION A/B (meme paire %s, meme job) ==="%PHYS)
print("A AVEC porte CX (intrication) : S=%.3f (%+.1f sigma vs 2) -> %s"%(s1,g1,"INTRIQUE" if s1>2 and g1>5 else "NON PROUVE"))
print("B SANS porte CX (temoin)      : S=%.3f (%+.1f sigma vs 2) -> %s"%(s0,g0,"classique, comme attendu" if s0<=2 else "ANOMALIE"))
print("QPU=%s JOB=%s"%(bk,jid))
print("INTRICATION_VALIDATION=%s"%("OK" if s1>2 and g1>5 and s0<=2 else "FAIL"))
