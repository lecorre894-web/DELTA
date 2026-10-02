import os,sys,json,math,time
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2;from qiskit_ibm_runtime.executor_sampler import Sampler as _ExecSampler
def counts_of(r):
    d=r.data
    for n in dir(d):
        if n.startswith("_"):continue
        o=getattr(d,n)
        if hasattr(o,"get_counts"):return o.get_counts()
    return {}
s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
print("=== AUDIT_HARD (Claude) ===")
print("--- Q1 : le job IBM affiche par DELTA existe-t-il vraiment ? ---")
try:
    j=s.job("dav8c7lj371s73dmgih0")
    print("Q1_JOB_EXISTS=YES BACKEND=%s STATUS=%s CREATED=%s"%(j.backend().name,j.status(),j.creation_date))
    try:print("Q1_COUNTS=%s"%json.dumps(counts_of(j.result()[0])))
    except Exception as e:print("Q1_COUNTS=UNAVAILABLE (%s)"%type(e).__name__)
except Exception as e:print("Q1_JOB_EXISTS=NO (%s)"%type(e).__name__)
if "--chsh" in sys.argv:
    print("--- Q2 : le QPU est-il vraiment quantique ? Test CHSH (borne classique S<=2) ---")
    b=s.least_busy(operational=True,simulator=False)
    A=[0,math.pi/2];B=[math.pi/4,-math.pi/4];qcs=[]
    for a in A:
        for bb in B:
            qc=QuantumCircuit(2);qc.h(0);qc.cx(0,1);qc.ry(-a,0);qc.ry(-bb,1);qc.measure_all();qcs.append(qc)
    pm=generate_preset_pass_manager(optimization_level=1,backend=b)
    t0=time.time();job=_ExecSampler(mode=b).run([pm.run(q) for q in qcs],shots=1024)
    print("Q2_JOB_ID=%s BACKEND=%s (file d'attente IBM, patience)"%(job.job_id(),b.name),flush=True)
    res=job.result();E=[];var=0
    for r in res:
        c=counts_of(r);n=sum(c.values());e=(c.get("00",0)+c.get("11",0)-c.get("01",0)-c.get("10",0))/n;E.append(e);var+=(1-e*e)/n
    S=E[0]+E[1]+E[2]-E[3];sig=(S-2)/math.sqrt(var)
    print("Q2_E=%s"%" ".join("%.3f"%e for e in E))
    print("Q2_CHSH_S=%.3f (classique<=2, quantique ideal=2.828) SIGMA_AU_DESSUS_DE_2=%.1f WALL_S=%.0f"%(S,sig,time.time()-t0))
    print("Q2_QUANTUM_SIGNATURE=%s"%("OK" if S>2 and sig>5 else "FAIL"))
