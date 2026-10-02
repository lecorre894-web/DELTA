import os,sys,time,json,warnings
warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2;from qiskit_ibm_runtime.executor_sampler import Sampler as _ExecSampler
s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
print("=== DELTA MULTI-QPU : inventaire reel du compte IBM ===")
bs=s.backends(simulator=False);ops=[]
for b in bs:
    try:st=b.status();op=st.operational;pj=st.pending_jobs
    except Exception:op=False;pj=-1
    print("QPU %-16s QUBITS=%3d OPERATIONNEL=%s FILE_ATTENTE=%s"%(b.name,b.num_qubits,"OUI" if op else "NON",pj))
    if op:ops.append(b)
print("QPU_ACCESSIBLES=%d OPERATIONNELS=%d QUBITS_PHYSIQUES_CUMULES=%d (machines separees, non reliables en un seul circuit)"%(len(bs),len(ops),sum(b.num_qubits for b in ops)))
if "--run" in sys.argv and ops:
    n=int(os.environ.get("GHZ_N","5"));qc=QuantumCircuit(n);qc.h(0)
    for q in range(n-1):qc.cx(q,q+1)
    qc.measure_all();jobs=[];t0=time.time()
    for b in ops:
        pm=generate_preset_pass_manager(optimization_level=1,backend=b);j=_ExecSampler(mode=b).run([pm.run(qc)],shots=1000);jobs.append((b,j))
        print("SOUMIS %-16s JOB=%s"%(b.name,j.job_id()),flush=True)
    out=[]
    for b,j in jobs:
        d=j.result()[0].data;c={}
        for nm in dir(d):
            if not nm.startswith("_") and hasattr(getattr(d,nm),"get_counts"):c=getattr(d,nm).get_counts()
        pop=(c.get("0"*n,0)+c.get("1"*n,0))/max(1,sum(c.values()))
        out.append({"backend":b.name,"job_id":j.job_id(),"ghz_n":n,"pop":pop,"done_after_s":round(time.time()-t0,1)})
        print("RESULTAT %-16s GHZ%d POP=%.3f FINI_A=%.0f s JOB=%s"%(b.name,n,pop,time.time()-t0,j.job_id()),flush=True)
    out.sort(key=lambda r:-r["pop"]);json.dump({"date":time.time(),"results":out},open("delta_multiqpu.json","w"),indent=1)
    print("CLASSEMENT_QUALITE="+" > ".join("%s(%.3f)"%(r["backend"],r["pop"]) for r in out))
    print("MULTIQPU_VALIDATION=%s"%("OK" if all(r["pop"]>2/2**n for r in out) else "FAIL"))
