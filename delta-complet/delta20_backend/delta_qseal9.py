import os,sys,json,math,hashlib,time,warnings,numpy as np
warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit
from delta_core import DeltaCore
SH=int(os.environ.get("SEAL_SHOTS","2048"));PHYS=[int(x) for x in os.environ.get("SEAL_QUBITS","152,153").split(",")];SIM=os.environ.get("SEAL_SIM","0")=="1"
def chsh():
    qs=[]
    for a in (0,math.pi/2):
        for b in (math.pi/4,-math.pi/4):
            q=QuantumCircuit(2);q.h(0);q.cx(0,1);q.ry(-a,0);q.ry(-b,1);q.measure_all();qs.append(q)
    return qs
def ent(c):
    n=sum(c.values());p=[v/n for v in c.values() if v>0];k=len(p)
    return 0.5*((k-1)*math.log2(2*math.pi*math.e*n)+sum(math.log2(x) for x in p)) if k>1 else 0.0
print("=== DELTA QSEAL-9 : sceau sur la meilleure paire physique %s + memoire typee + disque ==="%PHYS)
if SIM:
    from delta_qpu_cache import DeltaQPUCache;S0=DeltaQPUCache(":memory:");C=[S0._sim(q,SH)[0] for q in chsh()];bk="local";jid="sim";used=PHYS
else:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2;from qiskit_ibm_runtime.executor_sampler import Sampler as _ExecSampler
    s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
    b=s.backend(os.environ.get("SEAL_BACKEND","ibm_marrakesh"));edges=set(map(tuple,b.coupling_map.get_edges()))
    if tuple(PHYS) not in edges and tuple(PHYS[::-1]) not in edges:sys.exit("ARRET: qubits %s non relies directement sur %s"%(PHYS,b.name))
    pm=generate_preset_pass_manager(optimization_level=1,backend=b,initial_layout=PHYS);tc=[pm.run(q) for q in chsh()]
    used=list(tc[0].layout.final_index_layout())
    if sorted(used)!=sorted(PHYS):sys.exit("ARRET: le compilateur a deplace la paire vers %s"%used)
    job=_ExecSampler(mode=b).run(tc,shots=SH);print("SOUMIS %s JOB=%s qubits physiques imposes=%s"%(b.name,job.job_id(),used),flush=True)
    C=[]
    for r in job.result():
        d=r.data
        for n in dir(d):
            if not n.startswith("_") and hasattr(getattr(d,n),"get_counts"):C.append(getattr(d,n).get_counts())
    bk=b.name;jid=job.job_id()
E=[(c.get("00",0)+c.get("11",0)-c.get("01",0)-c.get("10",0))/sum(c.values()) for c in C];S=E[0]+E[1]+E[2]-E[3]
sig=(S-2)/math.sqrt(sum((1-e*e)/SH for e in E));H=sum(ent(c) for c in C);h=hashlib.sha256()
for c in C:h.update(json.dumps(c,sort_keys=True).encode())
h.update(jid.encode());seed=h.hexdigest();ok=S>2 and sig>5
P=np.array([[c.get(k,0)/sum(c.values()) for k in ("00","01","10","11")] for c in C]).ravel()
D=DeltaCore();m=D.put("qseal9_probs",P,tol=1e-6);n=D.checkpoint()
rec={"date":time.strftime("%Y-%m-%d %H:%M"),"backend":bk,"job_id":jid,"qubits_physiques":used,"shots":SH,"chsh_S":round(S,3),"sigma":round(sig,1),"entropie_bits":round(H),"seed_sha256":seed,"seed64":int(seed[:16],16),"memoire_type":m["kind"],"certifie":ok}
json.dump(rec,open("delta_qseal9.json","w"),indent=1);json.dump(rec,open(os.path.join(D.mem.dir,"qseal9.json"),"w"),indent=1)
D2=DeltaCore();back=D2.get("qseal9_probs");same=bool(np.max(np.abs(back-P))<1e-6)
print("CHSH S=%.3f (%.1f sigma) QPU=%s JOB=%s QUBITS=%s SHOTS=%d"%(S,sig,bk,jid,used,SH))
print("GRAINE=%s... ENTROPIE~%d bits"%(seed[:24],H))
print("MEMOIRE type=%s %d->%d octets | DISQUE checkpoint %d objets | RELU_APRES_REDEMARRAGE=%s"%(m["kind"],m["raw_bytes"],m["bytes"],n,"IDENTIQUE" if same else "DIFFERENT"))
print("RESERVE: la meilleure paire d'aujourd'hui peut changer apres recalibration ; refaire le classement 10 paires de temps en temps")
print("QSEAL9_VALIDATION=%s"%("OK" if ok and same else "FAIL"))
