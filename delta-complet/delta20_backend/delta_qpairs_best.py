import os,sys,json,math,time,warnings
warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2;from qiskit_ibm_runtime.executor_sampler import Sampler as _ExecSampler
P=int(os.environ.get("QP_PAIRS","10"));SH=1024
s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
b=s.backend(os.environ.get("QP_BACKEND","ibm_marrakesh"));T=b.target
g2=next(o for o in ("cz","ecr","cx") if o in T.operation_names)
def err(op,qs):
    try:return T[op][tuple(qs)].error or 0.0
    except Exception:return 1.0
E=[]
for (a,c),pr in T[g2].items():
    if pr is None or pr.error is None:continue
    E.append((pr.error+err("measure",[a])+err("measure",[c]),a,c))
E.sort();used=set();pairs=[]
for sc,a,c in E:
    if a in used or c in used:continue
    pairs.append((a,c,sc));used|={a,c}
    if len(pairs)==P:break
print("=== DELTA PAIRES GUERIES : %d meilleurs coupleurs directs de %s (porte %s, erreur porte+lecture) ==="%(P,b.name,g2))
for i,(a,c,sc) in enumerate(pairs):print("CHOIX paire %2d -> qubits physiques (%3d,%3d) erreur_estimee=%.4f"%(i,a,c,sc))
qs=[]
for x in (0,math.pi/2):
    for y in (math.pi/4,-math.pi/4):
        q=QuantumCircuit(2*P)
        for j in range(P):q.h(2*j);q.cx(2*j,2*j+1);q.ry(-x,2*j);q.ry(-y,2*j+1)
        q.measure_all();qs.append(q)
lay=[q for a,c,_ in pairs for q in (a,c)]
pm=generate_preset_pass_manager(optimization_level=1,backend=b,initial_layout=lay);tc=[pm.run(q) for q in qs]
fl=list(tc[0].layout.final_index_layout())
if fl!=lay:print("ATTENTION: le compilateur a modifie le placement -> %s"%fl)
job=_ExecSampler(mode=b).run(tc,shots=SH);print("SOUMIS JOB=%s"%job.job_id(),flush=True)
C=[]
for r in job.result():
    d=r.data
    for n in dir(d):
        if not n.startswith("_") and hasattr(getattr(d,n),"get_counts"):C.append(getattr(d,n).get_counts())
ok=0;Ss=[]
for j in range(P):
    e=[]
    for c in C:
        a=0
        for k,v in c.items():
            bb=k.replace(" ","");a+=v if bb[-1-2*j]==bb[-2-2*j] else -v
        e.append(a/SH)
    S=e[0]+e[1]+e[2]-e[3];sig=(S-2)/math.sqrt(sum((1-x*x)/SH for x in e));Ss.append(S);ok+=S>2 and sig>3
    print("PAIRE %2d physiques(%3d,%3d) S=%.3f %5.1f sigma %s"%(j,pairs[j][0],pairs[j][1],S,sig,"CERTIFIEE" if S>2 and sig>3 else "REJETEE"))
rec={"date":time.strftime("%Y-%m-%d %H:%M"),"backend":b.name,"job_id":job.job_id(),"pairs":[[a,c] for a,c,_ in pairs],"S":[round(x,3) for x in Ss],"certified":ok}
json.dump(rec,open("delta_qpairs_best.json","w"),indent=1)
print("BILAN %d/%d paires certifiees (avant guerison : 8/10) S_moyen=%.3f JOB=%s"%(ok,P,sum(Ss)/P,job.job_id()))
print("QPAIRS_VALIDATION=%s"%("OK" if ok==P else "PARTIEL"))
