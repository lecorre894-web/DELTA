"""DELTA QPU -> HYBRIDE AUTO v2 : GHZ-10 vise le seuil F>0.5
1) meilleure chaine de N qubits d'apres les erreurs publiees (2q + lecture)  2) GHZ du milieu vers les bouts (profondeur ~N/2)
3) decouplage dynamique  4) twirling de mesure. Modes : --fake (puce simulee bruitee, gratuit) | --qpu (vrai IBM, quota)."""
import os,sys,json,time,hashlib,warnings,numpy as np
warnings.filterwarnings("ignore",category=DeprecationWarning)
from qiskit import QuantumCircuit
from delta_qpu_hybride import colonnes,parite
from delta_hybrid import zz
N=int(os.environ.get("GHZN","10"));SH=int(os.environ.get("SHOTS","4000"))
def ghz_milieu(n,base):
    q=QuantumCircuit(n);m=(n-1)//2;q.h(m)
    for k in range(1,n):
        if m+k<n:q.cx(m+k-1,m+k)
        if m-k>=0:q.cx(m-k+1,m-k)
    if base=="X":q.h(range(n))
    q.measure_all();return q
def meilleure_chaine(b,n):
    t=b.target;g=next(x for x in ("cz","ecr","cx") if x in t.operation_names)
    e2={};adj={}
    for (i,j),p in t[g].items():
        if p is None or p.error is None:continue
        e=min(p.error,e2.get((j,i),1));e2[(i,j)]=e2[(j,i)]=e;adj.setdefault(i,set()).add(j);adj.setdefault(j,set()).add(i)
    rd={q[0]:(p.error if p and p.error is not None else 0.05) for q,p in t["measure"].items()}
    best=[9e9,None]
    def dfs(path,cost):
        if cost>=best[0]:return
        if len(path)==n:best[0],best[1]=cost,list(path);return
        for v in adj.get(path[-1],()):
            if v not in path and e2[(path[-1],v)]<0.5:path.append(v);dfs(path,cost+e2[(path[-2],v)]+rd.get(v,0.05));path.pop()
    for s in sorted(adj,key=lambda q:rd.get(q,1)):dfs([s],rd.get(s,0.05))
    return best[1],best[0],g
def tirer(mode):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    if mode=="fake":
        from qiskit_ibm_runtime.fake_provider import FakeTorino;b=FakeTorino()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
        bk=os.environ.get("BACKEND");b=s.backend(bk) if bk else s.least_busy(operational=True,simulator=False)
    ch,cout,g=meilleure_chaine(b,N);print("PUCE %s | porte %s | chaine %s | cout erreurs %.4f"%(b.name,g,ch,cout),flush=True)
    pm=generate_preset_pass_manager(optimization_level=3,backend=b,initial_layout=ch)
    qs=[pm.run(ghz_milieu(N,"Z")),pm.run(ghz_milieu(N,"X"))];prof=[q.depth(lambda x:len(x.qubits)==2) for q in qs]
    sp=SamplerV2(mode=b);opts=[]
    if mode=="qpu":
        for nm,f in (("decouplage XY4",lambda:(setattr(sp.options.dynamical_decoupling,"enable",True),setattr(sp.options.dynamical_decoupling,"sequence_type","XY4"))),("twirling mesure",lambda:setattr(sp.options.twirling,"enable_measure",True))):
            try:f();opts.append(nm)
            except Exception as e:print("option %s indisponible : %s"%(nm,e))
    job=sp.run(qs,shots=SH);print("JOB %s sur %s | profondeur 2q %s | options %s | attente..."%(job.job_id(),b.name,prof,opts or "aucune (simulation)"),flush=True);r=job.result()
    return [x.data.meas.get_bitstrings() for x in r],{"backend":b.name,"job_id":job.job_id(),"chaine":ch,"cout_erreurs":cout,"profondeur_2q":prof,"options":opts}
if __name__=="__main__":
    mode="qpu" if "--qpu" in sys.argv else "fake";t0=time.time();(bz,bx),meta=tirer(mode);tq=time.time()-t0
    t=time.perf_counter();Mz=colonnes(bz,N);Mx=colonnes(bx,N);Z=zz(Mz);vo=[Z[i,i+1] for i in range(N-1)];pz=parite(Mz);px=parite(Mx);ta=time.perf_counter()-t
    pop=sum(1 for s in bz if s in("0"*N,"1"*N))/SH;F=(pop+px)/2;faible=int(np.argmin(vo))
    sig=hashlib.sha256(json.dumps([sorted(bz)[:50],sorted(bx)[:50],SH,N]).encode()).hexdigest()[:12]
    print("=== DELTA QPU -> HYBRIDE AUTO v2 : GHZ-%d, %d tirs, %s ==="%(N,SH,meta["backend"]))
    print("ZZ voisins : min %.3f (maillon %d-%d) | moy %.3f | max %.3f"%(min(vo),faible,faible+1,np.mean(vo),max(vo)))
    print("parite Z^%d %+.3f | parite X^%d %+.3f | population GHZ %.3f"%(N,pz,N,px,pop))
    print("fidelite >= %.3f -> %s"%(F,"INTRICATION %d QUBITS CERTIFIEE (F>0.5)"%N if F>0.5 else "NON CERTIFIEE"))
    print("temps : QPU %.1f s | analyse hybride auto %.2f ms | sig %s"%(tq,ta*1e3,sig))
    json.dump({"date":time.strftime("%Y-%m-%d %H:%M"),"mode":mode,**meta,"n":N,"shots":SH,"zz_voisins":[round(float(v),4) for v in vo],"parite_z":pz,"parite_x":px,"pop":pop,"fidelite_min":F,"sig":sig},open("delta_qpu_hybride2_%s.json"%mode,"w"),indent=1)
    print("DELTA_QPU_HYBRIDE2=%s"%("OK" if F>0.5 else "FAIL"))
