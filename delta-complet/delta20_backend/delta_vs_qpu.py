import os,sys,time,math,random,json
import numpy as np
def ghz_sv(n):
    psi=np.zeros(2**n,dtype=np.complex128);psi[0]=1.0;t=psi.reshape([2]*n)
    a=t[0].copy();b=t[1].copy();t[0]=(a+b)/math.sqrt(2);t[1]=(a-b)/math.sqrt(2)
    for q in range(n-1):
        sl=[slice(None)]*n;sl[q]=1;sub=t[tuple(sl)];sub[...]=np.flip(sub,axis=q).copy()
    return psi
print("=== DELTA vs QPU IBM : GHZ N QUBITS (intrication totale) ===")
print("--- A : DELTA simule classiquement (vecteur d'etat exact, 1 coeur) ---")
pts=[];maxn=int(os.environ.get("SV_MAX","26"))
for n in range(16,maxn+1,2):
    try:
        t0=time.time();p=ghz_sv(n);w=time.time()-t0
        ok=abs(abs(p[0])**2-0.5)<1e-9 and abs(abs(p[-1])**2-0.5)<1e-9
        print("SV N=%2d MEM=%8.1f MB WALL=%7.3f s CHECK=%s"%(n,16*2**n/2**20,w,"OK" if ok else "FAIL"),flush=True);pts.append((n,w));del p
    except MemoryError:print("SV N=%2d MEMORY_WALL"%n);break
k=sum(math.log(w/(n*2**n)) for n,w in pts[-3:])/len(pts[-3:]);a=math.exp(k)
for n in (30,40,50,100,156):
    lt=math.log10(a*n)+n*math.log10(2);lm=math.log10(16)+n*math.log10(2)
    print("PROJECTION SV N=%3d TEMPS=10^%.1f s MEM=10^%.1f octets [NON EXECUTEE]"%(n,lt,lm))
print("PROJECTION SV N=1T TEMPS=10^%.3g s MEM=10^%.3g octets [IMPOSSIBLE]"%(1e12*math.log10(2),1e12*math.log10(2)+1.2))
if "--qpu" in sys.argv:
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2;from qiskit_ibm_runtime.executor_sampler import Sampler as _ExecSampler
    print("--- B : QPU IBM execute physiquement les memes GHZ ---")
    s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
    b=s.least_busy(operational=True,simulator=False);NS=[5,10,20,50,100];qcs=[]
    for n in NS:
        qc=QuantumCircuit(n);qc.h(0)
        for q in range(n-1):qc.cx(q,q+1)
        qc.measure_all();qcs.append(qc)
    pm=generate_preset_pass_manager(optimization_level=1,backend=b)
    t0=time.time();job=_ExecSampler(mode=b).run([pm.run(q) for q in qcs],shots=1000)
    print("QPU_JOB_ID=%s BACKEND=%s QUBITS_PHYSIQUES=%d"%(job.job_id(),b.name,b.num_qubits),flush=True)
    res=job.result();wall=time.time()-t0
    try:qs=job.metrics()["usage"]["quantum_seconds"]
    except Exception:qs=-1
    store={}
    for n,r in zip(NS,res):
        d=r.data;c={}
        for nm in dir(d):
            if not nm.startswith("_") and hasattr(getattr(d,nm),"get_counts"):c=getattr(d,nm).get_counts()
        tot=sum(c.values());pop=(c.get("0"*n,0)+c.get("1"*n,0))/tot
        print("QPU N=%3d SHOTS=%d POP_GHZ(0..0+1..1)=%.3f (ideal 1.000, aleatoire %.1e)"%(n,tot,pop,2/2**n))
        store[n]=[k for k,v in c.items() for _ in range(v)]
    print("QPU_WALL_TOTAL=%.0f s (file+execution) QPU_QUANTUM_SECONDS=%s"%(wall,qs))
    print("--- C : DELTA instruit le cache avec les echantillons QPU ---")
    keys=[(n,random.randrange(len(store[n]))) for n in NS for _ in range(200000)];t0=time.time();acc=0
    for n,i in keys:acc^=len(store[n][i])
    w=time.time()-t0
    print("CACHE_REPLAY LOOKUPS=%d NS_PER_SAMPLE=%.0f SAMPLES_PER_S=%.2f M (relecture, aucune info quantique nouvelle)"%(len(keys),w/len(keys)*1e9,len(keys)/w/1e6))
