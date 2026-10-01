import os,sys,time,json,hashlib,sqlite3,warnings
warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit,qasm2
SIM_MAX=int(os.environ.get("DELTA_SIM_MAX","26"))
DB=os.environ.get("DELTA_CACHE_DB",os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_cache.db"))
class DeltaQPUCache:
    def __init__(s,db=DB):
        s.c=sqlite3.connect(db);s.c.execute("create table if not exists r(k text primary key,mode text,counts text,origin text,created real,hits integer default 0)");s.svc=None;s.mem={};s.pend={}
    def key(s,qc,shots,mode):return hashlib.sha256((qasm2.dumps(qc)+"|%d|%s"%(shots,mode)).encode()).hexdigest()
    def run(s,qc,shots=1024,physical=False):
        mode="qpu" if physical or qc.num_qubits>SIM_MAX else "sim";k=s.key(qc,shots,mode);t0=time.perf_counter()
        if k in s.mem:
            out=s.mem[k];s.pend[k]=s.pend.get(k,0)+1;return dict(out,level="L1_RAM",wall=time.perf_counter()-t0)
        row=s.c.execute("select mode,counts,origin,created from r where k=?",(k,)).fetchone()
        if row:
            out={"key":k[:16],"mode":row[0],"counts":json.loads(row[1]),"origin":json.loads(row[2]),"created":row[3]};s.mem[k]=out
            s.c.execute("update r set hits=hits+1 where k=?",(k,));s.c.commit();return dict(out,level="L2_DISK",wall=time.perf_counter()-t0)
        counts,origin=s._sim(qc,shots) if mode=="sim" else s._qpu(qc,shots)
        out={"key":k[:16],"mode":mode,"counts":counts,"origin":origin,"created":time.time()};s.mem[k]=out
        s.c.execute("insert into r(k,mode,counts,origin,created) values(?,?,?,?,?)",(k,mode,json.dumps(counts),json.dumps(origin),out["created"]));s.c.commit()
        return dict(out,level="MISS_COMPUTED",wall=time.perf_counter()-t0)
    def _sim(s,qc,shots):
        import numpy as np
        from qiskit import transpile
        n=qc.num_qubits;t=transpile(qc.remove_final_measurements(inplace=False),basis_gates=["u","cx"],optimization_level=0)
        psi=np.zeros([2]*n,dtype=np.complex128);psi[(0,)*n]=1.0
        for ins in t.data:
            q=[t.find_bit(b).index for b in ins.qubits];ax=[n-1-i for i in q]
            if ins.operation.name=="cx":
                sl=[slice(None)]*n;sl[ax[0]]=1;sub=psi[tuple(sl)];a=ax[1]-(1 if ax[1]>ax[0] else 0);sub[...]=np.flip(sub,axis=a).copy()
            elif ins.operation.name=="u":
                m=np.asarray(ins.operation.to_matrix());psi=np.moveaxis(np.tensordot(m,psi,axes=([1],[ax[0]])),0,ax[0])
        p=np.abs(psi.reshape(-1))**2;p/=p.sum();rng=np.random.default_rng(1234);idx=rng.choice(p.size,size=shots,p=p)
        u,cn=np.unique(idx,return_counts=True)
        return {format(int(i),"0%db"%n):int(c) for i,c in zip(u,cn)},{"engine":"DELTA_STATEVECTOR_EXACT","qubits":n}
    def _qpu(s,qc,shots):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2
        if s.svc is None:s.svc=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
        b=s.svc.least_busy(operational=True,simulator=False);job=SamplerV2(mode=b).run([generate_preset_pass_manager(optimization_level=1,backend=b).run(qc)],shots=shots)
        d=job.result()[0].data;c={}
        for nm in dir(d):
            if not nm.startswith("_") and hasattr(getattr(d,nm),"get_counts"):c=getattr(d,nm).get_counts()
        return {k:int(v) for k,v in c.items()},{"engine":"IBM_QPU","backend":b.name,"job_id":job.job_id(),"physical_qubits":b.num_qubits}
    def flush(s):
        for k,v in s.pend.items():s.c.execute("update r set hits=hits+? where k=?",(v,k))
        s.c.commit();s.pend={}
    def stats(s):
        s.flush()
        n,h=s.c.execute("select count(*),coalesce(sum(hits),0) from r").fetchone();return {"entries":n,"hits_total":h,"db":DB}
def ghz(n):
    qc=QuantumCircuit(n);qc.h(0)
    for q in range(n-1):qc.cx(q,q+1)
    qc.measure_all();return qc
def top(c,k=2):return dict(sorted(c.items(),key=lambda x:-x[1])[:k])
if __name__=="__main__":
    D=DeltaQPUCache();print("=== DELTA QPU CACHE — routeur + cache persistant + aiguillage sim/QPU ===")
    plan=[("GHZ 2 (Bell)",ghz(2),False),("GHZ 20",ghz(20),False)]
    if "--qpu" in sys.argv:plan.append(("GHZ 10 PHYSIQUE",ghz(10),True))
    for name,qc,phys in plan:
        for i in (1,2,3):
            r=D.run(qc,1000,phys)
            print("%-16s APPEL %d MODE=%s NIVEAU=%-13s WALL=%10.6f s TOP=%s ORIGINE=%s"%(name,i,r["mode"],r["level"],r["wall"],json.dumps(top(r["counts"])),json.dumps(r["origin"])),flush=True)
    print("CACHE_STATS=%s"%json.dumps(D.stats()))
