import os,sys,json,math,time,hashlib,warnings,numpy as np
warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit
SH=int(os.environ.get("POOL_SHOTS","4096"));SIM=os.environ.get("POOL_SIM","0")=="1";HERE=os.path.dirname(os.path.abspath(__file__))
BIN=os.path.join(HERE,"delta_qpool.bin");META=os.path.join(HERE,"delta_qpool.json")
def build(pairs):
    P=len(pairs);qs=[]
    for a in (0,math.pi/2):
        for b in (math.pi/4,-math.pi/4):
            q=QuantumCircuit(2*P)
            for j in range(P):q.h(2*j);q.cx(2*j,2*j+1);q.ry(-a,2*j);q.ry(-b,2*j+1)
            q.measure_all();qs.append(q)
    return qs
def fill():
    best=json.load(open(os.path.join(HERE,"delta_qpairs_best.json")));pairs=best["pairs"];P=len(pairs);qs=build(pairs)
    if SIM:
        from delta_qpu_cache import DeltaQPUCache;D=DeltaQPUCache(":memory:");rng=np.random.default_rng(1);BS=[]
        for q in qs:
            c=D._sim(q,SH)[0];l=[k for k,v in c.items() for _ in range(v)];rng.shuffle(l);BS.append(l)
        bk,jid="local","sim"
    else:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2;from qiskit_ibm_runtime.executor_sampler import Sampler as _ExecSampler
        s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
        b=s.backend(best["backend"]);lay=[x for p in pairs for x in p]
        pm=generate_preset_pass_manager(optimization_level=1,backend=b,initial_layout=lay);job=_ExecSampler(mode=b).run([pm.run(q) for q in qs],shots=SH)
        print("SOUMIS %s JOB=%s"%(b.name,job.job_id()),flush=True);BS=[]
        for r in job.result():
            d=r.data
            for n in dir(d):
                if not n.startswith("_") and hasattr(getattr(d,n),"get_bitstrings"):BS.append(getattr(d,n).get_bitstrings())
        bk,jid=b.name,job.job_id()
    BS=[[x.replace(" ","") for x in l] for l in BS];ok=[]
    for j in range(P):
        E=[sum(1 if x[-1-2*j]==x[-2-2*j] else -1 for x in l)/len(l) for l in BS]
        S=E[0]+E[1]+E[2]-E[3];sig=(S-2)/math.sqrt(sum((1-e*e)/SH for e in E));ok.append(S>2 and sig>3)
    raw=np.array([int(x[-1-2*j]) for l in BS for x in l for j in range(P) if ok[j]],dtype=np.uint8)
    rb=np.packbits(raw).tobytes();out=b"".join(hashlib.sha256(rb[i:i+64]).digest() for i in range(0,len(rb)-63,64))
    open(BIN,"wb").write(out)
    meta={"date":time.strftime("%Y-%m-%d %H:%M"),"backend":bk,"job_id":jid,"shots":SH,"pairs":pairs,"pairs_certified":[j for j in range(P) if ok[j]],"raw_bits":int(raw.size),"bits_out":len(out)*8,"cursor_bits":0,"extracteur":"SHA-256 512->256 bits (compression x2)"}
    json.dump(meta,open(META,"w"),indent=1);return meta
def draw(nbits,consumer="?"):
    m=json.load(open(META))
    if m["cursor_bits"]+nbits>m["bits_out"]:raise RuntimeError("RESERVOIR EPUISE (%d bits restants) : relancer --fill, un bit n'est jamais reutilise"%(m["bits_out"]-m["cursor_bits"]))
    d=np.unpackbits(np.frombuffer(open(BIN,"rb").read(),dtype=np.uint8))[m["cursor_bits"]:m["cursor_bits"]+nbits]
    m["cursor_bits"]+=nbits;m.setdefault("journal",[]).append([consumer,nbits,time.strftime("%H:%M:%S")]);json.dump(m,open(META,"w"),indent=1)
    return d,{"job_id":m["job_id"],"backend":m["backend"]}
if __name__=="__main__":
    print("=== DELTA QPOOL : reservoir de bits quantiques ===")
    if "--fill" in sys.argv or not os.path.exists(META):m=fill();print("REMPLI QPU=%s JOB=%s paires certifiees=%s BITS_BRUTS=%d BITS_EXTRAITS=%d"%(m["backend"],m["job_id"],m["pairs_certified"],m["raw_bits"],m["bits_out"]))
    for who,n in (("graine_domaine_1T",256),("synapses_init",1024),("tirage_test",64)):
        b,o=draw(n,who);print("DISTRIBUE %-18s %5d bits  proportion_de_1=%.3f  origine=%s"%(who,n,b.mean(),o["job_id"]))
    m=json.load(open(META));print("RESERVOIR %d/%d bits consommes, %d restants (jamais reutilises)"%(m["cursor_bits"],m["bits_out"],m["bits_out"]-m["cursor_bits"]))
    print("QPOOL_VALIDATION=%s"%("OK" if m["pairs_certified"] and m["bits_out"]>0 else "FAIL"))
