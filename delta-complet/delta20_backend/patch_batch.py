f="delta_qpu_cache.py";s=open(f).read()
a="    def flush(s):"
b='''    def _qpu_batch(s,qcs,shots,bk=None):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2
        if s.svc is None:s.svc=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
        b=s.svc.backend(bk) if bk else s.svc.least_busy(operational=True,simulator=False);pm=generate_preset_pass_manager(optimization_level=1,backend=b)
        job=SamplerV2(mode=b).run([pm.run(q) for q in qcs],shots=shots);res=job.result();outs=[]
        for r in res:
            d=r.data;c={}
            for nm in dir(d):
                if not nm.startswith("_") and hasattr(getattr(d,nm),"get_counts"):c=getattr(d,nm).get_counts()
            outs.append({k:int(v) for k,v in c.items()})
        return outs,{"engine":"IBM_QPU","backend":b.name,"job_id":job.job_id(),"physical_qubits":b.num_qubits}
    def run_batch(s,qcs,shots=1024,physical=True,policy="quality",ttl_h=24):
        mode="qpu" if physical else "sim";bk,how=s.choose(policy) if physical else (None,"LOCAL");t0=time.perf_counter()
        out=[None]*len(qcs);miss=[];hits=0
        for i,qc in enumerate(qcs):
            k=s.key(qc,shots,mode,bk or "auto")
            if k in s.mem:out[i]=dict(s.mem[k],level="L1_RAM");hits+=1;continue
            row=s.c.execute("select mode,counts,origin,created from r where k=?",(k,)).fetchone()
            if row and not(mode=="qpu" and ttl_h and time.time()-row[3]>ttl_h*3600):
                o={"key":k[:16],"mode":row[0],"counts":json.loads(row[1]),"origin":json.loads(row[2]),"created":row[3]};s.mem[k]=o;out[i]=dict(o,level="L2_DISK");hits+=1;continue
            miss.append((i,k,qc))
        jid=None
        if miss:
            if mode=="sim":res=[s._sim(q,shots) for _,_,q in miss]
            else:
                cs,org=s._qpu_batch([q for _,_,q in miss],shots,bk);jid=org["job_id"];res=[(c,dict(org,batch_size=len(miss))) for c in cs]
            for (i,k,qc),(c,o) in zip(miss,res):
                o=dict(o,routing=how);rec={"key":k[:16],"mode":mode,"counts":c,"origin":o,"created":time.time()};s.mem[k]=rec
                s.c.execute("insert or replace into r(k,mode,counts,origin,created) values(?,?,?,?,?)",(k,mode,json.dumps(c),json.dumps(o),rec["created"]));out[i]=dict(rec,level="MISS_COMPUTED")
            s.c.commit()
        return out,{"circuits":len(qcs),"hits":hits,"computed":len(miss),"jobs":1 if (miss and mode=="qpu") else 0,"job_id":jid,"backend":bk,"wall":time.perf_counter()-t0}
    def flush(s):'''
assert s.count(a)==1 and "def run_batch" not in s,"ANCRE";s=s.replace(a,b);open(f,"w").write(s);print("PATCH BATCH OK")
