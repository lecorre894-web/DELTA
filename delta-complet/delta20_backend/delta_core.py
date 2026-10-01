import os,sys,json,time,shutil,numpy as np
from delta_mem import DeltaMem
from delta_qpu_cache import DeltaQPUCache,ghz,DB
HERE=os.path.dirname(os.path.abspath(__file__))
class DeltaCore:
    def __init__(s,tol=1e-3,budget_mb=None):
        s.mem=DeltaMem(budget_mb=budget_mb,tol=tol,cold_dir=os.path.join(HERE,"delta_mem_cold"));s.qpu=DeltaQPUCache();s.loaded=s._load()
    def _load(s):
        p=os.path.join(s.mem.dir,"index.json");n=0
        if os.path.exists(p):
            for k,m in json.load(open(p)).items():
                if m.get("tier")=="COLD_DISK" and os.path.exists(m.get("path","")):s.mem.meta[k]=m;n+=1
        return n
    def put(s,name,x,tol=None):return s.mem.put(name,x,tol)
    def get(s,name):return s.mem.get(name)
    def quantum(s,name,qc,shots=1000,physical=False,policy="quality",tol=None):
        r=s.qpu.run(qc,shots,physical,policy);n=qc.num_qubits
        if n<=20:
            p=np.zeros(2**n)
            for b,c in r["counts"].items():p[int(b,2)]=c/shots
            s.mem.put(name,p,tol if tol is not None else 1e-6)
        return r
    def checkpoint(s):
        idx={}
        for k,m in s.mem.meta.items():
            m=dict(m)
            if m["tier"]=="HOT_RAM":
                p=os.path.join(s.mem.dir,k+".bin");s.mem.hot[k].tofile(p);m["tier"]="COLD_DISK";m["path"]=p
            idx[k]=m
        json.dump(idx,open(os.path.join(s.mem.dir,"index.json"),"w"),indent=1);s.qpu.flush();return len(idx)
    def status(s):
        cold=sum(os.path.getsize(os.path.join(s.mem.dir,f)) for f in os.listdir(s.mem.dir))
        db=os.path.getsize(DB) if os.path.exists(DB) else 0;st=s.qpu.stats();rk=os.path.join(HERE,"delta_multiqpu.json")
        age=(time.time()-json.load(open(rk))["date"])/3600 if os.path.exists(rk) else -1
        return {"ram_hot_MB":round(s.mem.used/2**20,2),"ram_budget_MB":round(s.mem.budget/2**20),"objets_memoire":len(s.mem.meta),"charges_depuis_disque":s.loaded,
                "disque_froid_MB":round(cold/2**20,2),"cache_qpu_entrees":st["entries"],"cache_qpu_hits":st["hits_total"],"cache_qpu_MB":round(db/2**20,3),
                "classement_qpu_age_h":round(age,1),"disque_libre_GB":round(shutil.disk_usage(HERE).free/2**30,1)}
if __name__=="__main__":
    phys="--qpu" in sys.argv;D=DeltaCore();ok=True
    print("=== DELTA CORE : memoire typee + disque + cache QPU, soudes ===")
    print("REPRISE objets recharges depuis le disque = %d"%D.loaded)
    if "ghz8_probs" in D.mem.meta:
        p=D.get("ghz8_probs");print("RELU_DU_DISQUE ghz8_probs P(00000000)=%.3f P(11111111)=%.3f TYPE=%s"%(p[0],p[-1],D.mem.meta["ghz8_probs"]["kind"]))
    rng=np.random.default_rng(1);x=rng.standard_normal(2**20)
    m=D.put("donnees_capteur",x);e=float(np.sqrt(np.mean((D.get("donnees_capteur")-x)**2)/np.mean(x*x)));ok&=e<=1e-3
    print("PUT donnees_capteur TYPE=%s %d->%d Ko ERR=%.1e %s"%(m["kind"],m["raw_bytes"]>>10,m["bytes"]>>10,e,m["tier"]))
    r=D.quantum("ghz8_probs",ghz(8),1000,physical=phys);o=r["origin"]
    print("QUANTUM ghz8 NIVEAU=%s MOTEUR=%s QPU=%s JOB=%s ROUTAGE=%s"%(r["level"],o.get("engine"),o.get("backend","-"),o.get("job_id","-"),o.get("routing")))
    p=D.get("ghz8_probs");pop=p[0]+p[-1];print("MEMOIRE ghz8_probs TYPE=%s POP_GHZ=%.3f"%(D.mem.meta["ghz8_probs"]["kind"],pop));ok&=abs(p.sum()-1)<1e-3
    print("CHECKPOINT %d objets ecrits sur disque"%D.checkpoint())
    print("STATUS "+json.dumps(D.status(),ensure_ascii=False))
    print("DELTA_CORE_VALIDATION=%s"%("OK" if ok else "FAIL"))
