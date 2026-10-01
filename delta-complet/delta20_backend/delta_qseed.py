import os,sys,json,math,hashlib,time,numpy as np
from qiskit import QuantumCircuit
from delta_qpu_cache import DeltaQPUCache
def chsh():
    qs=[]
    for a in (0,math.pi/2):
        for b in (math.pi/4,-math.pi/4):
            q=QuantumCircuit(2);q.h(0);q.cx(0,1);q.ry(-a,0);q.ry(-b,1);q.measure_all();qs.append(q)
    return qs
def entropy_bits(c):
    n=sum(c.values());p=[v/n for v in c.values() if v>0];k=len(p)
    return 0.5*((k-1)*math.log2(2*math.pi*math.e*n)+sum(math.log2(x) for x in p)) if k>1 else 0.0
if __name__=="__main__":
    D=DeltaQPUCache();phys=os.environ.get("QSEED_SIM","0")!="1"
    out,st=D.run_batch(chsh(),1024,physical=phys)
    E=[];H=0.0;h=hashlib.sha256()
    for x in out:
        c=x["counts"];n=sum(c.values());E.append((c.get("00",0)+c.get("11",0)-c.get("01",0)-c.get("10",0))/n);H+=entropy_bits(c)
        h.update(json.dumps(c,sort_keys=True).encode())
    o=out[0]["origin"];h.update(str(o.get("job_id","sim")).encode());S=E[0]+E[1]+E[2]-E[3]
    var=sum((1-e*e)/1024 for e in E);sig=(S-2)/math.sqrt(var);seed=h.hexdigest()
    rec={"date":time.strftime("%Y-%m-%d %H:%M"),"seed_sha256":seed,"seed64":int(seed[:16],16),"chsh_S":round(S,3),"sigma":round(sig,1),"entropie_estimee_bits":round(H,1),"backend":o.get("backend","local"),"job_id":o.get("job_id","-"),"niveau":out[0]["level"]}
    json.dump(rec,open("delta_qseed.json","w"),indent=1)
    print("=== DELTA QSEED : sceau quantique 2 qubits (paire de Bell) ===")
    print("CHSH S=%.3f (%.1f sigma au-dessus de 2) QPU=%s JOB=%s NIVEAU=%s JOBS_PAYES=%d"%(S,sig,rec["backend"],rec["job_id"],rec["niveau"],st["jobs"]))
    print("GRAINE=%s... ENTROPIE_REELLE~%.0f bits (fluctuations des comptes ; SHA-256 ne cree pas d'entropie)"%(seed[:24],H))
    s64=np.uint64(rec["seed64"]);a=np.array([0,1,999999999999],dtype=np.uint64)
    m=lambda x:((x^s64)*np.uint64(0xBF58476D1CE4E5B9))%np.uint64(32)
    np.seterr(over="ignore");print("DOMAINE_1T brasse : adresses %s -> canaux DSPC %s (au lieu de %s sans graine)"%(list(map(int,a)),list(map(int,m(a))),list(map(int,a%np.uint64(32)))))
    print("RESERVE: via le cloud on fait confiance a IBM -> alea certifie par la physique, pas independant de l'appareil")
    print("QSEED_VALIDATION=%s"%("OK" if S>2 and sig>5 else "FAIL (pas de violation de Bell : graine non certifiee)"))
