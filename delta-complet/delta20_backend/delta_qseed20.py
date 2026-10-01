import os,sys,json,math,hashlib,time
from qiskit import QuantumCircuit
from delta_qpu_cache import DeltaQPUCache
P=int(os.environ.get("QSEED_PAIRS","10"));NQ=2*P;SH=1024
def circuits():
    qs=[]
    for a in (0,math.pi/2):
        for b in (math.pi/4,-math.pi/4):
            q=QuantumCircuit(NQ)
            for j in range(P):q.h(2*j);q.cx(2*j,2*j+1);q.ry(-a,2*j);q.ry(-b,2*j+1)
            q.measure_all();qs.append(q)
    return qs
def ent(c):
    n=sum(c.values());p=[v/n for v in c.values() if v>0];k=len(p)
    return 0.5*((k-1)*math.log2(2*math.pi*math.e*n)+sum(math.log2(x) for x in p)) if k>1 else 0.0
def marg(c,j):
    m={}
    for k,v in c.items():
        b=k.replace(" ","");pb=b[-1-2*j]+b[-2-2*j];m[pb]=m.get(pb,0)+v
    return m
if __name__=="__main__":
    D=DeltaQPUCache();out,st=D.run_batch(circuits(),SH,physical=os.environ.get("QSEED_SIM","0")!="1")
    M=[[marg(x["counts"],j) for x in out] for j in range(P)];o=out[0]["origin"]
    print("=== DELTA QSEED-20 : %d paires de Bell en parallele = %d qubits, 1 job ==="%(P,NQ))
    good=[];bad=[];Ss=[]
    for j in range(P):
        e=[(m.get("00",0)+m.get("11",0)-m.get("01",0)-m.get("10",0))/SH for m in M[j]];S=e[0]+e[1]+e[2]-e[3]
        sig=(S-2)/math.sqrt(sum((1-x*x)/SH for x in e));ok=S>2 and sig>3;Ss.append(S);(good if ok else bad).append(j)
        print("PAIRE %2d qubits(%2d,%2d) S=%.3f %5.1f sigma %s"%(j,2*j,2*j+1,S,sig,"CERTIFIEE" if ok else "REJETEE"))
    h=hashlib.sha256();H=0.0
    for j in good:
        for m in M[j]:h.update(json.dumps(m,sort_keys=True).encode());H+=ent(m)
    h.update(str(o.get("job_id","sim")).encode());seed=h.hexdigest()
    Sg=sum(Ss[j] for j in good)/max(1,len(good))
    rec={"date":time.strftime("%Y-%m-%d %H:%M"),"qubits":NQ,"pairs":P,"pairs_certified":good,"pairs_rejected":bad,"S_certified_mean":round(Sg,3),"S_pairs":[round(s,3) for s in Ss],"entropie_certifiee_bits":round(H),"seed_sha256":seed,"seed64":int(seed[:16],16),"backend":o.get("backend","local"),"job_id":o.get("job_id","-"),"niveau":out[0]["level"],"jobs":st["jobs"]}
    json.dump(rec,open("delta_qseed20.json","w"),indent=1)
    print("BILAN certifiees=%s rejetees=%s S_moyen_certifiees=%.3f QPU=%s JOB=%s JOBS_PAYES=%d NIVEAU=%s"%(good,bad,Sg,rec["backend"],rec["job_id"],st["jobs"],rec["niveau"]))
    print("GRAINE=%s... ENTROPIE_CERTIFIEE~%d bits (paires certifiees seulement)"%(seed[:24],H))
    print("QSEED20_VALIDATION=%s"%("OK" if len(good)>=0.8*P else "FAIL"))
