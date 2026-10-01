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
if __name__=="__main__":
    D=DeltaQPUCache();out,st=D.run_batch(circuits(),SH,physical=os.environ.get("QSEED_SIM","0")!="1")
    E=[[0.0]*4 for _ in range(P)];H=0.0;h=hashlib.sha256()
    for ci,x in enumerate(out):
        c=x["counts"];n=sum(c.values());h.update(json.dumps(c,sort_keys=True).encode())
        for j in range(P):
            m={}
            for k,v in c.items():
                b=k.replace(" ","");pb=b[-1-2*j]+b[-2-2*j];m[pb]=m.get(pb,0)+v
            E[j][ci]=(m.get("00",0)+m.get("11",0)-m.get("01",0)-m.get("10",0))/n;H+=ent(m)
    o=out[0]["origin"];h.update(str(o.get("job_id","sim")).encode());seed=h.hexdigest()
    print("=== DELTA QSEED-20 : %d paires de Bell en parallele = %d qubits, 1 job ==="%(P,NQ))
    Ss=[];ok_pairs=0
    for j in range(P):
        e=E[j];S=e[0]+e[1]+e[2]-e[3];sig=(S-2)/math.sqrt(sum((1-x*x)/SH for x in e));Ss.append(S);ok_pairs+=S>2 and sig>3
        print("PAIRE %2d qubits(%2d,%2d) S=%.3f %5.1f sigma %s"%(j,2*j,2*j+1,S,sig,"CERTIFIEE" if S>2 and sig>3 else "NON"))
    Sm=sum(Ss)/P;sigm=(Sm-2)/math.sqrt(sum(sum((1-x*x)/SH for x in E[j]) for j in range(P))/P**2)
    rec={"date":time.strftime("%Y-%m-%d %H:%M"),"qubits":NQ,"pairs":P,"pairs_certified":ok_pairs,"S_mean":round(Sm,3),"sigma_mean":round(sigm,1),"S_pairs":[round(s,3) for s in Ss],"entropie_min_bits":round(H),"seed_sha256":seed,"seed64":int(seed[:16],16),"backend":o.get("backend","local"),"job_id":o.get("job_id","-"),"niveau":out[0]["level"],"jobs":st["jobs"]}
    json.dump(rec,open("delta_qseed20.json","w"),indent=1)
    print("BILAN paires_certifiees=%d/%d S_moyen=%.3f (%.1f sigma) QPU=%s JOB=%s JOBS_PAYES=%d"%(ok_pairs,P,Sm,sigm,rec["backend"],rec["job_id"],st["jobs"]))
    print("GRAINE=%s... ENTROPIE~%d bits (borne basse, marges par paire) vs ~65 bits avec 2 qubits"%(seed[:24],H))
    print("NOTE: 10 paires independantes, PAS un etat intrique a 20 qubits (GHZ20 mesure 0.311 : trop bruite)")
    print("QSEED20_VALIDATION=%s"%("OK" if ok_pairs>=0.8*P and sigm>5 else "FAIL"))
