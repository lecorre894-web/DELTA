import os,json,math,warnings
warnings.filterwarnings("ignore")
from qiskit_ibm_runtime import QiskitRuntimeService
R=json.load(open("delta_qseed20.json"));P=R["pairs"];SH=1024
s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
j=s.job(R["job_id"]);res=j.result()
print("=== TEST JALOUSIE : resultat brut relu chez IBM, recalcule sans aucun code DELTA ===")
print("JOB=%s BACKEND=%s STATUT=%s"%(j.job_id(),j.backend().name,j.status()))
def counts(r):
    d=r.data
    for n in dir(d):
        if not n.startswith("_") and hasattr(getattr(d,n),"get_counts"):return getattr(d,n).get_counts()
C=[counts(r) for r in res];same=0
try:lay=[q for q in j.inputs["pubs"][0][0].layout.final_index_layout()]
except Exception:lay=None
for p in range(P):
    e=[]
    for c in C:
        a=0
        for k,v in c.items():
            b=k.replace(" ","");a+=v if b[-1-2*p]==b[-2-2*p] else -v
        e.append(a/SH)
    S=e[0]+e[1]+e[2]-e[3];d=abs(S-R["S_pairs"][p]);same+=d<6e-4
    phys="physiques %s"%(lay[2*p:2*p+2],) if lay else "physiques inconnus"
    print("PAIRE %2d IBM_brut S=%.3f DELTA S=%.3f ECART=%.1e %s"%(p,S,R["S_pairs"][p],d,phys))
print("VERDICT : %s"%("le Xeon n'a rien altere (%d/%d paires identiques a l arrondi JSON pres (3 decimales)) ; les paires rejetees viennent de la puce QPU"%(same,P) if same==P else "ECART detecte sur %d paires : a examiner"%(P-same)))
