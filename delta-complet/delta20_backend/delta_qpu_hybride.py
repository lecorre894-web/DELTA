"""DELTA QPU -> HYBRIDE AUTO : le QPU produit les tirs (GHZ-10 en base Z et X), le moteur hybride auto les analyse.
Local (simulateur exact) par defaut ; --qpu pour le vrai IBM (consomme du quota, lance a la main)."""
import os,sys,json,time,hashlib,numpy as np
from qiskit import QuantumCircuit
from delta_hybrid import compute_bool,zz
N=int(os.environ.get("GHZN","10"));SH=int(os.environ.get("SHOTS","4000"))
def ghz(n,base):
    q=QuantumCircuit(n);q.h(0)
    for i in range(n-1):q.cx(i,i+1)
    if base=="X":q.h(range(n))
    q.measure_all();return q
def colonnes(bs,n):
    """tirs -> n vecteurs de bits (un par qubit), ordre qiskit : dernier caractere = qubit 0"""
    M=np.array([[c=="1" for c in s[::-1]] for s in bs],dtype=np.uint8).T[:n]
    M=np.ascontiguousarray(M)
    return M
def pack(M):
    P=np.packbits(M,axis=1,bitorder="little");P=np.pad(P,((0,0),(0,(-P.shape[1])%8)));return np.ascontiguousarray(P.view(np.uint64))
def parite(M):
    """<Z1..Zn> : XOR des n vecteurs puis popcount par le moteur hybride auto"""
    W=pack(M);X=np.bitwise_xor.reduce(W,axis=0);ones=np.full_like(X,np.uint64(0xFFFFFFFFFFFFFFFF))
    impair=int(compute_bool(X[None,:],ones[None,:],np.zeros_like(X))[0,0]);return 1-2*impair/M.shape[1]
def tirer(qpu):
    qs=[ghz(N,"Z"),ghz(N,"X")]
    if not qpu:
        from qiskit.primitives import StatevectorSampler
        r=StatevectorSampler(seed=7).run(qs,shots=SH).result();return [x.data.meas.get_bitstrings() for x in r],{"backend":"simulateur_exact","job_id":"local"}
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2
    s=QiskitRuntimeService(channel="ibm_quantum_platform",token=os.environ["IQP_API_TOKEN"],instance=os.environ["IQP_INSTANCE_CRN"])
    b=s.least_busy(operational=True,simulator=False);pm=generate_preset_pass_manager(optimization_level=1,backend=b)
    job=SamplerV2(mode=b).run([pm.run(q) for q in qs],shots=SH);print("JOB %s sur %s, attente..."%(job.job_id(),b.name),flush=True);r=job.result()
    return [x.data.meas.get_bitstrings() for x in r],{"backend":b.name,"job_id":job.job_id()}
if __name__=="__main__":
    qpu="--qpu" in sys.argv;t0=time.time();(bz,bx),meta=tirer(qpu);tq=time.time()-t0
    t=time.perf_counter();Mz=colonnes(bz,N);Mx=colonnes(bx,N);tc=time.perf_counter()-t
    t=time.perf_counter()
    Z=zz(Mz);voisins=[Z[i,i+1] for i in range(N-1)]
    pz=parite(Mz);px=parite(Mx)
    ta=time.perf_counter()-t
    tous0=sum(1 for s in bz if s=="0"*N);tous1=sum(1 for s in bz if s=="1"*N);pop=(tous0+tous1)/SH
    F=(pop+px)/2  # borne inferieure de fidelite GHZ (temoin standard) ; >0.5 = intrication multipartite authentique
    sig=hashlib.sha256(json.dumps([sorted(bz)[:50],sorted(bx)[:50],SH,N]).encode()).hexdigest()[:12]
    print("=== DELTA QPU -> HYBRIDE AUTO : GHZ-%d, %d tirs, %s ==="%(N,SH,meta["backend"]))
    print("ZZ voisins : min %.3f | moy %.3f | max %.3f  (ideal +1)"%(min(voisins),np.mean(voisins),max(voisins)))
    print("parite Z^%d : %+.3f (ideal +1 si N pair) | parite X^%d : %+.3f (ideal +1)"%(N,pz,N,px))
    print("population GHZ (tous 0 + tous 1) : %.3f | fidelite >= %.3f -> %s"%(pop,F,"INTRICATION %d QUBITS CERTIFIEE (F>0.5)"%N if F>0.5 else "NON CERTIFIEE"))
    print("temps : QPU/simulation %.1f s | colonnes %.1f ms | analyse hybride auto %.2f ms | sig %s"%(tq,tc*1e3,ta*1e3,sig))
    json.dump({"date":time.strftime("%Y-%m-%d %H:%M"),**meta,"n":N,"shots":SH,"zz_voisins":[round(float(v),4) for v in voisins],"parite_z":pz,"parite_x":px,"pop":pop,"fidelite_min":F,"sig":sig},open("delta_qpu_hybride.json","w"),indent=1)
    print("DELTA_QPU_HYBRIDE=%s"%("OK" if F>0.5 else "FAIL"))
