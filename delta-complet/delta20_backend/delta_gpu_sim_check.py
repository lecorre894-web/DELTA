import os,time
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from delta_qpu_cache import DeltaQPUCache
D=DeltaQPUCache(":memory:")
def ghz(n):
    q=QuantumCircuit(n);q.h(0)
    for i in range(n-1):q.cx(i,i+1)
    q.measure_all();return q
def run(qc,shots,cpu):
    if cpu:os.environ["DELTA_SIM_CPU"]="1"
    t=time.perf_counter();c,o=D._sim(qc,shots);w=time.perf_counter()-t;os.environ.pop("DELTA_SIM_CPU",None);return c,o,w
ok=True
for nm,qc in (("GHZ10",ghz(10)),("RANDOM8",random_circuit(8,12,measure=True,seed=7)),("RANDOM12",random_circuit(12,20,measure=True,seed=11))):
    c1,o1,w1=run(qc,4096,True);c2,o2,w2=run(qc,4096,False);k=set(c1)|set(c2);tvd=sum(abs(c1.get(x,0)-c2.get(x,0)) for x in k)/2/4096;ok&=tvd<0.01
    print("VERIF %-8s %s vs %s  comptages_identiques=%s  TVD=%.5f"%(nm,o1["engine"],o2["engine"],c1==c2,tvd))
for n in (16,20,24,26,28):
    w1=run(ghz(n),1024,True)[2] if n<=24 else None;c2,o2,w2=run(ghz(n),1024,False);p=(c2.get("0"*n,0)+c2.get("1"*n,0))/1024;ok&=p==1.0
    print("GHZ %2d  CPU=%s  RTX=%7.3f s (%s)  P(GHZ)=%.3f"%(n,"%7.3f s"%w1 if w1 else "   --    ",w2,o2["dtype"],p))
print("GPU_SIM_VALIDATION=%s"%("OK" if ok else "FAIL"))
