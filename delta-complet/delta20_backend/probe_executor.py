import warnings;warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_ibm_runtime.executor_sampler import Sampler
b=FakeManilaV2();q=QuantumCircuit(2);q.h(0);q.cx(0,1);q.measure_all()
t=generate_preset_pass_manager(optimization_level=1,backend=b).run(q)
res=Sampler(mode=b).run([t],shots=1024).result()
print("TYPE_RESULT",type(res).__name__)
r=res[0];print("TYPE_PUB",type(r).__name__,[n for n in dir(r) if not n.startswith("_")])
d=getattr(r,"data",None);print("DATA",type(d).__name__,[n for n in dir(d) if not n.startswith("_")] if d is not None else None)
try:
    c=d.meas.get_counts();print("COUNTS",c);print("MIGRATION_VALIDATION=%s"%("OK" if c.get("00",0)+c.get("11",0)>800 else "FAIL"))
except Exception as e:print("FORMAT_DIFFERENT",repr(e)[:300])
