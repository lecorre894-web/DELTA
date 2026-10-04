import os, time
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=os.environ["IQP_API_TOKEN"],
    instance=os.environ["IQP_INSTANCE_CRN"],
)

print("Backends simulateurs disponibles :")
for b in service.backends():
    print("  -", b.name, "| sim:", getattr(b, "simulator", "?"))

sim = service.backend("simulator_statevector")
n = 30

qc = QuantumCircuit(n, n)
for q in range(n):
    qc.rx(0.7, q)
for q in range(n - 1):
    qc.cx(q, q + 1)
qc.measure(range(n), range(n))

tqc = transpile(qc, sim)
t0 = time.time()
job = SamplerV2(mode=sim).run([tqc], shots=100)
result = job.result()
print(f"IBM cloud | {n} qubits | {time.time()-t0:.1f}s au total")
