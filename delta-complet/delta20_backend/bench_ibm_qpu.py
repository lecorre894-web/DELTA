import os, time
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=os.environ["IQP_API_TOKEN"],
    instance=os.environ["IQP_INSTANCE_CRN"],
)

# Choisit le QPU le moins occupé parmi tes trois Heron
qpu = service.least_busy(operational=True, simulator=False)
print("QPU choisi :", qpu.name, "|", qpu.num_qubits, "qubits")

n = 30   # large pour un QPU (tes Heron en ont 156), impossible en local

qc = QuantumCircuit(n, n)
qc.h(0)
for q in range(n - 1):
    qc.cx(q, q + 1)     # état GHZ sur 30 qubits
qc.measure(range(n), range(n))

tqc = transpile(qc, qpu)
t0 = time.time()
job = SamplerV2(mode=qpu).run([tqc], shots=100)
result = job.result()
dt = time.time() - t0
print(f"QPU {qpu.name} | GHZ {n} qubits | {dt:.0f}s (file + exécution)")
