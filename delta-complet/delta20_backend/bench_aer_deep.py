from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import time, os

n = int(os.environ.get("QUBITS", 24))
depth = int(os.environ.get("DEPTH", 50))

qc = QuantumCircuit(n, n)
for _ in range(depth):
    for q in range(n):
        qc.rx(0.7, q)          # rotations
    for q in range(n - 1):
        qc.cx(q, q + 1)        # intrication en chaîne
qc.measure(range(n), range(n))

sim = AerSimulator(max_parallel_threads=os.cpu_count())
tqc = transpile(qc, sim)
t0 = time.time()
result = sim.run(tqc, shots=1000).result()
print(f"{n} qubits x {depth} couches | {os.cpu_count()} cœurs | {time.time()-t0:.2f}s")
