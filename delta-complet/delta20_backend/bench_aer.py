from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import time, os

n = int(os.environ.get("QUBITS", 24))
qc = QuantumCircuit(n, n)
qc.h(range(n))
qc.measure(range(n), range(n))

sim = AerSimulator(max_parallel_threads=os.cpu_count())
tqc = transpile(qc, sim)
t0 = time.time()
result = sim.run(tqc, shots=1000).result()
print(f"{n} qubits | {os.cpu_count()} cœurs | {time.time()-t0:.2f}s")
