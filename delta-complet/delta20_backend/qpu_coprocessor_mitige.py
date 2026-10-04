import os, json, time
from multiprocessing import Pool, cpu_count
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

TOKEN = os.environ.get("IQP_API_TOKEN")
INSTANCE = os.environ.get("IQP_INSTANCE_CRN")
CORES = cpu_count()
print(f"[XEON] Coeurs locaux detectes : {CORES}")

def classical_preprocess(seed):
    import random
    random.seed(seed)
    return sum(random.random() for _ in range(100000))

def run_classical_batch(n_tasks=CORES):
    with Pool(processes=CORES) as pool:
        results = pool.map(classical_preprocess, range(n_tasks))
    return results

def build_ghz_circuit(n_qubits=3):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

def submit_qpu_job_async(n_qubits=3, shots=4096, resilience_level=1):
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN, instance=INSTANCE)
    backend = service.least_busy(operational=True, simulator=False)
    circuit = build_ghz_circuit(n_qubits)
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)
    isa_circuit = pm.run(circuit)

    sampler = Sampler(mode=backend)
    sampler.options.resilience_level = resilience_level
    sampler.options.dynamical_decoupling.enable = True

    job = sampler.run([isa_circuit], shots=shots)

    with open("qpu_job_pending.json", "w") as f:
        json.dump({"job_id": job.job_id(), "backend": backend.name,
                   "n_qubits": n_qubits, "resilience_level": resilience_level,
                   "submitted_at": time.time()}, f)
    print(f"[QPU] Job mitige soumis : {job.job_id()} sur {backend.name} ({n_qubits} qubits, resilience={resilience_level})")
    return job.job_id()

if __name__ == "__main__":
    print("[XEON] Lancement du calcul classique local...")
    local_results = run_classical_batch()
    print(f"[XEON] Batch termine : {len(local_results)} taches")

    print("[QPU] Delegation GHZ3 avec mitigation d'erreur...")
    submit_qpu_job_async(n_qubits=3, resilience_level=1)
