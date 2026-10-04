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
        return pool.map(classical_preprocess, range(n_tasks))

def build_ghz_circuit(n_qubits=3):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

def select_best_backend(service):
    candidates = service.backends(operational=True, simulator=False)
    scored = []
    for b in candidates:
        try:
            props = b.properties()
            avg_err = sum(props.gate_error("cx", pair) for pair in b.coupling_map.get_edges()[:5]) / 5
            scored.append((avg_err, b))
        except Exception:
            continue
    if not scored:
        return service.least_busy(operational=True, simulator=False)
    scored.sort(key=lambda x: x[0])
    best_err, best_backend = scored[0]
    print(f"[QPU] Backend choisi : {best_backend.name} (erreur CX moyenne ~{best_err:.4f})")
    return best_backend

def post_select_ghz(counts, n_qubits=3):
    valid = {"0"*n_qubits: counts.get("0"*n_qubits, 0),
             "1"*n_qubits: counts.get("1"*n_qubits, 0)}
    total_valid = sum(valid.values())
    total_all = sum(counts.values())
    pop = total_valid / total_all if total_all else 0
    return pop, total_valid, total_all

def submit_qpu_job_async(n_qubits=3, shots=16384):
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN, instance=INSTANCE)
    backend = select_best_backend(service)
    circuit = build_ghz_circuit(n_qubits)
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)
    isa_circuit = pm.run(circuit)

    sampler = Sampler(mode=backend)
    sampler.options.update(dynamical_decoupling={"enable": True, "sequence_type": "XpXm"})

    job = sampler.run([isa_circuit], shots=shots)

    with open("qpu_job_pending.json", "w") as f:
        json.dump({"job_id": job.job_id(), "backend": backend.name,
                   "n_qubits": n_qubits, "shots": shots,
                   "submitted_at": time.time()}, f)
    print(f"[QPU] Job v3 soumis : {job.job_id()} sur {backend.name} ({n_qubits} qubits, {shots} shots)")
    return job.job_id()

if __name__ == "__main__":
    print("[XEON] Lancement du calcul classique local...")
    local_results = run_classical_batch()
    print(f"[XEON] Batch termine : {len(local_results)} taches")

    print("[QPU] Delegation GHZ3 avec backend optimal + post-selection...")
    submit_qpu_job_async()
