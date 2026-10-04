import json, os
from qiskit_ibm_runtime import QiskitRuntimeService

BASE_DIR = "/workspaces/DELTA/delta-complet/delta20_backend"
PENDING_FILE = os.path.join(BASE_DIR, "qpu_job_pending.json")
RESULT_FILE = os.path.join(BASE_DIR, "delta_qpu_result_v3.json")

TOKEN = os.environ.get("IQP_API_TOKEN")
INSTANCE = os.environ.get("IQP_INSTANCE_CRN")

with open(PENDING_FILE) as f:
    pending = json.load(f)

service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN, instance=INSTANCE)
job = service.job(pending["job_id"])
status = job.status()
print(f"[QPU] Job {pending['job_id']} ({pending['backend']}) : {status}")

if status == "DONE":
    result = job.result()
    counts = result[0].data.c.get_counts()
    n_qubits = pending["n_qubits"]
    valid = {"0"*n_qubits: counts.get("0"*n_qubits, 0), "1"*n_qubits: counts.get("1"*n_qubits, 0)}
    total_valid = sum(valid.values())
    total_all = sum(counts.values())
    pop_brute = max(counts.values()) / total_all
    pop_post_select = total_valid / total_all

    verdict_brute = "OK" if pop_brute > 0.5 else "ECHEC"
    verdict_post = "OK" if pop_post_select > 0.5 else "ECHEC"

    print(f"VERDICT_BRUT={verdict_brute} pop={pop_brute:.3f}")
    print(f"VERDICT_POST_SELECTION={verdict_post} pop={pop_post_select:.3f}")

    with open(RESULT_FILE, "w") as f:
        json.dump({"counts": counts, "pop_brute": pop_brute,
                    "pop_post_select": pop_post_select,
                    "verdict_brute": verdict_brute,
                    "verdict_post": verdict_post}, f, indent=2)
else:
    print("[QPU] Pas encore termine, relance ce script plus tard.")
