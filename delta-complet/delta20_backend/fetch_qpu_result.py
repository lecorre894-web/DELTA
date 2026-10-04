import json, os
from qiskit_ibm_runtime import QiskitRuntimeService

BASE_DIR = "/workspaces/DELTA/delta-complet/delta20_backend"
PENDING_FILE = os.path.join(BASE_DIR, "qpu_job_pending.json")
RESULT_FILE = os.path.join(BASE_DIR, "delta_qpu_result.json")

TOKEN = os.environ.get("IQP_API_TOKEN")

if not os.path.exists(PENDING_FILE):
    raise FileNotFoundError(f"Aucun job en attente trouvé à {PENDING_FILE}")

with open(PENDING_FILE) as f:
    pending = json.load(f)

service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
job = service.job(pending["job_id"])
status = job.status()
print(f"[QPU] Job {pending['job_id']} ({pending['backend']}) : {status}")

if status == "DONE":
    result = job.result()
    counts = result[0].data.c.get_counts()
    pop_dominant = max(counts.values()) / sum(counts.values())
    verdict = "OK" if pop_dominant > 0.5 else "ECHEC"
    print(f"VERDICT={verdict} pop={pop_dominant:.3f}")
    with open(RESULT_FILE, "w") as f:
        json.dump({"counts": counts, "pop": pop_dominant, "verdict": verdict,
                    "base_dir": BASE_DIR}, f, indent=2)
else:
    print("[QPU] Pas encore terminé, relance ce script plus tard.")
