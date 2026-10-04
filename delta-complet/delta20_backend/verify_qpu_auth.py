import os
from qiskit_ibm_runtime import QiskitRuntimeService

TOKEN = os.environ.get("IQP_API_TOKEN")
INSTANCE = os.environ.get("IQP_INSTANCE_CRN")

print(f"[CHECK] Token présent : {'oui' if TOKEN else 'NON - manquant'}")
print(f"[CHECK] Instance (CRN) présent : {'oui' if INSTANCE else 'NON - manquant'}")

try:
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=TOKEN,
        instance=INSTANCE
    )
    backends = service.backends()
    print(f"VERDICT=OK connexion reussie, {len(backends)} backend(s) visible(s)")
    for b in backends:
        print(f"  - {b.name} | qubits={b.num_qubits} | operational={b.status().operational}")
except Exception as e:
    print(f"VERDICT=ECHEC {type(e).__name__}: {e}")
