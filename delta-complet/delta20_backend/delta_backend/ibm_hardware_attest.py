import os
import json
from qiskit_ibm_runtime import QiskitRuntimeService

result = {
    "hardwareAttested": False,
    "quantumProvider": None,
    "quantumBackend": None,
    "physicalQubits": 0,
    "physicalQuantumHardware": False,
    "attestation": "FAILED"
}

try:
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ["IQP_API_TOKEN"],
        instance=os.environ["IQP_INSTANCE_CRN"]
    )

    backends = service.backends(
        simulator=False,
        operational=True
    )

    if not backends:
        result["attestation"] = "NO_OPERATIONAL_PHYSICAL_QPU"
    else:
        def pending(b):
            try:
                return b.status().pending_jobs
            except Exception:
                return 10**18

        qpu = min(backends, key=pending)

        result = {
            "hardwareAttested": True,
            "quantumProvider": "IBM_QUANTUM",
            "quantumBackend": qpu.name,
            "physicalQubits": qpu.num_qubits,
            "physicalQuantumHardware": True,
            "attestation": "IBM_QPU_VERIFIED"
        }

except Exception as e:
    result["attestation"] = "IBM_ATTESTATION_ERROR"
    result["errorType"] = type(e).__name__

print(json.dumps(result))
