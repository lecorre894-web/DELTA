import os
import json

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_sampler import Sampler

out = {
    "qpuExecution": "FAILED",
    "quantumProvider": None,
    "quantumBackend": None,
    "physicalQubits": 0,
    "physicalQuantumHardware": False,
    "ibmJobId": None,
    "shots": 0,
    "counts": {}
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
        out["qpuExecution"] = "NO_OPERATIONAL_PHYSICAL_QPU"
    else:
        def pending(b):
            try:
                return b.status().pending_jobs
            except Exception:
                return 10**18

        backend = min(backends, key=pending)

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        isa = transpile(
            qc,
            backend=backend,
            optimization_level=1
        )

        sampler = Sampler(mode=backend)
        job = sampler.run([isa], shots=1024)

        # Job ID is obtained before waiting for hardware completion.
        job_id = job.job_id()

        result = job.result()
        counts = result[0].data.meas.get_counts()
        shots = sum(counts.values())

        out = {
            "qpuExecution": "PHYSICAL_MEASURED",
            "quantumProvider": "IBM_QUANTUM",
            "quantumBackend": backend.name,
            "physicalQubits": backend.num_qubits,
            "physicalQuantumHardware": True,
            "ibmJobId": job_id,
            "shots": shots,
            "counts": counts
        }

except Exception as e:
    out["qpuExecution"] = "IBM_QPU_EXECUTION_ERROR"
    out["errorType"] = type(e).__name__
    out["error"] = str(e)[:300]

print(json.dumps(out))
