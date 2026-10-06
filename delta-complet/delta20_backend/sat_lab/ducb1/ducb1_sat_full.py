#!/usr/bin/env python3

import os
import time
import subprocess
from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseOracleGate
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_sampler import Sampler


# ================================================================
# DUCB-1 — UNIVERSAL SAT BENCHMARK
# ================================================================

SHOTS = 2048

print("=" * 78)
print(" DUCB-1 — UNIVERSAL SAT BENCHMARK")
print(" Ryzen / CaDiCaL / Kissat / DELTA / DELTA-QPU / IBM PHYSICAL QPU")
print("=" * 78)


# ================================================================
# COMMON SAT INSTANCE
#
# Unique satisfying assignment:
#
# x0 = 1
# x1 = 0
# x2 = 1
#
# CNF:
# ( x0 OR  x1)
# ( x0 OR ~x1)
# (~x0 OR  x2)
# ( x0 OR ~x2)
# (~x1 OR  x2)
# ================================================================

CLAUSES = [
    ( 1,  2),
    ( 1, -2),
    (-1,  3),
    ( 1, -3),
    (-2,  3),
]

NV = 3


def sat(bits):
    """
    bits[0] = x0
    bits[1] = x1
    bits[2] = x2
    """
    for clause in CLAUSES:
        ok = False

        for lit in clause:
            idx = abs(lit) - 1
            value = bool(bits[idx])

            if lit < 0:
                value = not value

            if value:
                ok = True
                break

        if not ok:
            return False

    return True


def bitstr_to_assignment(s):
    # Qiskit prints classical bits high -> low.
    # Reverse so array order becomes x0,x1,x2.
    s = s.replace(" ", "")
    return [int(x) for x in s[::-1]]


# ================================================================
# REFERENCE EXACT ENUMERATION
# ================================================================

print()
print("-" * 78)
print(" COMMON SAT — EXACT CPU REFERENCE")
print("-" * 78)

t0 = time.perf_counter()

cpu_solution = None

for x in range(1 << NV):
    bits = [(x >> i) & 1 for i in range(NV)]

    if sat(bits):
        cpu_solution = bits
        break

t1 = time.perf_counter()

print("CPU_SAT=", cpu_solution is not None)
print("CPU_MODEL=", cpu_solution)
print(f"CPU_TIME_SEC={t1-t0:.9f}")

if cpu_solution is None:
    raise SystemExit("REFERENCE SAT FAILURE")


# ================================================================
# WRITE COMMON DIMACS
# ================================================================

COMMON_CNF = "ducb1_common.cnf"

with open(COMMON_CNF, "w") as f:
    f.write("c DUCB-1 common physical-QPU SAT instance\n")
    f.write(f"p cnf {NV} {len(CLAUSES)}\n")

    for clause in CLAUSES:
        f.write(" ".join(map(str, clause)) + " 0\n")

print("COMMON_DIMACS=", os.path.abspath(COMMON_CNF))


# ================================================================
# CADICAL / KISSAT — COMMON INSTANCE
# ================================================================

SOLVERS = [
    (
        "CADICAL",
        os.path.expanduser("~/delta_sat_lab/cadical/build/cadical")
    ),
    (
        "KISSAT",
        os.path.expanduser("~/delta_sat_lab/kissat/build/kissat")
    ),
]

for name, exe in SOLVERS:

    print()
    print(f"{name}_COMMON_BEGIN=YES")

    if not os.path.exists(exe):
        print(f"{name}_COMMON=NOT_FOUND")
        continue

    t0 = time.perf_counter()

    p = subprocess.run(
        [exe, COMMON_CNF],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    t1 = time.perf_counter()

    out = p.stdout

    if "UNSATISFIABLE" in out:
        state = "UNSAT"
    elif "SATISFIABLE" in out:
        state = "SAT"
    else:
        state = "UNKNOWN"

    print(f"{name}_COMMON={state}")
    print(f"{name}_COMMON_RC={p.returncode}")
    print(f"{name}_COMMON_TIME_SEC={t1-t0:.9f}")


# ================================================================
# DELTA BIT64 — COMMON INSTANCE
# ================================================================

print()
print("-" * 78)
print(" DELTA BIT64 — COMMON SAT")
print("-" * 78)

delta_parser = os.path.expanduser(
    "~/DELTA/delta-complet/delta20_backend/delta_dimacs_sat.py"
)

if os.path.exists(delta_parser):

    t0 = time.perf_counter()

    p = subprocess.run(
        ["python", delta_parser, os.path.abspath(COMMON_CNF)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.path.dirname(delta_parser)
    )

    t1 = time.perf_counter()

    print(f"DELTA_BIT64_RC={p.returncode}")
    print(f"DELTA_BIT64_TIME_SEC={t1-t0:.9f}")

    # Compact output only.
    for line in p.stdout.splitlines():
        u = line.upper()
        if (
            "SAT" in u
            or "SOLUTION" in u
            or "VALID" in u
            or "TESTED" in u
        ):
            print("DELTA_BIT64>", line)

else:
    print("DELTA_BIT64=ENGINE_NOT_FOUND")


# ================================================================
# DELTA QPU LOGICAL — SAME COMMON SAT
#
# This remains explicitly LOGICAL/EMULATED.
# It is NOT counted as physical quantum hardware.
# ================================================================

print()
print("-" * 78)
print(" DELTA QPU LOGICAL — COMMON SAT")
print("-" * 78)

t0 = time.perf_counter()

logical_candidates = []

for x in range(1 << NV):
    bits = [(x >> i) & 1 for i in range(NV)]

    # DELTA logical scheduling layer:
    # candidate classification remains exact here.
    if sat(bits):
        logical_candidates.append(bits)

t1 = time.perf_counter()

print("DELTA_QBIT_EXECUTION=LOGICAL_EMULATED")
print("DELTA_QPU_SAT=", bool(logical_candidates))
print("DELTA_QPU_MODEL=", logical_candidates[0] if logical_candidates else None)
print(f"DELTA_QPU_TIME_SEC={t1-t0:.9f}")


# ================================================================
# IBM PHYSICAL QPU — SAT / GROVER
# ================================================================

print()
print("=" * 78)
print(" IBM PHYSICAL QPU — SAT GROVER")
print("=" * 78)

# Same clauses expressed directly as Boolean SAT expression.
#
# Variables correspond to x0,x1,x2.
#
expression = (
    "(x0 | x1)"
    " & (x0 | ~x1)"
    " & (~x0 | x2)"
    " & (x0 | ~x2)"
    " & (~x1 | x2)"
)

oracle_gate = PhaseOracleGate(
    expression,
    var_order=["x0", "x1", "x2"]
)

n = oracle_gate.num_qubits

print("IBM_SAT_VARIABLES=", NV)
print("IBM_ORACLE_QUBITS=", n)
print("IBM_SHOTS=", SHOTS)


# ------------------------------------------------
# Grover diffusion
# ------------------------------------------------

def diffuser(nq):

    qc = QuantumCircuit(nq)

    qc.h(range(nq))
    qc.x(range(nq))

    qc.h(nq - 1)

    if nq == 1:
        qc.z(0)
    else:
        qc.mcx(
            list(range(nq - 1)),
            nq - 1
        )

    qc.h(nq - 1)

    qc.x(range(nq))
    qc.h(range(nq))

    return qc.to_gate(label="DIFFUSER")


# ------------------------------------------------
# One Grover iteration.
#
# N=8 possible assignments, M=1 solution.
# One iteration gives strong amplification without
# making the physical circuit unnecessarily deep.
# ------------------------------------------------

grover = QuantumCircuit(n)

grover.h(range(n))

grover.append(
    oracle_gate,
    list(range(n))
)

grover.append(
    diffuser(n),
    list(range(n))
)

grover.measure_all()

print("IBM_LOGICAL_DEPTH=", grover.depth())


# ================================================================
# CONNECT TO REAL IBM HARDWARE
# ================================================================

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="open-instance"
)

backend = service.least_busy(
    operational=True,
    simulator=False,
    min_num_qubits=n
)

print("IBM_BACKEND=", backend.name)
print("IBM_BACKEND_QUBITS=", backend.num_qubits)

try:
    status = backend.status()
    print("IBM_PENDING_JOBS=", status.pending_jobs)
except Exception:
    print("IBM_PENDING_JOBS=UNKNOWN")


# ================================================================
# TRANSPILATION TO HARDWARE ISA
# ================================================================

t_trans0 = time.perf_counter()

pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=3
)

isa = pm.run(grover)

t_trans1 = time.perf_counter()

print("IBM_ISA_DEPTH=", isa.depth())
print("IBM_ISA_SIZE=", isa.size())
print("IBM_ISA_OPS=", dict(isa.count_ops()))
print(f"IBM_TRANSPILATION_SEC={t_trans1-t_trans0:.6f}")


# ================================================================
# PHYSICAL SUBMISSION
# ================================================================

sampler = Sampler(mode=backend)

print()
print("IBM_QPU_JOB_SUBMISSION=REAL")
print("IBM_QPU_EXECUTION=PHYSICAL_SUBMITTED")

t_submit = time.perf_counter()

job = sampler.run(
    [isa],
    shots=SHOTS
)

print("IBM_JOB_ID=", job.job_id())
print("IBM_WAITING_FOR_RESULT=YES")

result = job.result()

t_result = time.perf_counter()

print("IBM_RESULT_RECEIVED=YES")
print(f"IBM_SUBMIT_TO_RESULT_SEC={t_result-t_submit:.6f}")


# ================================================================
# PHYSICAL MEASUREMENTS
# ================================================================

pub = result[0]

data = pub.data

meas = None

if hasattr(data, "meas"):
    meas = data.meas
else:
    for name in dir(data):
        if name.startswith("_"):
            continue

        obj = getattr(data, name)

        if hasattr(obj, "get_bitstrings"):
            meas = obj
            break

if meas is None:
    raise RuntimeError(
        "IBM result received but measurement register not found."
    )

bitstrings = meas.get_bitstrings()

counts = Counter(bitstrings)

physical_valid = 0
physical_invalid = 0
valid_counts = Counter()

for raw, count in counts.items():

    assignment = bitstr_to_assignment(raw)

    if sat(assignment):
        physical_valid += count
        valid_counts[raw] += count
    else:
        physical_invalid += count

total = physical_valid + physical_invalid

rate = (
    physical_valid / total
    if total
    else 0.0
)


print()
print("-" * 78)
print(" IBM SAT PHYSICAL MEASUREMENTS")
print("-" * 78)

print("IBM_MEASURED_SHOTS=", total)
print("IBM_VALID_SAT_SHOTS=", physical_valid)
print("IBM_INVALID_SHOTS=", physical_invalid)
print(f"IBM_VALID_SAT_RATE={rate:.9f}")

print("IBM_TOP_MEASUREMENTS=")

for raw, count in counts.most_common(8):

    assignment = bitstr_to_assignment(raw)

    print(
        raw,
        count,
        "SAT=" + str(sat(assignment)),
        "ASSIGNMENT=" + str(assignment)
    )


if physical_valid:

    best_raw, best_count = valid_counts.most_common(1)[0]
    best_assignment = bitstr_to_assignment(best_raw)

    print("IBM_SAT=YES")
    print("IBM_MODEL=", best_assignment)
    print("IBM_MODEL_CLASSICALLY_VALIDATED=YES")

else:

    print("IBM_SAT=NO_VALID_SAMPLE")
    print("IBM_MODEL_CLASSICALLY_VALIDATED=NO")


# ================================================================
# FINAL DUCB-1 REPORT
# ================================================================

print()
print("=" * 78)
print(" DUCB-1 — FINAL CLASSIFICATION")
print("=" * 78)

print("CPU_BRIDGE_EXECUTION=PHYSICAL_MEASURED")
print("DELTA_QBIT_EXECUTION=LOGICAL_EMULATED")
print("IBM_QPU_EXECUTION=PHYSICAL_MEASURED")

print()
print("COMMON_SAT_REFERENCE_MODEL=", cpu_solution)
print("IBM_PHYSICAL_VALID_RATE=", f"{rate:.9f}")
print("IBM_PHYSICAL_BACKEND=", backend.name)
print("IBM_PHYSICAL_QUBITS_USED=", n)
print("IBM_PHYSICAL_SHOTS=", SHOTS)

print()
print("DUCB1_COMMON_SAT=COMPLETE")
print("DUCB1_IBM_PHYSICAL_SAT=COMPLETE")
print("=" * 78)
