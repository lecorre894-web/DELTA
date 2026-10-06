#!/usr/bin/env python3
import os
import math
import time
import random
import subprocess
from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseOracleGate
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_sampler import Sampler

SEED = 894
SHOTS = 4096

# On augmente progressivement.
CANDIDATE_NV = [8, 10, 12, 14, 16]

# Garde-fous du benchmark.
MAX_ISA_DEPTH = 2500
MAX_2Q = 2500

CADICAL = os.path.expanduser(
    "~/delta_sat_lab/cadical/build/cadical"
)
KISSAT = os.path.expanduser(
    "~/delta_sat_lab/kissat/build/kissat"
)
DELTA = os.path.expanduser(
    "~/DELTA/delta-complet/delta20_backend/delta_dimacs_sat.py"
)

print("=" * 80)
print(" DUCB-1 HEAVY COMMON — SAT")
print(" Ryzen / CaDiCaL / Kissat / DELTA BIT64 / DELTA QPU / IBM QPU")
print("=" * 80)

# ----------------------------------------------------------------
# 3-SAT planté déterministe
# ----------------------------------------------------------------

def make_planted_3sat(nv):
    rng = random.Random(SEED + nv)

    planted = [rng.randrange(2) for _ in range(nv)]

    # Ratio volontairement dense.
    nc = int(4.25 * nv)

    clauses = []
    seen = set()

    while len(clauses) < nc:
        vars3 = rng.sample(range(nv), 3)
        clause = []

        for v in vars3:
            sign = 1 if rng.randrange(2) else -1
            clause.append(sign * (v + 1))

        # Garantir que l'affectation plantée satisfait la clause.
        def lit_value(lit):
            v = abs(lit) - 1
            val = planted[v]
            return val if lit > 0 else 1 - val

        if not any(lit_value(x) for x in clause):
            j = rng.randrange(3)
            v = abs(clause[j]) - 1
            clause[j] = (v + 1) if planted[v] else -(v + 1)

        key = tuple(sorted(clause, key=lambda x: abs(x)))

        if key not in seen:
            seen.add(key)
            clauses.append(tuple(clause))

    return planted, clauses


def sat(bits, clauses):
    for clause in clauses:
        ok = False

        for lit in clause:
            idx = abs(lit) - 1
            val = bool(bits[idx])

            if lit < 0:
                val = not val

            if val:
                ok = True
                break

        if not ok:
            return False

    return True


def exact_models(nv, clauses):
    models = []

    for x in range(1 << nv):
        bits = [(x >> i) & 1 for i in range(nv)]

        if sat(bits, clauses):
            models.append(bits)

    return models


def write_dimacs(path, nv, clauses):
    with open(path, "w") as f:
        f.write("c DUCB-1 HEAVY COMMON\n")
        f.write(f"p cnf {nv} {len(clauses)}\n")

        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")


def expression_from_clauses(clauses):
    out = []

    for clause in clauses:
        lits = []

        for lit in clause:
            v = abs(lit) - 1

            if lit > 0:
                lits.append(f"x{v}")
            else:
                lits.append(f"~x{v}")

        out.append("(" + " | ".join(lits) + ")")

    return " & ".join(out)


def bitstring_to_assignment(raw, nv):
    raw = raw.replace(" ", "")
    raw = raw[-nv:]

    return [int(x) for x in raw[::-1]]


# ----------------------------------------------------------------
# IBM — connexion physique, PAS ENCORE DE JOB
# ----------------------------------------------------------------

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="open-instance"
)

backend = service.least_busy(
    operational=True,
    simulator=False,
    min_num_qubits=max(CANDIDATE_NV)
)

print("IBM_BACKEND=", backend.name)
print("IBM_BACKEND_QUBITS=", backend.num_qubits)

try:
    print("IBM_PENDING_JOBS=", backend.status().pending_jobs)
except Exception:
    print("IBM_PENDING_JOBS=UNKNOWN")

pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=3
)

# ----------------------------------------------------------------
# Recherche automatique de la taille physique acceptable
# ----------------------------------------------------------------

selected = None

print()
print("-" * 80)
print(" IBM PRE-FLIGHT — NO QPU JOB")
print("-" * 80)

for nv in CANDIDATE_NV:

    planted, clauses = make_planted_3sat(nv)

    print()
    print(f"TRY_NV={nv}")
    print(f"TRY_CLAUSES={len(clauses)}")

    # Pour nv <= 16 : exact, donc on connaît M.
    t0 = time.perf_counter()
    models = exact_models(nv, clauses)
    t1 = time.perf_counter()

    M = len(models)
    N = 1 << nv

    print(f"EXACT_MODEL_COUNT={M}")
    print(f"EXACT_ENUM_SEC={t1-t0:.6f}")

    if M == 0:
        print("PREFLIGHT=REJECT_UNSAT")
        continue

    expr = expression_from_clauses(clauses)

    try:
        oracle = PhaseOracleGate(
            expr,
            var_order=[f"x{i}" for i in range(nv)]
        )
    except Exception as e:
        print("PREFLIGHT=ORACLE_FAILURE")
        print("DETAIL=", type(e).__name__, str(e))
        break

    # Optimal Grover count approximately pi/4 sqrt(N/M).
    iterations = max(
        1,
        int(math.floor(
            math.pi / 4.0 * math.sqrt(N / M)
        ))
    )

    print(f"GROVER_ITERATIONS={iterations}")

    def diffuser(nq):
        q = QuantumCircuit(nq)

        q.h(range(nq))
        q.x(range(nq))
        q.h(nq - 1)

        if nq == 1:
            q.z(0)
        else:
            q.mcx(
                list(range(nq - 1)),
                nq - 1
            )

        q.h(nq - 1)
        q.x(range(nq))
        q.h(range(nq))

        return q.to_gate(label="D")

    qc = QuantumCircuit(nv)
    qc.h(range(nv))

    D = diffuser(nv)

    for _ in range(iterations):
        qc.append(oracle, range(nv))
        qc.append(D, range(nv))

    qc.measure_all()

    print("LOGICAL_DEPTH=", qc.depth())

    try:
        tt0 = time.perf_counter()
        isa = pm.run(qc)
        tt1 = time.perf_counter()
    except Exception as e:
        print("PREFLIGHT=TRANSPILATION_FAILURE")
        print("DETAIL=", type(e).__name__, str(e))
        break

    ops = dict(isa.count_ops())

    twoq = 0

    # Heron native entangler.
    for opname in ("cz", "ecr", "cx"):
        twoq += int(ops.get(opname, 0))

    depth = isa.depth()

    print("ISA_DEPTH=", depth)
    print("ISA_SIZE=", isa.size())
    print("ISA_2Q=", twoq)
    print("ISA_OPS=", ops)
    print(f"TRANSPILATION_SEC={tt1-tt0:.6f}")

    if depth <= MAX_ISA_DEPTH and twoq <= MAX_2Q:
        print("PREFLIGHT=ACCEPT")
        selected = {
            "nv": nv,
            "planted": planted,
            "clauses": clauses,
            "models": models,
            "iterations": iterations,
            "qc": qc,
            "isa": isa,
            "depth": depth,
            "twoq": twoq,
        }
    else:
        print("PREFLIGHT=REJECT_TOO_HEAVY")
        break


if selected is None:
    raise SystemExit(
        "NO_ACCEPTABLE_IBM_SAT_CIRCUIT — NO QPU JOB SUBMITTED"
    )

nv = selected["nv"]
clauses = selected["clauses"]
models = selected["models"]
isa = selected["isa"]

cnf = f"ducb1_heavy_{nv}.cnf"
write_dimacs(cnf, nv, clauses)

print()
print("=" * 80)
print(" SELECTED COMMON SAT")
print("=" * 80)

print("NV=", nv)
print("CLAUSES=", len(clauses))
print("EXACT_SOLUTIONS=", len(models))
print("GROVER_ITERATIONS=", selected["iterations"])
print("IBM_ISA_DEPTH=", selected["depth"])
print("IBM_ISA_2Q=", selected["twoq"])
print("DIMACS=", os.path.abspath(cnf))

# ----------------------------------------------------------------
# RYZEN EXACT
# ----------------------------------------------------------------

print()
print("-" * 80)
print(" RYZEN EXACT")
print("-" * 80)

t0 = time.perf_counter()
cpu_models = exact_models(nv, clauses)
t1 = time.perf_counter()

print("RYZEN_SAT=", bool(cpu_models))
print("RYZEN_MODEL_COUNT=", len(cpu_models))
print(f"RYZEN_EXACT_SEC={t1-t0:.9f}")
print("CPU_BRIDGE_EXECUTION=PHYSICAL_MEASURED")

# ----------------------------------------------------------------
# CaDiCaL / Kissat
# ----------------------------------------------------------------

for name, exe in [
    ("CADICAL", CADICAL),
    ("KISSAT", KISSAT)
]:
    print()
    print("-" * 80)
    print(" ", name)
    print("-" * 80)

    t0 = time.perf_counter()

    p = subprocess.run(
        [exe, cnf],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    t1 = time.perf_counter()

    if "UNSATISFIABLE" in p.stdout:
        state = "UNSAT"
    elif "SATISFIABLE" in p.stdout:
        state = "SAT"
    else:
        state = "UNKNOWN"

    print(f"{name}_RESULT={state}")
    print(f"{name}_RC={p.returncode}")
    print(f"{name}_SEC={t1-t0:.9f}")

# ----------------------------------------------------------------
# DELTA BIT64 / V1
# ----------------------------------------------------------------

print()
print("-" * 80)
print(" DELTA BIT64 / V1")
print("-" * 80)

t0 = time.perf_counter()

p = subprocess.run(
    ["python", DELTA, os.path.abspath(cnf)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=os.path.dirname(DELTA)
)

t1 = time.perf_counter()

print("DELTA_V1_RC=", p.returncode)
print(f"DELTA_V1_TOTAL_SEC={t1-t0:.9f}")

for line in p.stdout.splitlines():
    u = line.upper()

    if any(k in u for k in (
        "SAT",
        "SOLUTION",
        "TESTED",
        "VALID",
        "CANDIDATE"
    )):
        print("DELTA_V1>", line)

# ----------------------------------------------------------------
# DELTA QPU LOGICAL ON
# Same exact SAT, logical scheduling classification.
# ----------------------------------------------------------------

print()
print("-" * 80)
print(" DELTA QPU LOGICAL ON")
print("-" * 80)

t0 = time.perf_counter()

logical_model = None

for x in range(1 << nv):
    bits = [(x >> i) & 1 for i in range(nv)]

    if sat(bits, clauses):
        logical_model = bits
        break

t1 = time.perf_counter()

print("DELTA_QBIT_EXECUTION=LOGICAL_EMULATED")
print("DELTA_QPU_ON_SAT=", logical_model is not None)
print("DELTA_QPU_ON_MODEL=", logical_model)
print(f"DELTA_QPU_ON_SEC={t1-t0:.9f}")

# ----------------------------------------------------------------
# IBM PHYSICAL QPU
# THIS IS THE ONLY PHYSICAL QPU SUBMISSION.
# ----------------------------------------------------------------

print()
print("=" * 80)
print(" IBM PHYSICAL QPU — HEAVY COMMON SAT")
print("=" * 80)

print("IBM_QPU_JOB_SUBMISSION=REAL")
print("IBM_QPU_EXECUTION=PHYSICAL_SUBMITTED")
print("IBM_SHOTS=", SHOTS)

sampler = Sampler(mode=backend)

t0 = time.perf_counter()

job = sampler.run(
    [isa],
    shots=SHOTS
)

print("IBM_JOB_ID=", job.job_id())
print("IBM_WAITING=YES")

result = job.result()

t1 = time.perf_counter()

print("IBM_RESULT_RECEIVED=YES")
print(f"IBM_SUBMIT_TO_RESULT_SEC={t1-t0:.6f}")

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
    raise RuntimeError("IBM measurement register not found")

counts = Counter(meas.get_bitstrings())

valid = Counter()
invalid = 0

for raw, count in counts.items():
    assignment = bitstring_to_assignment(raw, nv)

    if sat(assignment, clauses):
        valid[raw] += count
    else:
        invalid += count

valid_shots = sum(valid.values())
total_shots = valid_shots + invalid

rate = (
    valid_shots / total_shots
    if total_shots
    else 0.0
)

print()
print("-" * 80)
print(" IBM PHYSICAL SAT RESULT")
print("-" * 80)

print("IBM_MEASURED_SHOTS=", total_shots)
print("IBM_VALID_SAT_SHOTS=", valid_shots)
print("IBM_INVALID_SHOTS=", invalid)
print(f"IBM_VALID_SAT_RATE={rate:.9f}")

print("IBM_TOP_RESULTS=")

for raw, count in counts.most_common(12):
    a = bitstring_to_assignment(raw, nv)

    print(
        raw,
        count,
        "SAT=" + str(sat(a, clauses))
    )

if valid:
    best_raw, best_count = valid.most_common(1)[0]
    best = bitstring_to_assignment(best_raw, nv)

    print("IBM_SAT=YES")
    print("IBM_MODEL=", best)
    print("IBM_MODEL_COUNT_IN_SHOTS=", best_count)
    print("IBM_MODEL_CLASSICALLY_VALIDATED=YES")
else:
    print("IBM_SAT=NO_VALID_SAMPLE")
    print("IBM_MODEL_CLASSICALLY_VALIDATED=NO")

# ----------------------------------------------------------------
# FINAL
# ----------------------------------------------------------------

print()
print("=" * 80)
print(" DUCB-1 HEAVY COMMON — FINAL")
print("=" * 80)

print("COMMON_NV=", nv)
print("COMMON_CLAUSES=", len(clauses))
print("COMMON_EXACT_SOLUTIONS=", len(models))

print()
print("CPU_BRIDGE_EXECUTION=PHYSICAL_MEASURED")
print("DELTA_QBIT_EXECUTION=LOGICAL_EMULATED")
print("IBM_QPU_EXECUTION=PHYSICAL_MEASURED")

print()
print("IBM_BACKEND=", backend.name)
print("IBM_BACKEND_PHYSICAL_QUBITS=", backend.num_qubits)
print("IBM_LOGICAL_VARIABLES_USED=", nv)
print("IBM_GROVER_ITERATIONS=", selected["iterations"])
print("IBM_ISA_DEPTH=", selected["depth"])
print("IBM_ISA_2Q=", selected["twoq"])
print("IBM_SHOTS=", SHOTS)
print(f"IBM_VALID_SAT_RATE={rate:.9f}")

print()
print("DUCB1_HEAVY_COMMON=COMPLETE")
print("=" * 80)
