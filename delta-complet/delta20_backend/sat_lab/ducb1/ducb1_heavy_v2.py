#!/usr/bin/env python3

import os
import time
import random
import subprocess
from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseOracleGate
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_sampler import Sampler

# ================================================================
# DUCB-1 HEAVY V2 ADAPTIVE
# ================================================================

SEED = 894
SHOTS = 4096

VARIABLES = [8, 10, 12, 14]

# Clauses = round(ratio * variables)
RATIOS = [1.50, 2.00, 2.50, 3.00, 3.50]

GROVER_ITERATIONS = [1, 2, 3]

# Physical circuit envelope.
MAX_DEPTH = 6000
MAX_2Q = 3000

CADICAL = os.path.expanduser(
    "~/delta_sat_lab/cadical/build/cadical"
)

KISSAT = os.path.expanduser(
    "~/delta_sat_lab/kissat/build/kissat"
)

DELTA = os.path.expanduser(
    "~/DELTA/delta-complet/delta20_backend/delta_dimacs_sat.py"
)

print("=" * 82)
print(" DUCB-1 HEAVY V2 ADAPTIVE")
print(" Ryzen / CaDiCaL / Kissat / DELTA V1 / DELTA QPU ON / IBM QPU")
print("=" * 82)

print("MAX_DEPTH=", MAX_DEPTH)
print("MAX_2Q=", MAX_2Q)
print("SHOTS=", SHOTS)


# ================================================================
# SAT utilities
# ================================================================

def satisfies(bits, clauses):

    for clause in clauses:

        good = False

        for lit in clause:

            idx = abs(lit) - 1
            val = bool(bits[idx])

            if lit < 0:
                val = not val

            if val:
                good = True
                break

        if not good:
            return False

    return True


def exact_models(nv, clauses):

    models = []

    for x in range(1 << nv):

        bits = [
            (x >> i) & 1
            for i in range(nv)
        ]

        if satisfies(bits, clauses):
            models.append(bits)

    return models


def make_planted(nv, nc):

    rng = random.Random(
        SEED + nv * 10000 + nc
    )

    planted = [
        rng.randrange(2)
        for _ in range(nv)
    ]

    clauses = []
    seen = set()

    while len(clauses) < nc:

        vv = rng.sample(
            range(nv),
            3
        )

        c = []

        for v in vv:

            if rng.randrange(2):
                lit = v + 1
            else:
                lit = -(v + 1)

            c.append(lit)

        def lit_value(lit):

            idx = abs(lit) - 1
            val = planted[idx]

            if lit < 0:
                val = 1 - val

            return val

        # Plant at least one true literal.
        if not any(
            lit_value(l)
            for l in c
        ):

            j = rng.randrange(3)
            v = abs(c[j]) - 1

            if planted[v]:
                c[j] = v + 1
            else:
                c[j] = -(v + 1)

        key = tuple(
            sorted(
                c,
                key=lambda z: abs(z)
            )
        )

        if key not in seen:
            seen.add(key)
            clauses.append(tuple(c))

    return planted, clauses


def write_dimacs(path, nv, clauses):

    with open(path, "w") as f:

        f.write(
            "c DUCB-1 HEAVY V2 ADAPTIVE\n"
        )

        f.write(
            f"p cnf {nv} {len(clauses)}\n"
        )

        for c in clauses:

            f.write(
                " ".join(map(str, c))
                + " 0\n"
            )


def make_expression(clauses):

    cc = []

    for clause in clauses:

        ll = []

        for lit in clause:

            idx = abs(lit) - 1

            if lit > 0:
                ll.append(f"x{idx}")
            else:
                ll.append(f"~x{idx}")

        cc.append(
            "("
            + " | ".join(ll)
            + ")"
        )

    return " & ".join(cc)


def diffuser(n):

    qc = QuantumCircuit(n)

    qc.h(range(n))
    qc.x(range(n))

    qc.h(n - 1)

    qc.mcx(
        list(range(n - 1)),
        n - 1
    )

    qc.h(n - 1)

    qc.x(range(n))
    qc.h(range(n))

    return qc.to_gate(
        label="DELTA-DIFFUSER"
    )


def decode(raw, nv):

    raw = raw.replace(" ", "")

    raw = raw[-nv:]

    return [
        int(x)
        for x in raw[::-1]
    ]


# ================================================================
# IBM CONNECTION — DISCOVERY ONLY
# ================================================================

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="open-instance"
)

backend = service.least_busy(
    operational=True,
    simulator=False,
    min_num_qubits=max(VARIABLES)
)

print()
print("IBM_BACKEND=", backend.name)
print(
    "IBM_BACKEND_QUBITS=",
    backend.num_qubits
)

try:
    print(
        "IBM_PENDING_JOBS=",
        backend.status().pending_jobs
    )
except Exception:
    print(
        "IBM_PENDING_JOBS=UNKNOWN"
    )


# ================================================================
# TRANSPILER
# ================================================================

pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=3
)


# ================================================================
# ADAPTIVE SEARCH
# NO QPU SUBMISSION HERE
# ================================================================

print()
print("=" * 82)
print(" ADAPTIVE IBM PREFLIGHT — ZERO QPU JOBS")
print("=" * 82)

accepted = []

for nv in VARIABLES:

    print()
    print("#" * 82)
    print(f" VARIABLE LEVEL = {nv}")
    print("#" * 82)

    for ratio in RATIOS:

        nc = max(
            1,
            round(nv * ratio)
        )

        planted, clauses = make_planted(
            nv,
            nc
        )

        t0 = time.perf_counter()

        models = exact_models(
            nv,
            clauses
        )

        exact_sec = (
            time.perf_counter() - t0
        )

        if not models:

            print(
                f"NV={nv} NC={nc} "
                "UNSAT -> SKIP"
            )

            continue

        expression = make_expression(
            clauses
        )

        try:

            oracle = PhaseOracleGate(
                expression,
                var_order=[
                    f"x{i}"
                    for i in range(nv)
                ]
            )

        except Exception as e:

            print(
                f"NV={nv} NC={nc} "
                "ORACLE_FAILURE",
                type(e).__name__,
                str(e)
            )

            continue

        D = diffuser(nv)

        for iterations in GROVER_ITERATIONS:

            qc = QuantumCircuit(nv)

            qc.h(range(nv))

            for _ in range(iterations):

                qc.append(
                    oracle,
                    range(nv)
                )

                qc.append(
                    D,
                    range(nv)
                )

            qc.measure_all()

            try:

                tt0 = time.perf_counter()

                isa = pm.run(qc)

                trans_sec = (
                    time.perf_counter()
                    - tt0
                )

            except Exception as e:

                print(
                    f"NV={nv} NC={nc} "
                    f"ITER={iterations} "
                    "TRANSPILE_FAILURE",
                    type(e).__name__,
                    str(e)
                )

                continue

            ops = dict(
                isa.count_ops()
            )

            twoq = sum(
                int(
                    ops.get(k, 0)
                )
                for k in (
                    "cz",
                    "cx",
                    "ecr"
                )
            )

            depth = isa.depth()

            ok = (
                depth <= MAX_DEPTH
                and
                twoq <= MAX_2Q
            )

            print(
                f"NV={nv:2d} "
                f"NC={nc:3d} "
                f"M={len(models):5d} "
                f"ITER={iterations} "
                f"DEPTH={depth:5d} "
                f"2Q={twoq:5d} "
                f"TRANS={trans_sec:.3f}s "
                f"{'ACCEPT' if ok else 'REJECT'}"
            )

            if ok:

                accepted.append({
                    "nv": nv,
                    "nc": nc,
                    "ratio": ratio,
                    "planted": planted,
                    "clauses": clauses,
                    "models": models,
                    "iterations": iterations,
                    "qc": qc,
                    "isa": isa,
                    "depth": depth,
                    "twoq": twoq,
                    "ops": ops,
                    "exact_sec": exact_sec,
                    "trans_sec": trans_sec,
                })


# ================================================================
# SELECT HEAVIEST ACCEPTED CIRCUIT
# ================================================================

if not accepted:

    raise SystemExit(
        "V2_NO_ACCEPTABLE_CIRCUIT — "
        "NO QPU JOB SUBMITTED"
    )

# Priority:
# 1. variables
# 2. clauses
# 3. Grover iterations
# 4. physical two-qubit work
selected = max(
    accepted,
    key=lambda r: (
        r["nv"],
        r["nc"],
        r["iterations"],
        r["twoq"]
    )
)

nv = selected["nv"]
clauses = selected["clauses"]
models = selected["models"]
isa = selected["isa"]

cnf = (
    f"ducb1_heavy_v2_"
    f"{nv}v_{len(clauses)}c.cnf"
)

write_dimacs(
    cnf,
    nv,
    clauses
)


print()
print("=" * 82)
print(" V2 SELECTED HEAVY COMMON SAT")
print("=" * 82)

print("SELECTED_NV=", nv)
print(
    "SELECTED_CLAUSES=",
    len(clauses)
)
print(
    "SELECTED_EXACT_SOLUTIONS=",
    len(models)
)
print(
    "SELECTED_ITERATIONS=",
    selected["iterations"]
)
print(
    "SELECTED_ISA_DEPTH=",
    selected["depth"]
)
print(
    "SELECTED_ISA_2Q=",
    selected["twoq"]
)
print(
    "SELECTED_ISA_OPS=",
    selected["ops"]
)
print(
    "SELECTED_DIMACS=",
    os.path.abspath(cnf)
)


# ================================================================
# RYZEN EXACT
# ================================================================

print()
print("=" * 82)
print(" RYZEN SUPPORT / EXACT ENUMERATION")
print("=" * 82)

t0 = time.perf_counter()

ryzen_models = exact_models(
    nv,
    clauses
)

ryzen_sec = (
    time.perf_counter() - t0
)

print(
    "RYZEN_RESULT=",
    "SAT"
    if ryzen_models
    else "UNSAT"
)

print(
    "RYZEN_MODEL_COUNT=",
    len(ryzen_models)
)

print(
    f"RYZEN_SEC={ryzen_sec:.9f}"
)

print(
    "CPU_BRIDGE_EXECUTION="
    "PHYSICAL_MEASURED"
)


# ================================================================
# CADICAL / KISSAT
# ================================================================

solver_results = {}

for name, exe in [
    ("CADICAL", CADICAL),
    ("KISSAT", KISSAT)
]:

    print()
    print("=" * 82)
    print(" ", name)
    print("=" * 82)

    t0 = time.perf_counter()

    p = subprocess.run(
        [exe, cnf],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    sec = (
        time.perf_counter() - t0
    )

    if "UNSATISFIABLE" in p.stdout:
        state = "UNSAT"

    elif "SATISFIABLE" in p.stdout:
        state = "SAT"

    else:
        state = "UNKNOWN"

    solver_results[name] = (
        state,
        sec
    )

    print(
        f"{name}_RESULT={state}"
    )

    print(
        f"{name}_RC={p.returncode}"
    )

    print(
        f"{name}_SEC={sec:.9f}"
    )


# ================================================================
# DELTA V1 / BIT64
# ================================================================

print()
print("=" * 82)
print(" DELTA V1 / BIT64")
print("=" * 82)

t0 = time.perf_counter()

p = subprocess.run(
    [
        "python",
        DELTA,
        os.path.abspath(cnf)
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=os.path.dirname(DELTA)
)

delta_sec = (
    time.perf_counter() - t0
)

print(
    "DELTA_V1_RC=",
    p.returncode
)

print(
    f"DELTA_V1_SEC="
    f"{delta_sec:.9f}"
)

for line in p.stdout.splitlines():

    u = line.upper()

    if any(
        k in u
        for k in (
            "SAT",
            "SOLUTION",
            "TESTED",
            "VALID",
            "CANDIDATE"
        )
    ):
        print(
            "DELTA_V1>",
            line
        )


# ================================================================
# DELTA QPU ON — LOGICAL/EMULATED
# ================================================================

print()
print("=" * 82)
print(" DELTA QPU ON")
print("=" * 82)

t0 = time.perf_counter()

delta_qpu_model = None
delta_qpu_tested = 0

for x in range(1 << nv):

    delta_qpu_tested += 1

    bits = [
        (x >> i) & 1
        for i in range(nv)
    ]

    if satisfies(
        bits,
        clauses
    ):

        delta_qpu_model = bits
        break

delta_qpu_sec = (
    time.perf_counter() - t0
)

print(
    "DELTA_QBIT_EXECUTION="
    "LOGICAL_EMULATED"
)

print(
    "DELTA_QPU_ON_RESULT=",
    "SAT"
    if delta_qpu_model
    else "UNSAT"
)

print(
    "DELTA_QPU_ON_TESTED=",
    delta_qpu_tested
)

print(
    "DELTA_QPU_ON_MODEL=",
    delta_qpu_model
)

print(
    f"DELTA_QPU_ON_SEC="
    f"{delta_qpu_sec:.9f}"
)


# ================================================================
# IBM PHYSICAL QPU
# ONE REAL JOB ONLY
# ================================================================

print()
print("=" * 82)
print(" IBM PHYSICAL QPU — V2 HEAVY COMMON")
print("=" * 82)

print(
    "IBM_BACKEND=",
    backend.name
)

print(
    "IBM_PHYSICAL_QUBITS=",
    backend.num_qubits
)

print(
    "IBM_LOGICAL_SAT_VARIABLES=",
    nv
)

print(
    "IBM_CLAUSES=",
    len(clauses)
)

print(
    "IBM_GROVER_ITERATIONS=",
    selected["iterations"]
)

print(
    "IBM_ISA_DEPTH=",
    selected["depth"]
)

print(
    "IBM_ISA_2Q=",
    selected["twoq"]
)

print(
    "IBM_SHOTS=",
    SHOTS
)

sampler = Sampler(
    mode=backend
)

print()
print(
    "IBM_QPU_JOB_SUBMISSION=REAL"
)

print(
    "IBM_QPU_EXECUTION="
    "PHYSICAL_SUBMITTED"
)

t0 = time.perf_counter()

job = sampler.run(
    [isa],
    shots=SHOTS
)

print(
    "IBM_JOB_ID=",
    job.job_id()
)

print(
    "IBM_WAITING_FOR_RESULT=YES"
)

result = job.result()

ibm_wall = (
    time.perf_counter() - t0
)

print(
    "IBM_RESULT_RECEIVED=YES"
)

print(
    f"IBM_SUBMIT_TO_RESULT_SEC="
    f"{ibm_wall:.6f}"
)


# ================================================================
# IBM VALIDATION
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

        obj = getattr(
            data,
            name
        )

        if hasattr(
            obj,
            "get_bitstrings"
        ):
            meas = obj
            break


if meas is None:

    raise RuntimeError(
        "IBM result received, "
        "measurement register missing"
    )


counts = Counter(
    meas.get_bitstrings()
)

valid = Counter()
invalid = 0

for raw, count in counts.items():

    assignment = decode(
        raw,
        nv
    )

    if satisfies(
        assignment,
        clauses
    ):

        valid[raw] += count

    else:

        invalid += count


valid_shots = sum(
    valid.values()
)

total_shots = (
    valid_shots
    + invalid
)

valid_rate = (
    valid_shots / total_shots
    if total_shots
    else 0.0
)


print()
print("=" * 82)
print(" IBM PHYSICAL SAT MEASUREMENTS")
print("=" * 82)

print(
    "IBM_MEASURED_SHOTS=",
    total_shots
)

print(
    "IBM_VALID_SAT_SHOTS=",
    valid_shots
)

print(
    "IBM_INVALID_SHOTS=",
    invalid
)

print(
    f"IBM_VALID_SAT_RATE="
    f"{valid_rate:.9f}"
)

print(
    "IBM_TOP_RESULTS="
)

for raw, count in counts.most_common(12):

    assignment = decode(
        raw,
        nv
    )

    print(
        raw,
        count,
        "SAT="
        + str(
            satisfies(
                assignment,
                clauses
            )
        )
    )


if valid:

    best_raw, best_count = (
        valid.most_common(1)[0]
    )

    best_model = decode(
        best_raw,
        nv
    )

    print(
        "IBM_SAT=YES"
    )

    print(
        "IBM_MODEL=",
        best_model
    )

    print(
        "IBM_MODEL_SHOTS=",
        best_count
    )

    print(
        "IBM_MODEL_CLASSICALLY_VALIDATED="
        "YES"
    )

else:

    print(
        "IBM_SAT=NO_VALID_SAMPLE"
    )

    print(
        "IBM_MODEL_CLASSICALLY_VALIDATED="
        "NO"
    )


# ================================================================
# FINAL REPORT
# ================================================================

print()
print("=" * 82)
print(" DUCB-1 HEAVY V2 — FINAL")
print("=" * 82)

print(
    f"COMMON_SAT="
    f"{nv} VARIABLES / "
    f"{len(clauses)} CLAUSES"
)

print(
    "EXACT_SOLUTIONS=",
    len(models)
)

print()

print(
    f"RYZEN_EXACT_SEC="
    f"{ryzen_sec:.9f}"
)

for name in (
    "CADICAL",
    "KISSAT"
):

    state, sec = (
        solver_results[name]
    )

    print(
        f"{name}_RESULT={state} "
        f"SEC={sec:.9f}"
    )

print(
    f"DELTA_V1_SEC="
    f"{delta_sec:.9f}"
)

print(
    f"DELTA_QPU_ON_SEC="
    f"{delta_qpu_sec:.9f}"
)

print(
    f"IBM_SUBMIT_TO_RESULT_SEC="
    f"{ibm_wall:.6f}"
)

print(
    f"IBM_VALID_SAT_RATE="
    f"{valid_rate:.9f}"
)

print()

print(
    "CPU_BRIDGE_EXECUTION="
    "PHYSICAL_MEASURED"
)

print(
    "DELTA_QBIT_EXECUTION="
    "LOGICAL_EMULATED"
)

print(
    "IBM_QPU_EXECUTION="
    "PHYSICAL_MEASURED"
)

print(
    "IBM_BACKEND=",
    backend.name
)

print(
    "IBM_BACKEND_QUBITS=",
    backend.num_qubits
)

print(
    "IBM_USED_VARIABLE_QUBITS=",
    nv
)

print(
    "IBM_ISA_DEPTH=",
    selected["depth"]
)

print(
    "IBM_ISA_2Q=",
    selected["twoq"]
)

print(
    "IBM_SHOTS=",
    SHOTS
)

print()
print(
    "DUCB1_HEAVY_V2_COMPLETE=YES"
)
print("=" * 82)
