#!/usr/bin/env python3
import os
import re
import time
import hashlib
import subprocess
import multiprocessing as mp
from pathlib import Path

TARGET = Path("DELTA_TARGET.cnf").resolve()
CADICAL = Path.home() / "delta_sat_lab/cadical/build/cadical"

EXPECTED_SHA = "6be9bfe6b44454a061a2e93350f589643f8a16f5bca6babdaa50a2d3d7069361"

WORKERS = min(32, os.cpu_count() or 1)
CUBE_BITS = 8
PROBE_VARS = 625

# Probe volontairement minuscule :
# on veut mesurer la réaction initiale de la formule,
# pas résoudre 1250 problèmes complets.
PROBE_CONFLICTS = 100


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_dimacs(path):
    text = Path(path).read_text(encoding="ascii")

    m = re.search(
        r"^p cnf\s+(\d+)\s+(\d+)",
        text,
        re.MULTILINE
    )

    if not m:
        raise RuntimeError("DIMACS header missing")

    return text, int(m.group(1)), int(m.group(2))


def branch_cnf(text, nv, nc, literals):
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if line.startswith("p cnf "):
            lines[i] = f"p cnf {nv} {nc + len(literals)}"
            break

    units = "\n".join(f"{x} 0" for x in literals)

    return (
        "\n".join(lines)
        + "\n"
        + units
        + "\n"
    ).encode()


TEXT, NV, NC = load_dimacs(TARGET)


def probe_one(job):
    var, sign = job

    lit = var if sign > 0 else -var
    data = branch_cnf(TEXT, NV, NC, [lit])

    start = time.perf_counter()

    p = subprocess.run(
        [
            str(CADICAL),
            f"--conflicts={PROBE_CONFLICTS}",
            "-"
        ],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    elapsed = time.perf_counter() - start

    out = p.stdout.decode(errors="replace")

    # Une polarité immédiatement UNSAT est extrêmement informative.
    if p.returncode == 20:
        status = "UNSAT"
    elif p.returncode == 10:
        status = "SAT"
    else:
        status = "UNKNOWN"

    # Extraire quelques compteurs CaDiCaL si disponibles.
    conflicts = 0
    decisions = 0
    propagations = 0

    for line in out.splitlines():

        low = line.lower()

        if "conflicts:" in low:
            m = re.search(r"conflicts:\s*([0-9]+)", low)
            if m:
                conflicts = int(m.group(1))

        if "decisions:" in low:
            m = re.search(r"decisions:\s*([0-9]+)", low)
            if m:
                decisions = int(m.group(1))

        if "propagations:" in low:
            m = re.search(r"propagations:\s*([0-9]+)", low)
            if m:
                propagations = int(m.group(1))

    return (
        var,
        sign,
        status,
        elapsed,
        conflicts,
        decisions,
        propagations
    )


def score_pair(a, b):

    # a et b = résultats +var / -var

    immediate = (
        int(a[2] in ("SAT", "UNSAT"))
        + int(b[2] in ("SAT", "UNSAT"))
    )

    # Plus de propagation précoce pour peu de décisions
    # = variable potentiellement structurante.
    propagation_score = (
        a[6] + b[6]
    ) / max(1, a[5] + b[5])

    # On préfère aussi les sondes qui atteignent rapidement
    # leur conclusion/budget.
    time_score = 1.0 / max(
        1e-6,
        a[3] + b[3]
    )

    return (
        immediate * 1e12
        + propagation_score * 1e3
        + time_score
    )


def make_cube(mask, variables):
    return [
        v if mask & (1 << bit) else -v
        for bit, v in enumerate(variables)
    ]


def solve_cube(job):
    idx, cube = job

    data = branch_cnf(TEXT, NV, NC, cube)

    start = time.perf_counter()

    p = subprocess.run(
        [str(CADICAL), "-"],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    elapsed = time.perf_counter() - start
    out = p.stdout.decode(errors="replace")

    if p.returncode == 10:
        status = "SAT"
    elif p.returncode == 20:
        status = "UNSAT"
    else:
        status = "UNKNOWN"

    return idx, status, elapsed, cube, out


def validate_model(output):
    assignment = {}

    for line in output.splitlines():
        if not line.startswith("v "):
            continue

        for token in line[2:].split():
            lit = int(token)
            if lit:
                assignment[abs(lit)] = lit > 0

    if not assignment:
        return False

    clause = []

    for line in TEXT.splitlines():
        if not line or line[0] in "cp":
            continue

        for token in line.split():
            lit = int(token)

            if lit:
                clause.append(lit)
                continue

            if not clause:
                return False

            if not any(
                (
                    assignment.get(abs(x)) is True
                    if x > 0
                    else assignment.get(abs(x)) is False
                )
                for x in clause
            ):
                return False

            clause = []

    return not clause


def main():

    print("=" * 70, flush=True)
    print(" DELTA SAT V2 — STRUCTURAL PROBE", flush=True)
    print("=" * 70, flush=True)

    actual = sha256(TARGET)

    print(f"TARGET_SHA256={actual}", flush=True)

    if actual != EXPECTED_SHA:
        print("TARGET_INTEGRITY=FAILED", flush=True)
        return 2

    print("TARGET_INTEGRITY=PASSED", flush=True)
    print(f"VARIABLES={NV}", flush=True)
    print(f"CLAUSES={NC}", flush=True)
    print(f"WORKERS={WORKERS}", flush=True)
    print(f"PROBE_CONFLICT_BUDGET={PROBE_CONFLICTS}", flush=True)

    jobs = []

    for v in range(1, min(PROBE_VARS, NV) + 1):
        jobs.append((v, +1))
        jobs.append((v, -1))

    print(f"PROBES={len(jobs)}", flush=True)
    print("DELTA_PROBE=START", flush=True)

    probe_start = time.perf_counter()

    results = {}

    with mp.Pool(WORKERS) as pool:
        for r in pool.imap_unordered(
            probe_one,
            jobs,
            chunksize=1
        ):
            v = r[0]

            results.setdefault(v, {})[r[1]] = r

    probe_wall = time.perf_counter() - probe_start

    ranking = []

    for v in results:
        if +1 not in results[v] or -1 not in results[v]:
            continue

        score = score_pair(
            results[v][+1],
            results[v][-1]
        )

        ranking.append((score, v))

    ranking.sort(
        key=lambda x: (-x[0], x[1])
    )

    selected = [
        v for _, v in ranking[:CUBE_BITS]
    ]

    print(f"PROBE_WALL_SECONDS={probe_wall:.6f}", flush=True)

    print(
        "DELTA_V2_BRANCH_VARIABLES="
        + ",".join(map(str, selected)),
        flush=True
    )

    print()
    print("TOP16_STRUCTURAL_VARIABLES", flush=True)

    for score, v in ranking[:16]:
        p = results[v][+1]
        n = results[v][-1]

        print(
            f"VAR={v} "
            f"SCORE={score:.6f} "
            f"POS={p[2]} "
            f"NEG={n[2]} "
            f"POS_T={p[3]:.6f} "
            f"NEG_T={n[3]:.6f} "
            f"PROP={p[6]+n[6]} "
            f"DEC={p[5]+n[5]}",
            flush=True
        )

    cubes = [
        (i, make_cube(i, selected))
        for i in range(1 << CUBE_BITS)
    ]

    print()
    print("DELTA_PARALLEL_SOLVE=START", flush=True)

    solve_start = time.perf_counter()

    completed = 0
    unsat = 0
    solution = None

    pool = mp.Pool(
        WORKERS,
        maxtasksperchild=4
    )

    try:
        for result in pool.imap_unordered(
            solve_cube,
            cubes,
            chunksize=1
        ):
            idx, status, elapsed, cube, output = result

            completed += 1

            if status == "UNSAT":
                unsat += 1

            print(
                f"CUBE={idx:03d} "
                f"RESULT={status} "
                f"SECONDS={elapsed:.6f} "
                f"COMPLETED={completed}/256",
                flush=True
            )

            if status == "SAT":

                valid = validate_model(output)

                print(
                    "MODEL_VALIDATION="
                    + ("PASSED" if valid else "FAILED"),
                    flush=True
                )

                if valid:
                    solution = (
                        idx,
                        cube
                    )

                    pool.terminate()
                    break

    finally:

        if solution:
            pool.terminate()
        else:
            pool.close()

        pool.join()

    solve_wall = time.perf_counter() - solve_start

    print()
    print("=" * 70, flush=True)
    print(" DELTA V2 RESULT", flush=True)
    print("=" * 70, flush=True)

    print(f"PROBE_SECONDS={probe_wall:.6f}", flush=True)
    print(f"SOLVE_SECONDS={solve_wall:.6f}", flush=True)

    total = probe_wall + solve_wall

    print(f"TOTAL_SECONDS={total:.6f}", flush=True)
    print(f"CUBES_COMPLETED={completed}", flush=True)
    print(f"CUBES_UNSAT={unsat}", flush=True)

    if solution:
        idx, cube = solution

        print("DELTA_RESULT=SAT", flush=True)
        print(
            "DELTA_PROOF_STATUS=MODEL_VALIDATED",
            flush=True
        )
        print(f"SAT_CUBE={idx}", flush=True)

        print(
            "SAT_CUBE_LITERALS="
            + " ".join(map(str, cube)),
            flush=True
        )

    elif unsat == 256:
        print("DELTA_RESULT=UNSAT", flush=True)
        print(
            "DELTA_PROOF_STATUS=ALL_CUBES_CLOSED",
            flush=True
        )

    else:
        print("DELTA_RESULT=UNKNOWN", flush=True)

    print()
    print("DELTA_COPROCESSOR=ACTIVE", flush=True)
    print("DELTA_QPU_V1=DISABLED", flush=True)
    print("DELTA_V2_STRUCTURAL_PROBE=ACTIVE", flush=True)
    print("CPU_BRIDGE_EXECUTION=PHYSICAL_MEASURED", flush=True)
    print("PERSISTENT_CACHE=NO", flush=True)
    print("BACKGROUND_SERVICE=NO", flush=True)

    return 0


if __name__ == "__main__":
    mp.set_start_method("fork")
    raise SystemExit(main())
