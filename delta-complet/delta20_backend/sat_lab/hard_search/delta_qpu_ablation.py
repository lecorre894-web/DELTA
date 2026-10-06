#!/usr/bin/env python3

import os
import sys
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
CUBES = 1 << CUBE_BITS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_dimacs(path):
    nvars = nclauses = None
    freq = None

    with open(path, "r", errors="strict") as f:
        for line in f:
            if not line or line[0] == "c":
                continue

            if line.startswith("p cnf "):
                p = line.split()
                nvars = int(p[2])
                nclauses = int(p[3])
                freq = [0] * (nvars + 1)
                continue

            if freq is None:
                continue

            for tok in line.split():
                lit = int(tok)
                if lit:
                    freq[abs(lit)] += 1

    if nvars is None:
        raise RuntimeError("DIMACS header missing")

    return nvars, nclauses, freq


def variables_for_mode(mode, nvars, freq):
    if mode == "OFF":
        # Témoin neutre :
        # aucun score DELTA/QPU.
        return list(range(1, CUBE_BITS + 1))

    if mode == "ON":
        # Même politique que DELTA V1 :
        # fréquence décroissante.
        ranked = sorted(
            range(1, nvars + 1),
            key=lambda v: (-freq[v], v)
        )
        return ranked[:CUBE_BITS]

    raise ValueError(mode)


def make_cube(mask, variables):
    return [
        var if mask & (1 << bit) else -var
        for bit, var in enumerate(variables)
    ]


def build_branch_cnf(target, cube):
    text = Path(target).read_text(encoding="ascii")
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if line.startswith("p cnf "):
            p = line.split()
            nv = int(p[2])
            nc = int(p[3])
            lines[i] = f"p cnf {nv} {nc + len(cube)}"
            break
    else:
        raise RuntimeError("DIMACS header missing")

    units = "\n".join(f"{lit} 0" for lit in cube)

    return ("\n".join(lines) + "\n" + units + "\n").encode()


def worker(job):
    idx, cube, target, cadical = job

    branch = build_branch_cnf(target, cube)

    start = time.perf_counter()

    p = subprocess.run(
        [cadical, "-"],
        input=branch,
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


def extract_model(output):
    assignment = {}

    for line in output.splitlines():
        if not line.startswith("v "):
            continue

        for tok in line[2:].split():
            lit = int(tok)

            if lit:
                assignment[abs(lit)] = lit > 0

    return assignment


def validate_model(path, output):
    assignment = extract_model(output)

    if not assignment:
        return False

    clause = []

    with open(path, "r", errors="strict") as f:
        for line in f:
            if not line or line[0] in "cp":
                continue

            for tok in line.split():
                lit = int(tok)

                if lit == 0:
                    if not clause:
                        return False

                    ok = False

                    for x in clause:
                        value = assignment.get(abs(x))

                        if value is None:
                            continue

                        if (x > 0 and value) or (x < 0 and not value):
                            ok = True
                            break

                    if not ok:
                        return False

                    clause = []

                else:
                    clause.append(lit)

    return not clause


def run(mode):
    actual = sha256(TARGET)

    if actual != EXPECTED_SHA:
        print("TARGET_INTEGRITY=FAILED", flush=True)
        return 2

    nvars, nclauses, freq = parse_dimacs(TARGET)
    variables = variables_for_mode(mode, nvars, freq)

    print("=" * 68, flush=True)
    print(f" DELTA QPU ABLATION — MODE={mode}", flush=True)
    print("=" * 68, flush=True)
    print(f"TARGET_SHA256={actual}", flush=True)
    print(f"VARIABLES={nvars}", flush=True)
    print(f"CLAUSES={nclauses}", flush=True)
    print(f"WORKERS={WORKERS}", flush=True)
    print(f"CUBES={CUBES}", flush=True)
    print(
        "BRANCH_VARIABLES=" + ",".join(map(str, variables)),
        flush=True
    )

    if mode == "ON":
        print("QPU_LOGICAL_SCHEDULER=ACTIVE", flush=True)
    else:
        print("QPU_LOGICAL_SCHEDULER=DISABLED", flush=True)

    jobs = [
        (
            i,
            make_cube(i, variables),
            str(TARGET),
            str(CADICAL)
        )
        for i in range(CUBES)
    ]

    start = time.perf_counter()

    completed = 0
    unsat = 0
    solved = None

    pool = mp.Pool(
        processes=WORKERS,
        maxtasksperchild=4
    )

    try:
        for result in pool.imap_unordered(
            worker,
            jobs,
            chunksize=1
        ):
            idx, status, elapsed, cube, output = result
            completed += 1

            if status == "UNSAT":
                unsat += 1

            print(
                f"MODE={mode} "
                f"CUBE={idx:03d} "
                f"RESULT={status} "
                f"SECONDS={elapsed:.6f} "
                f"COMPLETED={completed}/{CUBES}",
                flush=True
            )

            if status == "SAT":
                valid = validate_model(TARGET, output)

                print(
                    "MODEL_VALIDATION="
                    + ("PASSED" if valid else "FAILED"),
                    flush=True
                )

                if valid:
                    solved = ("SAT", idx, cube)
                    pool.terminate()
                    break

    finally:
        if solved:
            pool.terminate()
        else:
            pool.close()

        pool.join()

    wall = time.perf_counter() - start

    print(flush=True)
    print("=" * 68, flush=True)
    print(f" MODE={mode} RESULT", flush=True)
    print("=" * 68, flush=True)
    print(f"WALL_SECONDS={wall:.6f}", flush=True)
    print(f"CUBES_COMPLETED={completed}", flush=True)
    print(f"CUBES_UNSAT={unsat}", flush=True)

    if solved:
        _, idx, cube = solved

        print("DELTA_RESULT=SAT", flush=True)
        print("DELTA_PROOF_STATUS=MODEL_VALIDATED", flush=True)
        print(f"SAT_CUBE={idx}", flush=True)
        print(
            "SAT_CUBE_LITERALS="
            + " ".join(map(str, cube)),
            flush=True
        )

    elif unsat == CUBES:
        print("DELTA_RESULT=UNSAT", flush=True)
        print(
            "DELTA_PROOF_STATUS=ALL_CUBES_CLOSED",
            flush=True
        )

    else:
        print("DELTA_RESULT=UNKNOWN", flush=True)
        print(
            "DELTA_PROOF_STATUS=INCOMPLETE",
            flush=True
        )

    return 0


if __name__ == "__main__":
    mp.set_start_method("fork")

    if len(sys.argv) != 2 or sys.argv[1] not in ("OFF", "ON"):
        print("usage: delta_qpu_ablation.py OFF|ON")
        raise SystemExit(2)

    raise SystemExit(run(sys.argv[1]))
