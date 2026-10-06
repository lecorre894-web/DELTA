#!/usr/bin/env python3

import os
import re
import sys
import time
import hashlib
import subprocess
import multiprocessing as mp
from pathlib import Path

TARGET = Path("DELTA_TARGET.cnf").resolve()
CADICAL = Path.home() / "delta_sat_lab/cadical/build/cadical"
KISSAT  = Path.home() / "delta_sat_lab/kissat/build/kissat"

EXPECTED_SHA = \
"6be9bfe6b44454a061a2e93350f589643f8a16f5bca6babdaa50a2d3d7069361"

WORKERS = min(32, os.cpu_count() or 1)
BITS = 8

TEXT = TARGET.read_text(encoding="ascii")

m = re.search(r"^p cnf\s+(\d+)\s+(\d+)", TEXT, re.MULTILINE)
if not m:
    raise RuntimeError("DIMACS header missing")

NV = int(m.group(1))
NC = int(m.group(2))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def branch_cnf(literals):
    lines = TEXT.splitlines()

    for i, line in enumerate(lines):
        if line.startswith("p cnf "):
            lines[i] = f"p cnf {NV} {NC + len(literals)}"
            break

    units = "\n".join(f"{x} 0" for x in literals)

    return (
        "\n".join(lines) +
        "\n" +
        units +
        "\n"
    ).encode()


def cube(mask, variables):
    return [
        v if mask & (1 << bit) else -v
        for bit, v in enumerate(variables)
    ]


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


def validate_model(output):
    assignment = extract_model(output)

    if not assignment:
        return False

    clause = []

    for line in TEXT.splitlines():

        if not line or line[0] in "cp":
            continue

        for tok in line.split():
            lit = int(tok)

            if lit:
                clause.append(lit)
                continue

            if not clause:
                return False

            satisfied = False

            for x in clause:
                val = assignment.get(abs(x))

                if val is None:
                    continue

                if x > 0 and val:
                    satisfied = True
                    break

                if x < 0 and not val:
                    satisfied = True
                    break

            if not satisfied:
                return False

            clause = []

    return not clause


def solve_job(job):
    idx, literals = job

    start = time.perf_counter()
    cpu0 = time.process_time()

    p = subprocess.run(
        [str(CADICAL), "-"],
        input=branch_cnf(literals),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu0

    output = p.stdout.decode(errors="replace")

    if p.returncode == 10:
        status = "SAT"
    elif p.returncode == 20:
        status = "UNSAT"
    else:
        status = "UNKNOWN"

    return idx, status, wall, cpu, literals, output


def occurrence_ranking():
    freq = [0] * (NV + 1)

    for line in TEXT.splitlines():

        if not line or line[0] in "cp":
            continue

        for tok in line.split():
            lit = int(tok)
            if lit:
                freq[abs(lit)] += 1

    return sorted(
        range(1, NV + 1),
        key=lambda v: (-freq[v], v)
    )


def candidate_groups():
    ranked = occurrence_ranking()

    groups = []

    # Record V1
    groups.append(("V1_1_8", list(range(1, 9))))

    # Blocs naturels
    for start in (
        9, 17, 25, 33, 41, 49,
        57, 65, 97, 129, 193,
        257, 321, 385, 449,
        513, 577
    ):
        if start + 7 <= NV:
            groups.append(
                (
                    f"SEQ_{start}_{start+7}",
                    list(range(start, start + 8))
                )
            )

    # Fréquence V1 QPU : témoin négatif connu
    groups.append(
        ("FREQUENCY_TOP8", ranked[:8])
    )

    # Plusieurs fenêtres du classement de fréquence
    for offset in (8, 16, 32, 64, 128):
        if offset + 8 <= len(ranked):
            groups.append(
                (
                    f"FREQ_{offset}_{offset+7}",
                    ranked[offset:offset+8]
                )
            )

    # Répartition dans l'espace des variables
    step = max(1, NV // 8)
    spread = [
        min(NV, 1 + i * step)
        for i in range(8)
    ]

    if len(set(spread)) == 8:
        groups.append(("SPREAD", spread))

    # dédoublonnage
    seen = set()
    out = []

    for name, g in groups:
        key = tuple(g)

        if key not in seen:
            seen.add(key)
            out.append((name, g))

    return out


def pilot_group(name, variables):
    #
    # 8 cubes pilotes fixes.
    # Budget très court par cube : on mesure uniquement
    # la facilité de fermeture initiale.
    #
    masks = [0, 1, 2, 3, 7, 15, 31, 255]

    solved = 0
    unsat = 0
    sat = 0
    elapsed_sum = 0.0

    for mask in masks:
        literals = cube(mask, variables)

        start = time.perf_counter()

        try:
            p = subprocess.run(
                [
                    str(CADICAL),
                    "--conflicts=500",
                    "-"
                ],
                input=branch_cnf(literals),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=0.25
            )

            elapsed = time.perf_counter() - start

            if p.returncode in (10, 20):
                solved += 1

                if p.returncode == 10:
                    sat += 1
                else:
                    unsat += 1

                elapsed_sum += elapsed
            else:
                elapsed_sum += 0.25

        except subprocess.TimeoutExpired:
            elapsed_sum += 0.25

    #
    # Priorité absolue au nombre de cubes fermés.
    # À égalité : temps cumulé plus faible.
    #
    score = solved * 1000.0 - elapsed_sum

    return {
        "name": name,
        "vars": variables,
        "solved": solved,
        "unsat": unsat,
        "sat": sat,
        "time": elapsed_sum,
        "score": score
    }


def select_v3():
    print("V3_PILOT_SELECTION=START", flush=True)

    t0 = time.perf_counter()

    results = []

    for name, variables in candidate_groups():
        r = pilot_group(name, variables)
        results.append(r)

        print(
            f"PILOT={name} "
            f"VARS={','.join(map(str, variables))} "
            f"SOLVED={r['solved']}/8 "
            f"UNSAT={r['unsat']} "
            f"SAT={r['sat']} "
            f"PILOT_TIME={r['time']:.6f} "
            f"SCORE={r['score']:.6f}",
            flush=True
        )

    results.sort(
        key=lambda r: (
            -r["solved"],
            r["time"],
            r["name"]
        )
    )

    best = results[0]

    elapsed = time.perf_counter() - t0

    print(
        "V3_SELECTED="
        + best["name"],
        flush=True
    )

    print(
        "V3_BRANCH_VARIABLES="
        + ",".join(map(str, best["vars"])),
        flush=True
    )

    print(
        f"V3_SELECTION_SECONDS={elapsed:.6f}",
        flush=True
    )

    return best["vars"], elapsed


def run_delta(label, variables, selection_seconds=0.0):

    print()
    print("=" * 70, flush=True)
    print(f" {label}", flush=True)
    print("=" * 70, flush=True)

    print(
        "BRANCH_VARIABLES="
        + ",".join(map(str, variables)),
        flush=True
    )

    jobs = [
        (i, cube(i, variables))
        for i in range(1 << BITS)
    ]

    t0 = time.perf_counter()

    completed = 0
    unsat = 0
    result = "UNKNOWN"
    proof = "INCOMPLETE"

    pool = mp.Pool(
        WORKERS,
        maxtasksperchild=4
    )

    try:
        for r in pool.imap_unordered(
            solve_job,
            jobs,
            chunksize=1
        ):

            idx, status, wall, cpu, literals, output = r
            completed += 1

            if status == "UNSAT":
                unsat += 1

            print(
                f"{label} "
                f"CUBE={idx:03d} "
                f"RESULT={status} "
                f"SECONDS={wall:.6f} "
                f"COMPLETED={completed}/256",
                flush=True
            )

            if status == "SAT":

                valid = validate_model(output)

                print(
                    f"{label}_MODEL_VALIDATION="
                    + ("PASSED" if valid else "FAILED"),
                    flush=True
                )

                if valid:
                    result = "SAT"
                    proof = "MODEL_VALIDATED"

                    print(
                        f"{label}_SAT_CUBE={idx}",
                        flush=True
                    )

                    print(
                        f"{label}_SAT_CUBE_LITERALS="
                        + " ".join(map(str, literals)),
                        flush=True
                    )

                    pool.terminate()
                    break

    finally:
        if result == "SAT":
            pool.terminate()
        else:
            pool.close()

        pool.join()

    solve_wall = time.perf_counter() - t0
    total = selection_seconds + solve_wall

    if result != "SAT" and unsat == 256:
        result = "UNSAT"
        proof = "ALL_CUBES_CLOSED"

    print(f"{label}_SOLVE_SECONDS={solve_wall:.6f}", flush=True)
    print(f"{label}_TOTAL_SECONDS={total:.6f}", flush=True)
    print(f"{label}_CUBES_COMPLETED={completed}", flush=True)
    print(f"{label}_CUBES_UNSAT={unsat}", flush=True)
    print(f"{label}_RESULT={result}", flush=True)
    print(f"{label}_PROOF_STATUS={proof}", flush=True)


def main():

    print("=" * 70)
    print(" DELTA V3 SAT ARENA")
    print("=" * 70)

    actual = sha256(TARGET)

    print(f"TARGET_SHA256={actual}")

    if actual != EXPECTED_SHA:
        print("TARGET_INTEGRITY=FAILED")
        return 2

    print("TARGET_INTEGRITY=PASSED")
    print(f"VARIABLES={NV}")
    print(f"CLAUSES={NC}")
    print(f"WORKERS={WORKERS}")

    mode = sys.argv[1]

    if mode == "V1":
        run_delta(
            "DELTA_V1",
            list(range(1, 9))
        )

    elif mode == "V3":
        variables, selection_time = select_v3()

        run_delta(
            "DELTA_V3",
            variables,
            selection_time
        )

    else:
        raise RuntimeError("mode must be V1 or V3")

    return 0


if __name__ == "__main__":
    mp.set_start_method("fork")
    raise SystemExit(main())
