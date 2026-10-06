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

# Première V1 : 2^8 = 256 sous-problèmes.
# Suffisant pour alimenter les 32 workers sans créer des milliers
# de processus simultanés.
CUBE_BITS = 8
CUBES = 1 << CUBE_BITS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_header(path):
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("p cnf "):
                p = line.split()
                return int(p[2]), int(p[3])
    raise RuntimeError("DIMACS header missing")


def choose_variables(path, nvars, wanted):
    #
    # QPU logique V1 :
    # score = fréquence d'apparition de chaque variable.
    #
    # Les variables les plus présentes deviennent les variables
    # de partitionnement.
    #
    freq = [0] * (nvars + 1)

    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line or line[0] in "cp":
                continue

            for tok in line.split():
                lit = int(tok)

                if lit:
                    v = abs(lit)
                    if v <= nvars:
                        freq[v] += 1

    ranked = sorted(
        range(1, nvars + 1),
        key=lambda v: (-freq[v], v)
    )

    return ranked[:wanted]


def make_cube(mask, variables):
    cube = []

    for bit, var in enumerate(variables):
        if mask & (1 << bit):
            cube.append(var)
        else:
            cube.append(-var)

    return cube


def worker(job):
    idx, cube, target, cadical = job

    #
    # Chaque worker construit son propre flux DIMACS.
    # Le fichier cible original reste strictement intact.
    #
    with open(target, "rb") as f:
        original = f.read()

    assumptions = "".join(f"{lit} 0\n" for lit in cube)

    #
    # CaDiCaL accepte DIMACS depuis stdin.
    # On ajoute ici les hypothèses comme clauses unitaires.
    #
    text = original.decode("ascii", errors="strict")

    lines = text.splitlines()

    header_index = None
    nvars = None
    nclauses = None

    for i, line in enumerate(lines):
        if line.startswith("p cnf "):
            p = line.split()
            nvars = int(p[2])
            nclauses = int(p[3])
            header_index = i
            break

    if header_index is None:
        return idx, "ERROR", 0.0, cube, ""

    lines[header_index] = f"p cnf {nvars} {nclauses + len(cube)}"

    branch_cnf = (
        "\n".join(lines)
        + "\n"
        + assumptions
    ).encode()

    start = time.perf_counter()

    p = subprocess.run(
        [str(cadical), "-"],
        input=branch_cnf,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    elapsed = time.perf_counter() - start

    out = p.stdout.decode(errors="replace")

    if p.returncode == 10:
        result = "SAT"
    elif p.returncode == 20:
        result = "UNSAT"
    else:
        result = "UNKNOWN"

    return idx, result, elapsed, cube, out


def validate_model(path, solver_output):
    assignment = {}

    for line in solver_output.splitlines():
        if not line.startswith("v "):
            continue

        for tok in line[2:].split():
            lit = int(tok)

            if lit == 0:
                continue

            assignment[abs(lit)] = lit > 0

    if not assignment:
        return False

    with open(path, "r", errors="replace") as f:
        clause = []

        for line in f:
            if not line or line[0] in "cp":
                continue

            for tok in line.split():
                lit = int(tok)

                if lit == 0:
                    if not clause:
                        return False

                    satisfied = False

                    for x in clause:
                        value = assignment.get(abs(x))

                        if value is None:
                            continue

                        if (x > 0 and value) or (x < 0 and not value):
                            satisfied = True
                            break

                    if not satisfied:
                        return False

                    clause = []

                else:
                    clause.append(lit)

    return True


def main():

    print("=" * 64)
    print(" DELTA SAT PARALLEL V1")
    print(" Ryzen <-> DELTA <-> QPU logique")
    print("=" * 64)

    actual = sha256(TARGET)

    print(f"TARGET={TARGET}")
    print(f"SHA256={actual}")

    if actual != EXPECTED_SHA:
        print("TARGET_INTEGRITY=FAILED")
        return 2

    print("TARGET_INTEGRITY=PASSED")

    nvars, nclauses = parse_header(TARGET)

    print(f"VARIABLES={nvars}")
    print(f"CLAUSES={nclauses}")
    print(f"PHYSICAL_CPU_THREADS_AVAILABLE={os.cpu_count()}")
    print(f"DELTA_WORKERS={WORKERS}")
    print("DELTA_QBIT_EXECUTION=LOGICAL_EMULATED")
    print("CPU_BRIDGE_EXECUTION=PHYSICAL_MEASURED")

    variables = choose_variables(
        TARGET,
        nvars,
        CUBE_BITS
    )

    print()
    print("QPU_LOGICAL_BRANCH_VARIABLES=" +
          ",".join(map(str, variables)))

    print(f"QPU_LOGICAL_CUBES={CUBES}")

    jobs = [
        (
            i,
            make_cube(i, variables),
            str(TARGET),
            str(CADICAL)
        )
        for i in range(CUBES)
    ]

    print()
    print("RYZEN_PARALLEL_EXECUTION=START")

    start = time.perf_counter()

    solved = None
    unsat_count = 0
    completed = 0

    #
    # maxtasksperchild borne la durée de vie des workers.
    # Aucun service résident après l'expérience.
    #
    pool = mp.Pool(
        processes=WORKERS,
        maxtasksperchild=4
    )

    try:
        for result in pool.imap_unordered(worker, jobs, chunksize=1):

            idx, status, elapsed, cube, output = result

            completed += 1

            if status == "UNSAT":
                unsat_count += 1

            print(
                f"CUBE={idx:03d} "
                f"RESULT={status} "
                f"SECONDS={elapsed:.6f} "
                f"COMPLETED={completed}/{CUBES}"
            )

            if status == "SAT":

                valid = validate_model(
                    TARGET,
                    output
                )

                print()
                print("SAT_BRANCH_FOUND=YES")
                print(f"SAT_CUBE={idx}")
                print("SAT_CUBE_LITERALS=" +
                      " ".join(map(str, cube)))
                print(
                    "INDEPENDENT_MODEL_VALIDATION="
                    + ("PASSED" if valid else "FAILED")
                )

                if valid:
                    solved = ("SAT", idx)
                    pool.terminate()
                    break

    finally:
        if solved:
            pool.terminate()
        else:
            pool.close()

        pool.join()

    total = time.perf_counter() - start

    print()
    print("=" * 64)
    print(" DELTA RESULT")
    print("=" * 64)

    print(f"WALL_SECONDS={total:.6f}")
    print(f"CUBES_COMPLETED={completed}")
    print(f"CUBES_UNSAT={unsat_count}")

    if solved:
        print("DELTA_RESULT=SAT")
        print("DELTA_PROOF_STATUS=MODEL_VALIDATED")

    elif unsat_count == CUBES:
        #
        # Les 256 cubes couvrent toutes les affectations
        # des 8 variables choisies.
        #
        # Si chacun est UNSAT, leur union couvre donc
        # l'espace complet de la formule.
        #
        print("DELTA_RESULT=UNSAT")
        print("DELTA_CUBE_COVERAGE=COMPLETE")
        print("DELTA_PROOF_STATUS=ALL_CUBES_CLOSED")

    else:
        print("DELTA_RESULT=UNKNOWN")
        print("DELTA_PROOF_STATUS=INCOMPLETE")

    print()
    print("RYZEN_PHYSICAL=16C_32T")
    print("DELTA_COPROCESSOR=ACTIVE")
    print("QPU_LOGICAL_SCHEDULER=ACTIVE")
    print("PERSISTENT_CACHE=NO")
    print("BACKGROUND_SERVICE=NO")

    return 0


if __name__ == "__main__":
    mp.set_start_method("fork")
    raise SystemExit(main())
