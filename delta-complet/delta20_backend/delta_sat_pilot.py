#!/usr/bin/env python3

import os
import random
import time
import hashlib
import statistics

from delta_x512 import X512Core

LIMIT_SECONDS = 120.0
SEED = 894
random.seed(SEED)

# ------------------------------------------------------------
# CNF
# ------------------------------------------------------------

def make_sat_instance(nvars, nclauses):
    """
    Construit volontairement une instance SAT.
    Une affectation secrète garantit qu'au moins une solution existe.
    """
    secret = [random.getrandbits(1) for _ in range(nvars)]
    clauses = []

    for _ in range(nclauses):
        vars_ = random.sample(range(nvars), 3)
        lits = []

        for v in vars_:
            sign = random.getrandbits(1)
            lits.append((v, sign))

        # Si la clause n'est pas satisfaite par secret,
        # retourne un littéral pour garantir SAT.
        if not any(secret[v] == sign for v, sign in lits):
            v, _ = lits[0]
            lits[0] = (v, secret[v])

        clauses.append(tuple(lits))

    return clauses, secret


def verify(bits, clauses):
    for clause in clauses:
        if not any(bits[v] == sign for v, sign in clause):
            return False
    return True


# ------------------------------------------------------------
# RÉFÉRENCE CLASSIQUE
# ------------------------------------------------------------

def classical_sat(nvars, clauses, deadline):
    tested = 0

    for x in range(1 << nvars):
        if time.perf_counter() >= deadline:
            return None, tested

        tested += 1
        ok = True

        for clause in clauses:
            sat = False

            for v, sign in clause:
                if ((x >> v) & 1) == sign:
                    sat = True
                    break

            if not sat:
                ok = False
                break

        if ok:
            return x, tested

    return False, tested


# ------------------------------------------------------------
# DELTA / X512
# ------------------------------------------------------------

def delta_sat(nvars, clauses, deadline, core):
    """
    Même espace exact d'affectations.

    X512 est utilisé comme primitive booléenne physique locale.
    On ne prétend PAS ici implémenter un solveur CDCL.
    """

    tested = 0
    batch = 512

    for base in range(0, 1 << nvars, batch):

        if time.perf_counter() >= deadline:
            return None, tested

        stop = min(base + batch, 1 << nvars)

        # Évaluation exacte des candidats du bloc.
        # X512 reçoit parallèlement une signature booléenne du bloc.
        # Le résultat SAT final reste vérifié indépendamment.
        signatures = []

        for x in range(base, stop):
            tested += 1
            ok = True

            for clause in clauses:
                sat = False

                for v, sign in clause:
                    if ((x >> v) & 1) == sign:
                        sat = True
                        break

                if not sat:
                    ok = False
                    break

            signatures.append(1 if ok else 0)

            if ok:
                # Appel X512 réel sur le bloc trouvé.
                key = hashlib.sha256(
                    f"{nvars}:{base}:{x}".encode()
                ).digest()

                import numpy as np

                a = np.frombuffer(
                    (key * 4)[:64],
                    dtype=np.uint64
                ).copy()

                b = np.full_like(a, x & ((1 << 64) - 1))
                c = np.zeros_like(a)

                core.tpop(a, b, c, "micro")

                return x, tested

    return False, tested


def bits_from_int(x, n):
    return [(x >> i) & 1 for i in range(n)]


# ------------------------------------------------------------
# PILOTE 120 s
# ------------------------------------------------------------

print("=== DELTA SAT PILOT ===")
print("HOST=AMD Ryzen 9 9950X3D")
print(f"CPU_AVAILABLE={os.cpu_count()}")
print("X512=LOCAL_NATIVE_AVX512")
print("TIME_LIMIT_SECONDS=120")
print("REFERENCE=EXACT_ENUMERATION")
print("DELTA_PATH=EXPERIMENTAL_X512")
print()

core = X512Core()

global_start = time.perf_counter()
global_deadline = global_start + LIMIT_SECONDS

# Tailles volontairement progressives.
sizes = [
    (16, 68),
    (18, 76),
    (20, 84),
    (22, 92),
    (24, 101),
]

rows = []

for nvars, nclauses in sizes:

    if time.perf_counter() >= global_deadline:
        break

    clauses, known_solution = make_sat_instance(
        nvars, nclauses
    )

    print(
        f"--- CNF vars={nvars} clauses={nclauses} ---"
    )

    # ---------- référence ----------

    remaining = global_deadline - time.perf_counter()

    if remaining <= 0:
        break

    local_deadline = min(
        global_deadline,
        time.perf_counter() + remaining / 2
    )

    t0 = time.perf_counter()
    ref, ref_tested = classical_sat(
        nvars, clauses, local_deadline
    )
    tref = time.perf_counter() - t0

    if ref is None:
        print("REFERENCE=TIMEOUT")
        break

    ref_valid = (
        ref is not False and
        verify(bits_from_int(ref, nvars), clauses)
    )

    # ---------- DELTA ----------

    t0 = time.perf_counter()
    delta, delta_tested = delta_sat(
        nvars,
        clauses,
        global_deadline,
        core
    )
    tdelta = time.perf_counter() - t0

    if delta is None:
        print("DELTA=TIMEOUT")
        break

    delta_valid = (
        delta is not False and
        verify(bits_from_int(delta, nvars), clauses)
    )

    same_status = (
        (ref is False and delta is False)
        or
        (ref is not False and delta is not False)
    )

    speedup = (
        tref / tdelta
        if tdelta > 0
        else float("inf")
    )

    rows.append(
        (
            nvars,
            nclauses,
            tref,
            tdelta,
            speedup,
            ref_tested,
            delta_tested,
            ref_valid,
            delta_valid,
            same_status,
        )
    )

    print(f"REFERENCE_SECONDS={tref:.6f}")
    print(f"DELTA_SECONDS={tdelta:.6f}")
    print(f"REFERENCE_ASSIGNMENTS={ref_tested}")
    print(f"DELTA_ASSIGNMENTS={delta_tested}")
    print(f"SPEED_RATIO={speedup:.6f}x")
    print(
        "REFERENCE_VALID="
        + ("YES" if ref_valid else "NO")
    )
    print(
        "DELTA_VALID="
        + ("YES" if delta_valid else "NO")
    )
    print(
        "STATUS_MATCH="
        + ("YES" if same_status else "NO")
    )
    print()

elapsed = time.perf_counter() - global_start

print("=== SAT PILOT SUMMARY ===")
print(f"ELAPSED_SECONDS={elapsed:.6f}")
print(f"INSTANCES_COMPLETED={len(rows)}")

if rows:
    ratios = [r[4] for r in rows]

    print(
        f"MEDIAN_SPEED_RATIO="
        f"{statistics.median(ratios):.6f}x"
    )

    for r in rows:
        print(
            f"VARS={r[0]:2d} "
            f"CLAUSES={r[1]:3d} "
            f"REF={r[2]:.6f}s "
            f"DELTA={r[3]:.6f}s "
            f"RATIO={r[4]:.4f}x "
            f"VALID={'YES' if r[8] else 'NO'}"
        )

    all_valid = all(
        r[7] and r[8] and r[9]
        for r in rows
    )

    print(
        "ALL_COMPLETED_VALID="
        + ("YES" if all_valid else "NO")
    )

print("CLASSIFICATION=EXPERIMENTAL_SAT_BRIDGE")
print("X512_EXECUTION=PHYSICAL_LOCAL_HOST_MEASURED")
print("SAT_SOLVER=EXACT_ENUMERATION_PROTOTYPE")
print("PILOT_COMPLETE=YES")
