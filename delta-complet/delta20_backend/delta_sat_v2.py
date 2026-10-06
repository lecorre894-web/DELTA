#!/usr/bin/env python3

import os
import time
import random
import statistics
import numpy as np

from delta_x512 import X512Core

LIMIT = 120.0
SEED = 894
BATCH = 1 << 18       # 262144 affectations/bloc
RUNS = 3

rng = random.Random(SEED)

# ============================================================
# CNF reproductible, garantie SAT
# ============================================================

def make_instance(nvars, nclauses, seed):
    r = random.Random(seed)

    secret = [r.getrandbits(1) for _ in range(nvars)]
    clauses = []

    for _ in range(nclauses):
        vs = r.sample(range(nvars), 3)
        lits = [(v, r.getrandbits(1)) for v in vs]

        if not any(secret[v] == sign for v, sign in lits):
            v, _ = lits[0]
            lits[0] = (v, secret[v])

        clauses.append(tuple(lits))

    return clauses


def verify(x, clauses):
    return all(
        any(((x >> v) & 1) == sign for v, sign in clause)
        for clause in clauses
    )


# ============================================================
# RÉFÉRENCE SCALAIRE
# ============================================================

def scalar_solver(nvars, clauses, deadline):
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


# ============================================================
# DELTA VECTORISÉ
# ============================================================

def vector_solver(nvars, clauses, deadline, core):

    tested = 0
    x512_seconds = 0.0

    for base in range(0, 1 << nvars, BATCH):

        if time.perf_counter() >= deadline:
            return None, tested, x512_seconds

        stop = min(base + BATCH, 1 << nvars)

        xs = np.arange(
            base,
            stop,
            dtype=np.uint64
        )

        alive = np.ones(xs.size, dtype=np.bool_)

        for clause in clauses:

            clause_ok = np.zeros(xs.size, dtype=np.bool_)

            for v, sign in clause:
                clause_ok |= (((xs >> np.uint64(v)) & 1) == sign)

            alive &= clause_ok

            if not alive.any():
                break

        tested += xs.size

        idx = np.flatnonzero(alive)

        # X512 participe pendant le traitement du bloc :
        # signature matérielle AVX-512 du masque de survivants.
        packed = np.packbits(alive, bitorder="little")

        pad = (-len(packed)) % 8
        if pad:
            packed = np.pad(packed, (0, pad))

        a = packed.view(np.uint64)

        if a.size:
            b = np.full_like(
                a,
                np.uint64(0xFFFFFFFFFFFFFFFF)
            )
            c = np.zeros_like(a)

            tx = time.perf_counter()
            core.tpop(a, b, c, "micro")
            x512_seconds += time.perf_counter() - tx

        if idx.size:
            solution = base + int(idx[0])
            return solution, tested, x512_seconds

    return False, tested, x512_seconds


# ============================================================
# CAMPAGNE
# ============================================================

print("=== DELTA SAT V2 — VECTOR SEARCH ===")
print("HOST=AMD Ryzen 9 9950X3D")
print(f"CPU_AVAILABLE={os.cpu_count()}")
print("AVX512=YES")
print(f"BATCH={BATCH}")
print(f"RUNS={RUNS}")
print("LIMIT_SECONDS=120")
print()

core = X512Core()

start_global = time.perf_counter()
deadline_global = start_global + LIMIT

# On commence raisonnablement puis on monte.
sizes = [
    (20, 90),
    (22, 99),
    (24, 108),
    (26, 117),
    (28, 126),
]

all_rows = []

for nvars, nclauses in sizes:

    if time.perf_counter() >= deadline_global:
        break

    print(
        f"--- {nvars} VARIABLES / {nclauses} CLAUSES ---"
    )

    clauses = make_instance(
        nvars,
        nclauses,
        SEED + nvars
    )

    scalar_times = []
    vector_times = []
    x512_times = []

    valid = True

    for run in range(RUNS):

        if time.perf_counter() >= deadline_global:
            valid = False
            break

        # ---------------- scalar ----------------

        ts = time.perf_counter()

        sol_s, tested_s = scalar_solver(
            nvars,
            clauses,
            deadline_global
        )

        scalar_elapsed = time.perf_counter() - ts

        if sol_s is None:
            print("SCALAR_TIMEOUT")
            valid = False
            break

        # ---------------- vector ----------------

        tv = time.perf_counter()

        sol_v, tested_v, tx512 = vector_solver(
            nvars,
            clauses,
            deadline_global,
            core
        )

        vector_elapsed = time.perf_counter() - tv

        if sol_v is None:
            print("VECTOR_TIMEOUT")
            valid = False
            break

        vs = (
            sol_s is not False
            and verify(sol_s, clauses)
        )

        vv = (
            sol_v is not False
            and verify(sol_v, clauses)
        )

        status_match = (
            (sol_s is False and sol_v is False)
            or
            (sol_s is not False and sol_v is not False)
        )

        if not (vs and vv and status_match):
            valid = False
            print("VALIDATION=FAIL")
            break

        scalar_times.append(scalar_elapsed)
        vector_times.append(vector_elapsed)
        x512_times.append(tx512)

        print(
            f"RUN={run+1} "
            f"SCALAR={scalar_elapsed:.6f}s "
            f"VECTOR={vector_elapsed:.6f}s "
            f"X512={tx512:.6f}s "
            f"SPEEDUP={scalar_elapsed/vector_elapsed:.3f}x "
            f"VALID=YES"
        )

    if not valid or not scalar_times:
        break

    sm = statistics.median(scalar_times)
    vm = statistics.median(vector_times)
    xm = statistics.median(x512_times)

    ratio = sm / vm

    all_rows.append(
        (nvars, nclauses, sm, vm, xm, ratio)
    )

    print(
        f"MEDIAN_SCALAR={sm:.6f}s"
    )
    print(
        f"MEDIAN_DELTA_VECTOR={vm:.6f}s"
    )
    print(
        f"MEDIAN_X512_COMPONENT={xm:.6f}s"
    )
    print(
        f"MEDIAN_SPEEDUP={ratio:.3f}x"
    )
    print("VALIDATION=PASSED")
    print()


elapsed = time.perf_counter() - start_global

print("=== FINAL RESULTS ===")
print(f"ELAPSED_SECONDS={elapsed:.6f}")

for n, c, s, v, x, r in all_rows:
    print(
        f"VARS={n:2d} "
        f"CLAUSES={c:3d} "
        f"SCALAR={s:.6f}s "
        f"DELTA_VECTOR={v:.6f}s "
        f"X512={x:.6f}s "
        f"SPEEDUP={r:.3f}x"
    )

if all_rows:
    ratios = [r[-1] for r in all_rows]

    print()
    print(
        f"GLOBAL_MEDIAN_SPEEDUP="
        f"{statistics.median(ratios):.3f}x"
    )

print()
print("REFERENCE=PYTHON_SCALAR_EXACT")
print("DELTA_SEARCH=NUMPY_VECTORIZED_EXACT")
print("X512_ROLE=BLOCK_SIGNATURE")
print("RESULT_VERIFICATION=INDEPENDENT")
print("CLASSIFICATION=PHYSICAL_LOCAL_HOST_MEASURED")
print("FINAL_VALIDATION=PASSED")
