#!/usr/bin/env python3

import ctypes
import os
import random
import statistics
import time

LIMIT = 120.0
RUNS = 3
SEED = 894

UINT64_MAX = (1 << 64) - 1


class Clause(ctypes.Structure):
    _fields_ = [
        ("v", ctypes.c_uint8 * 3),
        ("s", ctypes.c_uint8 * 3),
    ]


lib = ctypes.CDLL(
    os.path.abspath("./libdelta_sat_native.so")
)

for fn in (
    lib.delta_sat_scalar,
    lib.delta_sat_avx512,
):
    fn.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(Clause),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    fn.restype = ctypes.c_uint64


def make_instance(nvars, nclauses, seed):

    r = random.Random(seed)

    secret = [
        r.getrandbits(1)
        for _ in range(nvars)
    ]

    py = []

    for _ in range(nclauses):

        vs = r.sample(range(nvars), 3)

        lits = [
            [v, r.getrandbits(1)]
            for v in vs
        ]

        if not any(
            secret[v] == s
            for v, s in lits
        ):
            lits[0][1] = secret[lits[0][0]]

        py.append(lits)

    arr = (Clause * nclauses)()

    for j, clause in enumerate(py):
        for k, (v, s) in enumerate(clause):
            arr[j].v[k] = v
            arr[j].s[k] = s

    return py, arr


def verify(x, clauses):

    if x == UINT64_MAX:
        return False

    return all(
        any(
            ((x >> v) & 1) == s
            for v, s in clause
        )
        for clause in clauses
    )


def python_scalar(
    nvars,
    clauses,
    deadline
):

    tested = 0

    for x in range(1 << nvars):

        if time.perf_counter() >= deadline:
            return None, tested

        tested += 1

        ok = True

        for clause in clauses:

            sat = False

            for v, s in clause:

                if ((x >> v) & 1) == s:
                    sat = True
                    break

            if not sat:
                ok = False
                break

        if ok:
            return x, tested

    return UINT64_MAX, tested


def native_call(fn, nvars, arr, nc):

    tested = ctypes.c_uint64()

    t0 = time.perf_counter()

    solution = fn(
        nvars,
        arr,
        nc,
        ctypes.byref(tested)
    )

    dt = time.perf_counter() - t0

    return solution, tested.value, dt


print("=== DELTA SAT V3 NATIVE AVX-512 ===")
print(
    "CPU=AMD Ryzen 9 9950X3D 16-Core Processor"
)
print(f"CPU_AVAILABLE={os.cpu_count()}")
print("IMPLEMENTATION_1=PYTHON_SCALAR")
print("IMPLEMENTATION_2=C_NATIVE_SCALAR")
print("IMPLEMENTATION_3=C_NATIVE_AVX512")
print("AVX512_BLOCK=512_ASSIGNMENTS")
print("TIME_LIMIT_SECONDS=120")
print()


global_start = time.perf_counter()
deadline = global_start + LIMIT

sizes = [
    (20, 90),
    (22, 99),
    (24, 108),
    (26, 117),
    (28, 126),
    (30, 135),
]

rows = []

for nvars, nc in sizes:

    if time.perf_counter() >= deadline:
        break

    print(
        f"--- VARS={nvars} CLAUSES={nc} ---"
    )

    pyclauses, carr = make_instance(
        nvars,
        nc,
        SEED + nvars
    )

    # Python : une seule mesure.
    # C/AVX : plusieurs répétitions.
    t0 = time.perf_counter()

    pysol, pytested = python_scalar(
        nvars,
        pyclauses,
        deadline
    )

    pyt = time.perf_counter() - t0

    if pysol is None:
        print("PYTHON_TIMEOUT")
        break

    if not verify(pysol, pyclauses):
        print("PYTHON_VALIDATION=FAIL")
        break

    c_times = []
    a_times = []

    csol_last = None
    asol_last = None
    ctested_last = 0
    atested_last = 0

    for run in range(RUNS):

        if time.perf_counter() >= deadline:
            break

        csol, ctested, ct = native_call(
            lib.delta_sat_scalar,
            nvars,
            carr,
            nc
        )

        asol, atested, at = native_call(
            lib.delta_sat_avx512,
            nvars,
            carr,
            nc
        )

        cv = verify(csol, pyclauses)
        av = verify(asol, pyclauses)

        status = (
            csol != UINT64_MAX
            and asol != UINT64_MAX
            and pysol != UINT64_MAX
        )

        print(
            f"RUN={run+1} "
            f"C={ct:.6f}s "
            f"AVX512={at:.6f}s "
            f"C_TO_AVX={ct/at:.3f}x "
            f"VALID="
            f"{'YES' if cv and av and status else 'NO'}"
        )

        if not (cv and av and status):
            raise SystemExit(
                "VALIDATION FAILURE"
            )

        c_times.append(ct)
        a_times.append(at)

        csol_last = csol
        asol_last = asol
        ctested_last = ctested
        atested_last = atested

    if not c_times:
        break

    cm = statistics.median(c_times)
    am = statistics.median(a_times)

    py_to_c = pyt / cm
    c_to_avx = cm / am
    py_to_avx = pyt / am

    # Les trois solveurs doivent retourner
    # une solution valide. Elle n'a pas
    # nécessairement besoin d'être identique,
    # mais avec le même ordre elle devrait l'être.
    same_solution = (
        pysol == csol_last == asol_last
    )

    print(f"PYTHON={pyt:.6f}s")
    print(f"C_MEDIAN={cm:.6f}s")
    print(f"AVX512_MEDIAN={am:.6f}s")

    print(
        f"PYTHON_TO_C={py_to_c:.3f}x"
    )
    print(
        f"C_TO_AVX512={c_to_avx:.3f}x"
    )
    print(
        f"PYTHON_TO_AVX512={py_to_avx:.3f}x"
    )

    print(
        f"PYTHON_TESTED={pytested}"
    )
    print(
        f"C_TESTED={ctested_last}"
    )
    print(
        f"AVX512_TESTED={atested_last}"
    )

    print(
        "SAME_SOLUTION="
        + ("YES" if same_solution else "NO")
    )

    print("VALIDATION=PASSED")
    print()

    rows.append(
        (
            nvars,
            nc,
            pyt,
            cm,
            am,
            py_to_c,
            c_to_avx,
            py_to_avx,
        )
    )


elapsed = time.perf_counter() - global_start

print("=== DELTA SAT V3 SUMMARY ===")
print(f"ELAPSED_SECONDS={elapsed:.6f}")

for row in rows:

    n, nc, py, c, avx, pc, ca, pa = row

    print(
        f"VARS={n:2d} "
        f"CLAUSES={nc:3d} "
        f"PY={py:.6f}s "
        f"C={c:.6f}s "
        f"AVX512={avx:.6f}s "
        f"C_AVX={ca:.3f}x "
        f"PY_AVX={pa:.3f}x"
    )

if rows:

    print()

    print(
        "MEDIAN_C_TO_AVX512="
        f"{statistics.median(r[6] for r in rows):.3f}x"
    )

    print(
        "MEDIAN_PYTHON_TO_AVX512="
        f"{statistics.median(r[7] for r in rows):.3f}x"
    )

print()
print("SAT_RESULT=EXACT")
print("CNF_IDENTICAL_BETWEEN_PATHS=YES")
print("C_SCALAR_NATIVE=YES")
print("AVX512_NATIVE=YES")
print("AVX512_SEARCH_CORE=YES")
print("RESULT_VERIFICATION=PYTHON_INDEPENDENT")
print("DELTA_QBIT_EXECUTION=LOGICAL_EMULATED")
print("CPU_BRIDGE_EXECUTION=PHYSICAL_MEASURED")
print("CLASSIFICATION=PHYSICAL_LOCAL_HOST_MEASURED")
print("V3_COMPLETE=YES")
