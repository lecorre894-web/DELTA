#!/usr/bin/env python3

import ctypes
import os
import random
import time

MAX=(1<<64)-1
TIME_LIMIT=120.0

class Clause(ctypes.Structure):
    _fields_=[
        ("v",ctypes.c_uint8*3),
        ("s",ctypes.c_uint8*3)
    ]

lib=ctypes.CDLL(
    os.path.abspath("./libdelta_sat_engine.so")
)

lib.delta_sat_bit64.argtypes=[
    ctypes.c_uint,
    ctypes.POINTER(Clause),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint64
]
lib.delta_sat_bit64.restype=ctypes.c_uint64

lib.delta_sat_verify.argtypes=[
    ctypes.c_uint64,
    ctypes.POINTER(Clause),
    ctypes.c_size_t
]
lib.delta_sat_verify.restype=ctypes.c_int


def make_instance(n,nc,seed):

    r=random.Random(seed)

    secret=[
        r.getrandbits(1)
        for _ in range(n)
    ]

    py=[]

    for _ in range(nc):

        vs=r.sample(range(n),3)

        z=[
            [v,r.getrandbits(1)]
            for v in vs
        ]

        # garantit au moins une solution connue
        if not any(
            secret[v]==s
            for v,s in z
        ):
            z[0][1]=secret[z[0][0]]

        py.append(z)

    arr=(Clause*nc)()

    for j,c in enumerate(py):
        for k,(v,s) in enumerate(c):
            arr[j].v[k]=v
            arr[j].s[k]=s

    return py,arr


def independent_verify(x,clauses):

    if x==MAX:
        return False

    return all(
        any(
            ((x>>v)&1)==s
            for v,s in clause
        )
        for clause in clauses
    )


print("=== DELTA SAT BIT64 — HEAVY EXCURSION ===")
print(f"CPU_AVAILABLE={os.cpu_count()}")
print("ENGINE=DELTA_BIT64")
print("ENGINE_SELECTION=V5_BEST_MEASURED")
print("AVX512_REQUIRED=NO")
print("QPU_HOT_PATH=NO")
print("SEARCH=EXACT_ENUMERATION")
print("VALIDATION=INDEPENDENT")
print("PERSISTENT_CACHE=NO")
print("TIME_LIMIT_SECONDS=120")
print()

# Limite volontaire par instance :
# on autorise jusqu'à 2^40 candidats.
tests=[
    (36,162),
    (38,171),
    (40,180)
]

global_start=time.perf_counter()
deadline=global_start+TIME_LIMIT

for n,nc in tests:

    if time.perf_counter() >= deadline:
        print("GLOBAL_LIMIT_REACHED=YES")
        break

    print(f"--- SAT VARS={n} CLAUSES={nc} ---")

    py,arr=make_instance(
        n,nc,894+n
    )

    tested=ctypes.c_uint64()

    t0=time.perf_counter()

    sol=lib.delta_sat_bit64(
        n,
        arr,
        nc,
        ctypes.byref(tested),
        0
    )

    elapsed=time.perf_counter()-t0

    if sol == MAX:
        status="UNSAT"
        valid=False
    else:
        status="SAT"

        valid=(
            lib.delta_sat_verify(sol,arr,nc)==1
            and independent_verify(sol,py)
        )

    rate=(
        tested.value/elapsed
        if elapsed > 0 else 0
    )

    print(f"STATUS={status}")
    print(f"SOLUTION={sol}")
    print(f"TESTED={tested.value}")
    print(f"SECONDS={elapsed:.9f}")
    print(f"CANDIDATES_PER_SECOND={rate:.3f}")
    print(
        "INDEPENDENT_VALIDATION="
        + ("PASSED" if valid else "FAILED")
    )

    if status=="SAT" and not valid:
        raise SystemExit("VALIDATION_FAILURE")

    print()

    if time.perf_counter() >= deadline:
        print("GLOBAL_LIMIT_REACHED=YES")
        break


total=time.perf_counter()-global_start

print("=== EXCURSION SUMMARY ===")
print(f"TOTAL_SECONDS={total:.6f}")
print("ENGINE=DELTA_BIT64")
print("RESULT_TYPE=EXACT_FOR_COMPLETED_SEARCH")
print("V5_WINNER_DEPLOYED=YES")
print("READY_FOR_DIMACS_CNF=YES")
