#!/usr/bin/env python3
import ctypes
import os
import sys
import time
import random

MAX=(1<<64)-1
LIB="./libdelta_sat_engine.so"

class Clause(ctypes.Structure):
    _fields_=[
        ("v",ctypes.c_uint8*3),
        ("s",ctypes.c_uint8*3)
    ]

L=ctypes.CDLL(os.path.abspath(LIB))

L.delta_sat_bit64.argtypes=[
    ctypes.c_uint,
    ctypes.POINTER(Clause),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint64
]
L.delta_sat_bit64.restype=ctypes.c_uint64

def load_dimacs(path):
    nvars=None
    declared=None
    clauses=[]

    with open(path,"r",encoding="ascii") as f:
        pending=[]

        for lineno,line in enumerate(f,1):
            line=line.strip()

            if not line or line.startswith("c"):
                continue

            if line.startswith("p"):
                z=line.split()

                if len(z)!=4 or z[1]!="cnf":
                    raise ValueError(
                        f"ligne {lineno}: en-tête DIMACS invalide"
                    )

                nvars=int(z[2])
                declared=int(z[3])
                continue

            for token in line.split():
                lit=int(token)

                if lit==0:
                    if len(pending)!=3:
                        raise ValueError(
                            f"clause non-3-SAT: {len(pending)} littéraux"
                        )

                    clauses.append(tuple(pending))
                    pending=[]
                else:
                    pending.append(lit)

        if pending:
            raise ValueError("dernière clause sans terminateur 0")

    if nvars is None:
        raise ValueError("en-tête 'p cnf' absent")

    if not (1 <= nvars <= 62):
        raise ValueError(
            f"{nvars} variables: moteur BIT64 actuel limité à 62"
        )

    if declared != len(clauses):
        raise ValueError(
            f"clauses déclarées={declared}, lues={len(clauses)}"
        )

    for c in clauses:
        for lit in c:
            if abs(lit)<1 or abs(lit)>nvars:
                raise ValueError(
                    f"littéral hors domaine: {lit}"
                )

    arr=(Clause*len(clauses))()

    for j,c in enumerate(clauses):
        for k,lit in enumerate(c):
            # DIMACS x1 devient bit 0.
            arr[j].v[k]=abs(lit)-1

            # positif : variable doit valoir 1
            # négatif : variable doit valoir 0
            arr[j].s[k]=1 if lit>0 else 0

    return nvars,clauses,arr

def verify(solution,clauses):
    if solution==MAX:
        return False

    for clause in clauses:
        ok=False

        for lit in clause:
            bit=(solution>>(abs(lit)-1))&1

            if (lit>0 and bit==1) or \
               (lit<0 and bit==0):
                ok=True
                break

        if not ok:
            return False

    return True

def assignment(solution,nvars):
    return [
        (i+1) if ((solution>>i)&1) else -(i+1)
        for i in range(nvars)
    ]

if len(sys.argv)!=2:
    print("usage: python delta_dimacs_sat.py fichier.cnf")
    raise SystemExit(2)

path=sys.argv[1]

nvars,clauses,arr=load_dimacs(path)

print("=== DELTA DIMACS SAT ===")
print(f"FILE={path}")
print(f"VARIABLES={nvars}")
print(f"CLAUSES={len(clauses)}")
print("FORMAT=DIMACS_CNF_3SAT")
print("ENGINE=DELTA_BIT64")
print("SEARCH=EXACT")
print("KNOWN_SOLUTION_GIVEN_TO_SOLVER=NO")
print()

tested=ctypes.c_uint64()

t0=time.perf_counter()

sol=L.delta_sat_bit64(
    nvars,
    arr,
    len(clauses),
    ctypes.byref(tested),
    0
)

elapsed=time.perf_counter()-t0

print(f"SECONDS={elapsed:.9f}")
print(f"TESTED={tested.value}")

if elapsed:
    print(
        f"CANDIDATES_PER_SECOND="
        f"{tested.value/elapsed:.3f}"
    )

if sol==MAX:
    print("RESULT=UNSAT")
    print(
        "SEARCH_SPACE_EXHAUSTED="
        + ("YES" if tested.value==(1<<nvars) else "NO")
    )

else:
    valid=verify(sol,clauses)

    print("RESULT=SAT")
    print(f"SOLUTION_INTEGER={sol}")
    print(
        "INDEPENDENT_DIMACS_VALIDATION="
        + ("PASSED" if valid else "FAILED")
    )

    if not valid:
        raise SystemExit("VALIDATION_FAILURE")

    a=assignment(sol,nvars)

    print(
        "DIMACS_MODEL="
        + " ".join(map(str,a))
        + " 0"
    )

print()
print("SOLVER_CLASS=DELTA_BIT64_EXACT_3SAT")
print("QPU_HOT_PATH=NO")
print("AVX512_REQUIRED=NO")
print("PERSISTENT_CACHE=NO")
print("DIMACS_RUN_COMPLETE=YES")
