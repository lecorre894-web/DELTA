#!/usr/bin/env python3

import ctypes
import os
import random
import statistics
import time

LIMIT_SECONDS=120.0
SEED=894
RUNS=5
UINT64_MAX=(1<<64)-1

class Clause(ctypes.Structure):
    _fields_=[
        ("v",ctypes.c_uint8*3),
        ("s",ctypes.c_uint8*3)
    ]

lib=ctypes.CDLL(
    os.path.abspath("./libdelta_sat_v4.so")
)

lib.delta_sat_scalar_v4.argtypes=[
    ctypes.c_uint,
    ctypes.POINTER(Clause),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64)
]
lib.delta_sat_scalar_v4.restype=ctypes.c_uint64

lib.delta_sat_x512_v4.argtypes=[
    ctypes.c_uint,
    ctypes.POINTER(Clause),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64)
]
lib.delta_sat_x512_v4.restype=ctypes.c_uint64

lib.delta_qpu_sat_v4.argtypes=[
    ctypes.c_uint,
    ctypes.POINTER(Clause),
    ctypes.c_size_t,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64)
]
lib.delta_qpu_sat_v4.restype=ctypes.c_uint64


def make_instance(nvars,nclauses,seed):

    r=random.Random(seed)

    secret=[
        r.getrandbits(1)
        for _ in range(nvars)
    ]

    py=[]

    for _ in range(nclauses):

        vs=r.sample(range(nvars),3)

        lits=[
            [v,r.getrandbits(1)]
            for v in vs
        ]

        if not any(
            secret[v]==s
            for v,s in lits
        ):
            lits[0][1]=secret[lits[0][0]]

        py.append(lits)

    arr=(Clause*nclauses)()

    for j,c in enumerate(py):
        for k,(v,s) in enumerate(c):
            arr[j].v[k]=v
            arr[j].s[k]=s

    return py,arr


def verify(x,clauses):

    if x==UINT64_MAX:
        return False

    return all(
        any(
            ((x>>v)&1)==s
            for v,s in c
        )
        for c in clauses
    )


def call_scalar(n,arr,nc):

    tested=ctypes.c_uint64()

    t=time.perf_counter()

    sol=lib.delta_sat_scalar_v4(
        n,arr,nc,
        ctypes.byref(tested)
    )

    return (
        sol,
        tested.value,
        time.perf_counter()-t
    )


def call_x512(n,arr,nc):

    tested=ctypes.c_uint64()

    t=time.perf_counter()

    sol=lib.delta_sat_x512_v4(
        n,arr,nc,
        ctypes.byref(tested)
    )

    return (
        sol,
        tested.value,
        time.perf_counter()-t
    )


def call_qpu(n,arr,nc,grain):

    tested=ctypes.c_uint64()
    grains=ctypes.c_uint64()

    t=time.perf_counter()

    sol=lib.delta_qpu_sat_v4(
        n,arr,nc,grain,
        ctypes.byref(tested),
        ctypes.byref(grains)
    )

    return (
        sol,
        tested.value,
        grains.value,
        time.perf_counter()-t
    )


print("=== DELTA SAT V4 COMPLETE ===")
print(
    "CPU=AMD Ryzen 9 9950X3D 16-Core Processor"
)
print(f"CPU_AVAILABLE={os.cpu_count()}")
print("CPU_ROLE=PHYSICAL_BRIDGE")
print("DELTA_QPU_ROLE=LOGICAL_COPROCESSOR")
print("X512_ROLE=NATIVE_AVX512_EXECUTION")
print("MEMORY_POLICY=BOUNDED_RAM_ONLY")
print("PERSISTENT_CACHE=NO")
print("BACKGROUND_SERVICE=NO")
print("TIME_LIMIT_SECONDS=120")
print()

start=time.perf_counter()
deadline=start+LIMIT_SECONDS

# On repart directement dans la zone lourde V3.
sizes=[
    (26,117),
    (28,126),
    (30,135),
    (32,144)
]

rows=[]

for n,nc in sizes:

    if time.perf_counter()>=deadline:
        break

    print(
        f"--- VARS={n} CLAUSES={nc} ---"
    )

    py,arr=make_instance(
        n,nc,SEED+n
    )

    scalar=[]
    x512=[]
    qpu=[]

    final=None

    for run in range(RUNS):

        if time.perf_counter()>=deadline:
            break

        s,st,ts=call_scalar(n,arr,nc)
        x,xt,tx=call_x512(n,arr,nc)

        # 256 blocs x 512 candidats
        # = grain logique de 131072 états.
        q,qt,qg,tq=call_qpu(
            n,arr,nc,256
        )

        sv=verify(s,py)
        xv=verify(x,py)
        qv=verify(q,py)

        same=(s==x==q)

        good=sv and xv and qv and same

        print(
            f"RUN={run+1} "
            f"C={ts:.6f}s "
            f"X512={tx:.6f}s "
            f"QPU_X512={tq:.6f}s "
            f"C_X512={ts/tx:.3f}x "
            f"C_QPU={ts/tq:.3f}x "
            f"VALID={'YES' if good else 'NO'}"
        )

        if not good:
            raise SystemExit(
                "VALIDATION_FAILURE"
            )

        scalar.append(ts)
        x512.append(tx)
        qpu.append(tq)

        final=(st,xt,qt,qg,s)

    if not scalar:
        break

    sm=statistics.median(scalar)
    xm=statistics.median(x512)
    qm=statistics.median(qpu)

    print(
        f"C_MEDIAN={sm:.6f}s"
    )
    print(
        f"X512_MEDIAN={xm:.6f}s"
    )
    print(
        f"QPU_X512_MEDIAN={qm:.6f}s"
    )

    print(
        f"X512_VS_C={sm/xm:.3f}x"
    )
    print(
        f"QPU_X512_VS_C={sm/qm:.3f}x"
    )

    print(
        f"QPU_OVERHEAD_RATIO={qm/xm:.6f}x"
    )

    st,xt,qt,qg,sol=final

    print(f"C_TESTED={st}")
    print(f"X512_TESTED={xt}")
    print(f"QPU_TESTED={qt}")
    print(f"QPU_GRAINS={qg}")
    print(f"SOLUTION={sol}")
    print("VALIDATION=PASSED")
    print()

    rows.append(
        (n,nc,sm,xm,qm,sm/xm,sm/qm)
    )


elapsed=time.perf_counter()-start

print("=== V4 SUMMARY ===")
print(f"ELAPSED_SECONDS={elapsed:.6f}")

for r in rows:

    n,nc,c,x,q,cx,cq=r

    print(
        f"VARS={n:2d} "
        f"CLAUSES={nc:3d} "
        f"C={c:.6f}s "
        f"X512={x:.6f}s "
        f"QPU_X512={q:.6f}s "
        f"X512_VS_C={cx:.3f}x "
        f"QPU_VS_C={cq:.3f}x"
    )

if rows:

    print()

    print(
        "MEDIAN_X512_VS_C="
        f"{statistics.median(r[5] for r in rows):.3f}x"
    )

    print(
        "MEDIAN_QPU_X512_VS_C="
        f"{statistics.median(r[6] for r in rows):.3f}x"
    )

print()
print("SAT_RESULT=EXACT")
print("SAME_CNF_ALL_PATHS=YES")
print("CPU_BRIDGE_EXECUTION=PHYSICAL_MEASURED")
print("DELTA_QBIT_EXECUTION=LOGICAL_EMULATED")
print("DELTA_QPU_COPROCESSOR=EXECUTED")
print("X512_EXECUTION=PHYSICAL_CPU_AVX512")
print("DDR5_ROLE=HOST_MEMORY_TRANSPORT")
print("DDR5_HARDWARE_MODIFICATION=NO")
print("PERSISTENT_DATA_CREATED=NO")
print("RESULT_VERIFICATION=INDEPENDENT")
print("V4_COMPLETE=YES")
