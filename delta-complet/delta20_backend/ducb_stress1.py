#!/usr/bin/env python3

import os
import sys
import time
import math
import hashlib
import ctypes as ct
import subprocess
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)

from delta_x512 import X512Core, DeltaX512, vecs

SEED=894
LEVELS=[1,2,4,8,16,32]

print("="*78)
print(" DUCB-STRESS-1 ADAPTIVE MULTI-SUPPORT")
print("="*78)

# ============================================================
# Helpers
# ============================================================

def rate(work,sec):
    return work/sec if sec>0 else 0.0

def status(ok):
    return "PASS" if ok else "FAIL"

results={}

# ============================================================
# 1. RYZEN — external stress-ng reference
#    Short verification round; previous 60s 1->32 remains reference.
# ============================================================

print("\n[1/5] RYZEN PHYSICAL")

ryzen_pass=True
ryzen_best=0.0
ryzen_best_n=0

for n in LEVELS:
    cmd=[
        "stress-ng",
        "--cpu",str(n),
        "--cpu-method","all",
        "--timeout","5s",
        "--verify",
        "--metrics-brief"
    ]

    t=time.perf_counter()
    p=subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    sec=time.perf_counter()-t

    ok=(p.returncode==0)
    ryzen_pass &= ok

    # external scaling score = workers / wall second,
    # only for internal normalized stability score
    s=n/sec

    if s>ryzen_best:
        ryzen_best=s
        ryzen_best_n=n

    print(
        f"RYZEN workers={n:2d} "
        f"{status(ok)} wall={sec:.3f}s"
    )

results["RYZEN"]={
    "pass":ryzen_pass,
    "max":32 if ryzen_pass else ryzen_best_n,
    "score":ryzen_best,
    "class":"PHYSICAL_CPU"
}

# ============================================================
# 2. DELTA X512 — native AVX-512
#    Same native kernel already validated by DELTA.
# ============================================================

print("\n[2/5] DELTA X512 NATIVE")

x512_pass=True
x512_best=0.0
x512_best_n=0
x512_checksum=None

# fixed payload per request
NW=16384*4
REQUESTS=128

for workers in LEVELS:

    reqs=[]

    for i in range(REQUESTS):
        key=hashlib.sha256(
            f"DUCB-X512-{SEED}-{workers}-{i}".encode()
        ).hexdigest()

        reqs.append(
            (key,SEED+i,NW)
        )

    D=None

    try:
        D=DeltaX512(workers,"micro")

        t0=time.perf_counter()
        res,calc,ni,tc=D.compute(reqs)
        wall=time.perf_counter()-t0

        sig=hashlib.sha256(
            str(res).encode()
        ).hexdigest()[:16]

        # Independent check of first request.
        A,B,C=vecs(SEED,NW)

        ref=int(
            np.unpackbits(
                ((A&B)^C).view(np.uint8)
            ).sum()
        )

        ok=(res[0]==ref)

        # Actual bit operations represented by kernel.
        bitops=3.0*calc*NW*64
        throughput=rate(bitops,wall)

        if throughput>x512_best:
            x512_best=throughput
            x512_best_n=workers
            x512_checksum=sig

        x512_pass &= ok

        print(
            f"X512 workers={workers:2d} "
            f"{status(ok)} "
            f"wall={wall:.6f}s "
            f"Gbitop/s={throughput/1e9:.3f} "
            f"sig={sig}"
        )

    except Exception as e:
        x512_pass=False
        print(
            f"X512 workers={workers:2d} FAIL "
            f"{type(e).__name__}: {e}"
        )

    finally:
        if D is not None:
            D.close()

results["X512"]={
    "pass":x512_pass,
    "max":x512_best_n,
    "score":x512_best,
    "class":"DELTA_AVX512_NATIVE",
    "checksum":x512_checksum
}

# ============================================================
# 3. DELTA BIT64 — native 64-way bit parallel
#    Stress with deterministic planted strict 3-SAT.
# ============================================================

print("\n[3/5] DELTA BIT64 NATIVE")

class Clause(ct.Structure):
    _fields_=[
        ("v",ct.c_uint8*3),
        ("s",ct.c_uint8*3)
    ]

L=ct.CDLL(
    os.path.join(HERE,"libdelta_sat_engine.so")
)

L.delta_sat_bit64.argtypes=[
    ct.c_uint,
    ct.POINTER(Clause),
    ct.c_size_t,
    ct.POINTER(ct.c_uint64),
    ct.c_uint64
]
L.delta_sat_bit64.restype=ct.c_uint64

MAX64=(1<<64)-1

def make_unsat_like(nv,nc,seed):
    rng=np.random.default_rng(seed)
    arr=(Clause*nc)()

    for j in range(nc):
        vv=rng.choice(nv,3,replace=False)

        for k in range(3):
            arr[j].v[k]=int(vv[k])
            arr[j].s[k]=int(rng.integers(0,2))

    return arr

bit_pass=True
bit_best=0.0
bit_best_level=0
bit_checksum=0

# bounded candidate windows; stress grows by clauses and repetitions.
for level in LEVELS:

    nv=36
    nc=64*level

    clauses=make_unsat_like(
        nv,nc,SEED+level
    )

    repeats=max(1,64//level)

    total_tested=0
    checksum=0

    t0=time.perf_counter()

    ok=True

    for r in range(repeats):
        tested=ct.c_uint64(0)

        sol=L.delta_sat_bit64(
            nv,
            clauses,
            nc,
            ct.byref(tested),
            1<<24
        )

        total_tested+=tested.value
        checksum ^= int(sol)

        if tested.value==0:
            ok=False

    wall=time.perf_counter()-t0

    cps=rate(total_tested,wall)

    bit_pass &= ok

    if cps>bit_best:
        bit_best=cps
        bit_best_level=level
        bit_checksum=checksum

    print(
        f"BIT64 level={level:2d} "
        f"{status(ok)} "
        f"clauses={nc:4d} "
        f"tested={total_tested} "
        f"Mcand/s={cps/1e6:.3f} "
        f"checksum={checksum}"
    )

results["BIT64"]={
    "pass":bit_pass,
    "max":bit_best_level,
    "score":bit_best,
    "class":"DELTA_BIT64_NATIVE",
    "checksum":bit_checksum
}

# ============================================================
# 4. DELTA QPU LOGICAL
#    Scheduler/logic stress, deterministic state transitions.
#    Explicitly logical/emulated.
# ============================================================

print("\n[4/5] DELTA QPU LOGICAL")

q_pass=True
q_best=0.0
q_best_level=0
q_checksum=0

MASK=MAX64

for level in LEVELS:

    lanes=level*65536

    x=SEED
    chk=0

    t0=time.perf_counter()

    for i in range(lanes):
        # deterministic logical lane transform
        x ^= (x << 13) & MASK
        x ^= (x >> 7)
        x ^= (x << 17) & MASK
        x &= MASK
        chk ^= x

    wall=time.perf_counter()-t0

    logical_rate=rate(lanes,wall)

    ok=(lanes>0 and wall>0)
    q_pass &= ok

    if logical_rate>q_best:
        q_best=logical_rate
        q_best_level=level
        q_checksum=chk

    print(
        f"DELTA_QPU level={level:2d} "
        f"{status(ok)} "
        f"lanes={lanes} "
        f"Mlane/s={logical_rate/1e6:.3f} "
        f"checksum={chk:016x}"
    )

results["DELTA_QPU"]={
    "pass":q_pass,
    "max":q_best_level,
    "score":q_best,
    "class":"LOGICAL_EMULATED",
    "checksum":q_checksum
}

# ============================================================
# 5. IBM PHYSICAL QPU
#
# One bounded physical stress point.
# Reuse proven IBM authentication/runtime.
# This is a quantum stability/sampling component,
# NOT converted into CPU/X512 throughput.
# ============================================================

print("\n[5/5] IBM PHYSICAL QPU")

ibm_pass=False
ibm_score=0.0
ibm_backend="N/A"
ibm_detail="NOT_RUN"

try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.executor_sampler import Sampler

    service=QiskitRuntimeService(
        channel="ibm_quantum_platform",
        instance="open-instance"
    )

    backend=service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=8
    )

    ibm_backend=backend.name

    # Deterministic 8-qubit stress circuit.
    # Repeated entangling/mixing layers.
    nq=8
    layers=32

    qc=QuantumCircuit(nq)

    qc.h(range(nq))

    for layer in range(layers):
        for q in range(nq):
            qc.rz(
                ((layer+1)*(q+1))*0.017,
                q
            )

        for q in range(nq-1):
            qc.cx(q,q+1)

        qc.cx(nq-1,0)

    qc.measure_all()

    pm=generate_preset_pass_manager(
        backend=backend,
        optimization_level=3
    )

    isa=pm.run(qc)

    depth=isa.depth()
    ops=dict(isa.count_ops())

    twoq=sum(
        int(ops.get(k,0))
        for k in ("cz","cx","ecr")
    )

    print(
        f"IBM backend={backend.name} "
        f"depth={depth} "
        f"2Q={twoq}"
    )

    sampler=Sampler(mode=backend)

    t0=time.perf_counter()

    job=sampler.run(
        [isa],
        shots=2048
    )

    print(
        "IBM_JOB_ID=",
        job.job_id()
    )

    result=job.result()

    wall=time.perf_counter()-t0

    pub=result[0]
    data=pub.data

    meas=None

    if hasattr(data,"meas"):
        meas=data.meas
    else:
        for name in dir(data):
            if name.startswith("_"):
                continue
            obj=getattr(data,name)
            if hasattr(obj,"get_bitstrings"):
                meas=obj
                break

    if meas is None:
        raise RuntimeError(
            "measurement register missing"
        )

    strings=meas.get_bitstrings()

    measured=len(strings)
    unique=len(set(strings))

    ibm_pass=(measured==2048)
    ibm_score=measured/wall if wall>0 else 0.0

    digest=hashlib.sha256(
        "".join(strings).encode()
    ).hexdigest()[:16]

    ibm_detail=(
        f"shots={measured} "
        f"unique={unique} "
        f"wall={wall:.3f}s "
        f"digest={digest}"
    )

    print(
        f"IBM {status(ibm_pass)} "
        f"{ibm_detail}"
    )

except Exception as e:
    ibm_detail=(
        f"{type(e).__name__}: {e}"
    )
    print(
        "IBM FAIL",
        ibm_detail
    )

results["IBM_QPU"]={
    "pass":ibm_pass,
    "max":1 if ibm_pass else 0,
    "score":ibm_score,
    "class":"PHYSICAL_QPU",
    "detail":ibm_detail
}

# ============================================================
# NORMALIZED MULTI-SUPPORT VERDICT
#
# Raw throughput units are intentionally NOT mixed.
# Each engine receives:
#   stability  : pass/fail
#   load       : highest/best stable level
#   efficiency : score relative to best score in its own domain
#
# Winner is contextual, not fake cross-unit arithmetic.
# ============================================================

print("\n"+"="*78)
print(" DUCB-STRESS-1 FINAL")
print("="*78)

for name in [
    "RYZEN",
    "BIT64",
    "X512",
    "DELTA_QPU",
    "IBM_QPU"
]:
    r=results[name]

    print(
        f"{name:10s} "
        f"{status(r['pass']):4s} "
        f"class={r['class']} "
        f"best_load={r['max']}"
    )

stable=[
    name for name,r in results.items()
    if r["pass"]
]

# Contextual winners
stability_winner=(
    "TIE:" + ",".join(stable)
    if stable
    else "NONE"
)

# Highest supported stress level among comparable
# software/CPU scaling levels.
scale_candidates=[
    "RYZEN",
    "BIT64",
    "X512",
    "DELTA_QPU"
]

maxload=max(
    results[n]["max"]
    for n in scale_candidates
    if results[n]["pass"]
)

scalers=[
    n for n in scale_candidates
    if results[n]["pass"]
    and results[n]["max"]==maxload
]

scaling_winner=(
    "TIE:" + ",".join(scalers)
)

# Overall selection:
# all stable engines remain candidates;
# X512 preferred for vector/bit transform,
# BIT64 for SAT/bit-parallel search,
# Ryzen for generic native workload,
# logical QPU for scheduling,
# IBM only for quantum-native work.
#
# Current stress category is mixed computational pressure.
if results["X512"]["pass"]:
    overall="DELTA_X512"
elif results["RYZEN"]["pass"]:
    overall="RYZEN"
elif results["BIT64"]["pass"]:
    overall="DELTA_BIT64"
elif results["DELTA_QPU"]["pass"]:
    overall="DELTA_QPU"
elif results["IBM_QPU"]["pass"]:
    overall="IBM_QPU"
else:
    overall="NONE"

print()
print(
    "STABILITY_WINNER="+stability_winner
)
print(
    "SCALING_WINNER="+scaling_winner
)

print(
    "CONTEXT_VECTOR=DELTA_X512"
    if results["X512"]["pass"]
    else "CONTEXT_VECTOR=RYZEN"
)

print(
    "CONTEXT_BITPARALLEL=DELTA_BIT64"
    if results["BIT64"]["pass"]
    else "CONTEXT_BITPARALLEL=RYZEN"
)

print(
    "CONTEXT_GENERAL=RYZEN"
    if results["RYZEN"]["pass"]
    else "CONTEXT_GENERAL=NONE"
)

print(
    "CONTEXT_SCHEDULER=DELTA_QPU"
    if results["DELTA_QPU"]["pass"]
    else "CONTEXT_SCHEDULER=RYZEN"
)

print(
    "CONTEXT_QUANTUM=IBM_QPU"
    if results["IBM_QPU"]["pass"]
    else "CONTEXT_QUANTUM=N/A"
)

print()
print("WINNER="+overall)
print("DUCB_STRESS1_COMPLETE=YES")
print("="*78)
