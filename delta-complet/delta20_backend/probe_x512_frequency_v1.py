#!/usr/bin/env python3

import os
import time
import queue
import hashlib
import threading
import statistics
import numpy as np

from delta_x512 import X512Core, aligne, vecs

CPU = os.cpu_count() or 32

# Configuration gagnante DUAL-LINK V2
WORKERS_A = CPU
WORKERS_B = CPU

NREQ = 1024
NW = 16384 * 4
SEED = 894
MEMORY_SLOTS = 8

REFERENCE_V2_GBIT = 7.411


def sig(x):
    return hashlib.sha256(
        repr(x).encode()
    ).hexdigest()[:16]


def read_cpu_mhz():
    vals=[]

    try:
        with open("/proc/cpuinfo","r") as f:
            for line in f:
                if line.lower().startswith("cpu mhz"):
                    vals.append(float(line.split(":")[1]))

    except Exception:
        pass

    if not vals:
        return None,None,None

    return (
        min(vals),
        statistics.mean(vals),
        max(vals)
    )


class SharedBank:

    def __init__(self, slots, nw):

        self.nw=nw

        self.A=[]
        self.B=[]
        self.C=[]

        for _ in range(slots):

            self.A.append(
                aligne(
                    np.empty(nw,np.uint64)
                )
            )

            self.B.append(
                aligne(
                    np.empty(nw,np.uint64)
                )
            )

            self.C.append(
                aligne(
                    np.empty(nw,np.uint64)
                )
            )

        self.free=queue.Queue()

        for i in range(slots):
            self.free.put(i)

        self.generated=0

        self.generation_seconds=0.0
        self.copy_seconds=0.0

        self.lock=threading.Lock()


    def acquire(self):
        return self.free.get()


    def release(self,s):
        self.free.put(s)


    def fill(self,slot,seed):

        t0=time.perf_counter()

        a,b,c=vecs(
            seed,
            self.nw
        )

        t1=time.perf_counter()

        np.copyto(
            self.A[slot],a
        )

        np.copyto(
            self.B[slot],b
        )

        np.copyto(
            self.C[slot],c
        )

        t2=time.perf_counter()

        with self.lock:

            self.generated+=1

            self.generation_seconds += (
                t1-t0
            )

            self.copy_seconds += (
                t2-t1
            )


class Processor:

    def __init__(
        self,
        name,
        workers,
        bank,
        workq,
        results
    ):

        self.name=name
        self.workers=workers
        self.bank=bank
        self.workq=workq
        self.results=results

        self.cores=[
            X512Core()
            for _ in range(workers)
        ]

        self.threads=[]

        self.jobs=0
        self.ni=0

        # Somme des temps de tous les appels.
        self.tpop_sum=0.0

        # Plus long cumul d'un worker :
        # approximation utile du chemin actif.
        self.worker_tpop=[
            0.0
            for _ in range(workers)
        ]

        self.wait_sum=0.0

        self.lock=threading.Lock()


    def worker(self,wid):

        core=self.cores[wid]

        local_jobs=0
        local_ni=0
        local_tpop=0.0
        local_wait=0.0

        while True:

            tw0=time.perf_counter()

            item=self.workq.get()

            tw1=time.perf_counter()

            local_wait += (
                tw1-tw0
            )

            if item is None:
                break

            job_id,slot=item

            A=self.bank.A[slot]
            B=self.bank.B[slot]
            C=self.bank.C[slot]

            tc0=time.perf_counter()

            r,ni=core.tpop(
                A,B,C,
                prog="micro"
            )

            tc1=time.perf_counter()

            self.results[job_id]=r

            local_jobs += 1
            local_ni += ni

            local_tpop += (
                tc1-tc0
            )

            self.bank.release(slot)


        with self.lock:

            self.jobs += local_jobs
            self.ni += local_ni

            self.tpop_sum += (
                local_tpop
            )

            self.worker_tpop[wid] = (
                local_tpop
            )

            self.wait_sum += (
                local_wait
            )


    def start(self):

        for i in range(
            self.workers
        ):

            t=threading.Thread(
                target=self.worker,
                args=(i,)
            )

            self.threads.append(t)
            t.start()


    def join(self):

        for t in self.threads:
            t.join()


bank=SharedBank(
    MEMORY_SLOTS,
    NW
)

workq=queue.Queue(
    maxsize=MEMORY_SLOTS
)

results=[None]*NREQ

A=Processor(
    "A",
    WORKERS_A,
    bank,
    workq,
    results
)

B=Processor(
    "B",
    WORKERS_B,
    bank,
    workq,
    results
)


def producer():

    for jid in range(NREQ):

        slot=bank.acquire()

        bank.fill(
            slot,
            SEED+jid
        )

        workq.put(
            (jid,slot)
        )

    for _ in range(
        WORKERS_A+WORKERS_B
    ):
        workq.put(None)


# CPU clock sampler
clock_samples=[]
clock_stop=False


def clock_probe():

    while not clock_stop:

        x=read_cpu_mhz()

        if x[1] is not None:
            clock_samples.append(x)

        time.sleep(0.05)


print("="*76)
print("DELTA X512 DUAL-LINK V2 — PROCESSING FREQUENCY PROBE")
print("="*76)

print(
    f"CPU_LOGICAL={CPU}"
)

print(
    f"X512_CONTEXTS="
    f"{WORKERS_A}+{WORKERS_B}"
)

print(
    f"REQUESTS={NREQ}"
)

print(
    f"WORDS_PER_REQUEST={NW}"
)

print(
    f"SHARED_MEMORY_SLOTS={MEMORY_SLOTS}"
)

print()


clock_thread=threading.Thread(
    target=clock_probe,
    daemon=True
)

clock_thread.start()

A.start()
B.start()

P=threading.Thread(
    target=producer
)

t0=time.perf_counter()

P.start()

P.join()

A.join()
B.join()

wall=time.perf_counter()-t0

clock_stop=True
clock_thread.join(timeout=0.2)


total_jobs=A.jobs+B.jobs
total_ni=A.ni+B.ni

tpop_sum=(
    A.tpop_sum+
    B.tpop_sum
)

# Effective aggregate processing frequency:
# instructions X512 completed per wall second.
ginstr_wall=(
    total_ni /
    wall /
    1e9
)

# Core-active rate:
# NI divided by actual accumulated TPOP execution time.
ginstr_active=(
    total_ni /
    tpop_sum /
    1e9
    if tpop_sum>0
    else 0.0
)

requests_s=(
    total_jobs /
    wall
)

bitops=(
    3.0 *
    NREQ *
    NW *
    64
)

gbit=(
    bitops /
    wall /
    1e9
)

# Data payload read by TPOP:
# A+B+C, 3 uint64 arrays per request.
payload_bytes=(
    NREQ *
    NW *
    8 *
    3
)

effective_payload_gbs=(
    payload_bytes /
    wall /
    1e9
)

avg_tpop_us=(
    tpop_sum /
    total_jobs *
    1e6
)

generation_wall_equiv=(
    bank.generation_seconds
)

copy_wall_equiv=(
    bank.copy_seconds
)

result_sig=sig(results)


if clock_samples:

    cpu_min=min(
        x[0]
        for x in clock_samples
    )

    cpu_avg=statistics.mean(
        x[1]
        for x in clock_samples
    )

    cpu_max=max(
        x[2]
        for x in clock_samples
    )

else:

    cpu_min=cpu_avg=cpu_max=0.0


print("--------------- PROCESSING ----------------")

print(
    f"WALL={wall:.6f}s"
)

print(
    f"JOBS={total_jobs}"
)

print(
    f"REQUESTS_S={requests_s:.2f}"
)

print(
    f"TOTAL_NI={total_ni}"
)

print(
    f"X512_GINSTR_S_WALL="
    f"{ginstr_wall:.6f}"
)

print(
    f"X512_GINSTR_S_ACTIVE="
    f"{ginstr_active:.6f}"
)

print(
    f"GBITOP_S={gbit:.3f}"
)

print(
    f"AVG_TPOP_US={avg_tpop_us:.3f}"
)

print(
    f"EFFECTIVE_PAYLOAD_GB_S="
    f"{effective_payload_gbs:.3f}"
)

print()


print("--------------- PIPELINE ------------------")

print(
    f"GENERATION_SUM_S="
    f"{generation_wall_equiv:.6f}"
)

print(
    f"COPY_SUM_S="
    f"{copy_wall_equiv:.6f}"
)

print(
    f"TPOP_SUM_S="
    f"{tpop_sum:.6f}"
)

print(
    f"A_TPOP_SUM_S="
    f"{A.tpop_sum:.6f}"
)

print(
    f"B_TPOP_SUM_S="
    f"{B.tpop_sum:.6f}"
)

print(
    f"A_JOBS={A.jobs}"
)

print(
    f"B_JOBS={B.jobs}"
)

print()


print("--------------- PHYSICAL CPU --------------")

print(
    f"CPU_MHZ_MIN={cpu_min:.1f}"
)

print(
    f"CPU_MHZ_AVG={cpu_avg:.1f}"
)

print(
    f"CPU_MHZ_MAX={cpu_max:.1f}"
)

print()


print("--------------- VALIDATION ----------------")

print(
    f"SIG={result_sig}"
)

print(
    "EXPECTED_SIG="
    "5fbecb10a8ef936f"
)

checksum_ok=(
    result_sig ==
    "5fbecb10a8ef936f"
)

jobs_ok=(
    total_jobs==NREQ
)

generation_ok=(
    bank.generated==NREQ
)

valid=(
    checksum_ok
    and
    jobs_ok
    and
    generation_ok
)

print(
    "CHECKSUM="
    + (
        "IDENTICAL"
        if checksum_ok
        else "FAIL"
    )
)

print(
    "PROBE_VALIDATION="
    + (
        "PASS"
        if valid
        else "FAIL"
    )
)

print()


# Diagnostic of where the wall time is going.
# generation/copy/tpop sums overlap across threads, so these
# percentages are indicators, not additive wall-time shares.

print("--------------- DIAGNOSTIC ----------------")

print(
    f"REFERENCE_DUALLINK_V2_GBITOP_S="
    f"{REFERENCE_V2_GBIT:.3f}"
)

print(
    f"CURRENT_GBITOP_S="
    f"{gbit:.3f}"
)

ratio=(
    gbit /
    REFERENCE_V2_GBIT
)

print(
    f"REFERENCE_RATIO="
    f"{ratio:.4f}x"
)

if not valid:

    bottleneck="INVALID"

elif bank.generation_seconds > tpop_sum*4:

    bottleneck="INPUT_GENERATION"

elif bank.copy_seconds > tpop_sum*2:

    bottleneck="MEMORY_COPY"

elif tpop_sum > bank.generation_seconds:

    bottleneck="TPOP_COMPUTE"

else:

    bottleneck="SCHEDULING_OR_PIPELINE"

print(
    "PRIMARY_BOTTLENECK="
    + bottleneck
)

print("="*76)
