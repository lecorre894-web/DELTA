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

# ============================================================
# DELTA X512 OC +100 V1
# ============================================================

BASE_CLOCK_MHZ   = 4300.0
TARGET_CLOCK_MHZ = 4400.0
OC_MHZ           = TARGET_CLOCK_MHZ - BASE_CLOCK_MHZ
OC_RATIO         = TARGET_CLOCK_MHZ / BASE_CLOCK_MHZ

# Référence mesurée DUAL-LINK V2
BASELINE_GBIT = 7.411
TARGET_GBIT   = BASELINE_GBIT * OC_RATIO

WORKERS_A = CPU
WORKERS_B = CPU

NREQ = 1024
NW = 16384 * 4
SEED = 894

# Plus de marge pour découpler producteurs/consommateurs
MEMORY_SLOTS = 16

# Deux générateurs
GENERATORS = 2

EXPECTED_SIG = "5fbecb10a8ef936f"


def signature(values):
    return hashlib.sha256(
        repr(values).encode()
    ).hexdigest()[:16]


# ============================================================
# PHYSICAL CLOCK PROBE
# ============================================================

def read_cpu_mhz():

    vals=[]

    try:
        with open("/proc/cpuinfo","r") as f:

            for line in f:

                if line.lower().startswith("cpu mhz"):

                    vals.append(
                        float(
                            line.split(":")[1]
                        )
                    )

    except Exception:
        pass

    if not vals:
        return None

    return statistics.mean(vals)


# ============================================================
# SHARED MEMORY
# ============================================================

class SharedMemory:

    def __init__(self, slots, nw):

        self.nw=nw

        self.A=[]
        self.B=[]
        self.C=[]

        for _ in range(slots):

            self.A.append(
                aligne(
                    np.empty(
                        nw,
                        dtype=np.uint64
                    )
                )
            )

            self.B.append(
                aligne(
                    np.empty(
                        nw,
                        dtype=np.uint64
                    )
                )
            )

            self.C.append(
                aligne(
                    np.empty(
                        nw,
                        dtype=np.uint64
                    )
                )
            )

        self.free=queue.Queue()

        for i in range(slots):
            self.free.put(i)

        self.generated=0
        self.gen_seconds=0.0
        self.copy_seconds=0.0

        self.lock=threading.Lock()


    def acquire(self):
        return self.free.get()


    def release(self, slot):
        self.free.put(slot)


    def generate(self, slot, seed):

        t0=time.perf_counter()

        a,b,c=vecs(
            seed,
            self.nw
        )

        t1=time.perf_counter()

        np.copyto(
            self.A[slot],
            a
        )

        np.copyto(
            self.B[slot],
            b
        )

        np.copyto(
            self.C[slot],
            c
        )

        t2=time.perf_counter()

        with self.lock:

            self.generated += 1
            self.gen_seconds += t1-t0
            self.copy_seconds += t2-t1


# ============================================================
# X512 PROCESSOR
# ============================================================

class X512Processor:

    def __init__(
        self,
        name,
        workers,
        memory,
        ready,
        results
    ):

        self.name=name
        self.workers=workers
        self.memory=memory
        self.ready=ready
        self.results=results

        self.cores=[
            X512Core()
            for _ in range(workers)
        ]

        self.threads=[]

        self.jobs=0
        self.ni=0
        self.tpop_seconds=0.0

        self.lock=threading.Lock()


    def worker(self, wid):

        core=self.cores[wid]

        local_jobs=0
        local_ni=0
        local_tpop=0.0

        while True:

            item=self.ready.get()

            if item is None:
                break

            job_id,slot=item

            A=self.memory.A[slot]
            B=self.memory.B[slot]
            C=self.memory.C[slot]

            t0=time.perf_counter()

            result,ni=core.tpop(
                A,B,C,
                prog="micro"
            )

            t1=time.perf_counter()

            self.results[job_id]=result

            local_jobs += 1
            local_ni += ni
            local_tpop += t1-t0

            self.memory.release(slot)


        with self.lock:

            self.jobs += local_jobs
            self.ni += local_ni
            self.tpop_seconds += local_tpop


    def start(self):

        for wid in range(self.workers):

            t=threading.Thread(
                target=self.worker,
                args=(wid,),
                name=f"{self.name}-{wid}"
            )

            self.threads.append(t)
            t.start()


    def join(self):

        for t in self.threads:
            t.join()


# ============================================================
# BUILD PIPELINE
# ============================================================

memory=SharedMemory(
    MEMORY_SLOTS,
    NW
)

ready=queue.Queue(
    maxsize=MEMORY_SLOTS
)

results=[None]*NREQ

A=X512Processor(
    "X512-A",
    WORKERS_A,
    memory,
    ready,
    results
)

B=X512Processor(
    "X512-B",
    WORKERS_B,
    memory,
    ready,
    results
)


# ============================================================
# DUAL GENERATOR
# ============================================================

next_job=0
job_lock=threading.Lock()


def get_job():

    global next_job

    with job_lock:

        if next_job >= NREQ:
            return None

        j=next_job
        next_job += 1

        return j


def generator(gid):

    while True:

        jid=get_job()

        if jid is None:
            return

        slot=memory.acquire()

        memory.generate(
            slot,
            SEED+jid
        )

        ready.put(
            (jid,slot)
        )


# ============================================================
# CLOCK SAMPLER
# ============================================================

clock_samples=[]
clock_stop=False


def clock_sampler():

    while not clock_stop:

        mhz=read_cpu_mhz()

        if mhz is not None:
            clock_samples.append(mhz)

        time.sleep(0.025)


# ============================================================
# RUN
# ============================================================

print("="*78)
print(" DELTA X512 DUAL-LINK — OC +100 PROCESSING PROFILE V1")
print("="*78)

print(f"PHYSICAL_BASE_REFERENCE={BASE_CLOCK_MHZ:.0f}_MHz")
print(f"DELTA_TARGET_CLOCK={TARGET_CLOCK_MHZ:.0f}_MHz")
print(f"DELTA_OC=+{OC_MHZ:.0f}_MHz")
print(f"TARGET_MULTIPLIER={OC_RATIO:.6f}x")

print()

print(f"X512_A_CONTEXTS={WORKERS_A}")
print(f"X512_B_CONTEXTS={WORKERS_B}")
print(f"TOTAL_CONTEXTS={WORKERS_A+WORKERS_B}")
print(f"GENERATORS={GENERATORS}")
print(f"SHARED_MEMORY_SLOTS={MEMORY_SLOTS}")
print(f"REQUESTS={NREQ}")
print(f"WORDS_PER_REQUEST={NW}")

print()

clock_thread=threading.Thread(
    target=clock_sampler,
    daemon=True
)

clock_thread.start()

A.start()
B.start()

generators=[
    threading.Thread(
        target=generator,
        args=(i,),
        name=f"GENERATOR-{i}"
    )
    for i in range(GENERATORS)
]

t0=time.perf_counter()

for g in generators:
    g.start()

for g in generators:
    g.join()

# All work is now queued or executing.
for _ in range(WORKERS_A+WORKERS_B):
    ready.put(None)

A.join()
B.join()

wall=time.perf_counter()-t0

clock_stop=True
clock_thread.join(timeout=0.2)


# ============================================================
# METRICS
# ============================================================

jobs=A.jobs+B.jobs
total_ni=A.ni+B.ni

tpop_seconds=(
    A.tpop_seconds+
    B.tpop_seconds
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

ginstr_wall=(
    total_ni /
    wall /
    1e9
)

ginstr_active=(
    total_ni /
    tpop_seconds /
    1e9
    if tpop_seconds
    else 0.0
)

requests_s=(
    jobs /
    wall
)

result_sig=signature(
    results
)

physical_avg=(
    statistics.mean(clock_samples)
    if clock_samples
    else 0.0
)

physical_min=(
    min(clock_samples)
    if clock_samples
    else 0.0
)

physical_max=(
    max(clock_samples)
    if clock_samples
    else 0.0
)


# ============================================================
# OC EQUIVALENT
#
# Conversion du débit mesuré en fréquence DELTA équivalente,
# relativement au DUAL-LINK V2 = 4300 MHz de référence.
# ============================================================

processing_ratio=(
    gbit /
    BASELINE_GBIT
)

equivalent_mhz=(
    BASE_CLOCK_MHZ *
    processing_ratio
)

equivalent_oc=(
    equivalent_mhz -
    BASE_CLOCK_MHZ
)

gain_pct=(
    (processing_ratio-1.0)*100.0
)

target_reached=(
    gbit >= TARGET_GBIT
)


# ============================================================
# VALIDATION
# ============================================================

checksum_ok=(
    result_sig==EXPECTED_SIG
)

jobs_ok=(
    jobs==NREQ
)

generation_ok=(
    memory.generated==NREQ
)

valid=(
    checksum_ok
    and
    jobs_ok
    and
    generation_ok
)


# ============================================================
# REPORT
# ============================================================

print("--------------- PHYSICAL CLOCK ---------------")

print(
    f"CPU_MHZ_MIN={physical_min:.1f}"
)

print(
    f"CPU_MHZ_AVG={physical_avg:.1f}"
)

print(
    f"CPU_MHZ_MAX={physical_max:.1f}"
)

print()


print("--------------- PIPELINE ---------------------")

print(
    f"WALL={wall:.6f}s"
)

print(
    f"GENERATION_SUM_S={memory.gen_seconds:.6f}"
)

print(
    f"COPY_SUM_S={memory.copy_seconds:.6f}"
)

print(
    f"TPOP_SUM_S={tpop_seconds:.6f}"
)

print(
    f"X512_A_JOBS={A.jobs}"
)

print(
    f"X512_B_JOBS={B.jobs}"
)

print(
    f"TOTAL_JOBS={jobs}"
)

print()


print("--------------- PROCESSING -------------------")

print(
    f"REQUESTS_S={requests_s:.2f}"
)

print(
    f"TOTAL_NI={total_ni}"
)

print(
    f"X512_GINSTR_S_WALL={ginstr_wall:.6f}"
)

print(
    f"X512_GINSTR_S_ACTIVE={ginstr_active:.6f}"
)

print(
    f"GBITOP_S={gbit:.3f}"
)

print()


print("--------------- OC PROFILE -------------------")

print(
    f"BASELINE_GBITOP_S={BASELINE_GBIT:.3f}"
)

print(
    f"TARGET_4400_GBITOP_S={TARGET_GBIT:.3f}"
)

print(
    f"OC_MEASURED_GBITOP_S={gbit:.3f}"
)

print(
    f"PROCESSING_GAIN={processing_ratio:.6f}x"
)

print(
    f"PROCESSING_GAIN_PERCENT={gain_pct:+.3f}%"
)

print(
    f"DELTA_EQUIVALENT_CLOCK={equivalent_mhz:.1f}_MHz"
)

print(
    f"DELTA_EQUIVALENT_OC={equivalent_oc:+.1f}_MHz"
)

print(
    "TARGET_4400_REACHED="
    + ("YES" if target_reached else "NO")
)

print()


print("--------------- VALIDATION -------------------")

print(
    f"SIG={result_sig}"
)

print(
    f"EXPECTED_SIG={EXPECTED_SIG}"
)

print(
    "CHECKSUM="
    + ("IDENTICAL" if checksum_ok else "FAIL")
)

print(
    "OC_PROFILE_VALIDATION="
    + ("PASS" if valid else "FAIL")
)


# ============================================================
# WINNER — DERNIÈRE LIGNE
# ============================================================

if not valid:

    winner="INVALID"

elif target_reached:

    winner="DELTA_X512_OC100"

else:

    winner="DELTA_X512_DUAL_LINK_V2"


print()
print("="*78)
print("WINNER="+winner)
print("="*78)
