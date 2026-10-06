#!/usr/bin/env python3
import os
import time
import queue
import hashlib
import threading
import statistics
import numpy as np

from delta_x512 import X512Core, aligne, vecs

# ============================================================
# DELTA GLOBAL +100 — TARGET 4550
# ============================================================

CPU = os.cpu_count() or 32

DELTA_BASE_MHZ   = 4450.0
DELTA_TARGET_MHZ = 4550.0
RATIO             = DELTA_TARGET_MHZ / DELTA_BASE_MHZ

NREQ = 1024
NW = 16384 * 4
SEED = 894

WORKERS_A = CPU
WORKERS_B = CPU
GENERATORS = 2
SLOTS = 16

EXPECTED_SIG = "5fbecb10a8ef936f"

# Dernières mesures validées
BASE_PIPELINE_GBIT = 12.442
TARGET_PIPELINE_GBIT = BASE_PIPELINE_GBIT * RATIO

BASE_CORE_MEAN_US = 15.572
TARGET_CORE_MEAN_US = BASE_CORE_MEAN_US / RATIO

BASE_CORE_GINSTR = 2.367407
TARGET_CORE_GINSTR = BASE_CORE_GINSTR * RATIO

# Cartographie réellement mesurée.
# Les meilleurs sont placés en tête.
BEST_ORDER = [
    30,14,29,31,13,28,15,16,
    17,3,11,2,5,4,1,0,
    7,8,10,12,24,6,9,27,
    26,21,25,23,19,22,20,18
]

available = sorted(os.sched_getaffinity(0))
BEST_ORDER = [c for c in BEST_ORDER if c in available]

# ============================================================
# Helpers
# ============================================================

def signature(values):
    return hashlib.sha256(
        repr(values).encode()
    ).hexdigest()[:16]


def pin_current_thread(cpu):
    try:
        # Sous Linux, pid 0 = thread appelant pour sched_setaffinity.
        os.sched_setaffinity(0, {cpu})
        return True
    except Exception:
        return False


# ============================================================
# Shared aligned memory
# ============================================================

class SharedBank:

    def __init__(self, slots, nw):
        self.A=[]
        self.B=[]
        self.C=[]

        for _ in range(slots):
            self.A.append(aligne(np.empty(nw,np.uint64)))
            self.B.append(aligne(np.empty(nw,np.uint64)))
            self.C.append(aligne(np.empty(nw,np.uint64)))

        self.free=queue.Queue()

        for i in range(slots):
            self.free.put(i)

        self.generated=0
        self.gen_s=0.0
        self.copy_s=0.0
        self.lock=threading.Lock()

    def acquire(self):
        return self.free.get()

    def release(self,s):
        self.free.put(s)

    def fill(self,s,seed):
        t0=time.perf_counter()

        a,b,c=vecs(seed,NW)

        t1=time.perf_counter()

        np.copyto(self.A[s],a)
        np.copyto(self.B[s],b)
        np.copyto(self.C[s],c)

        t2=time.perf_counter()

        with self.lock:
            self.generated += 1
            self.gen_s += t1-t0
            self.copy_s += t2-t1


# ============================================================
# One complete benchmark configuration
# ============================================================

def run_case(name, pin_mode):

    bank=SharedBank(SLOTS,NW)

    workq=queue.Queue(maxsize=SLOTS)

    results=[None]*NREQ

    metrics_lock=threading.Lock()

    total_jobs=0
    total_ni=0

    tpop_samples=[]
    worker_jobs=[0]*(WORKERS_A+WORKERS_B)

    next_job=0
    next_job_lock=threading.Lock()

    # --------------------------------------------------------
    # X512 workers
    # --------------------------------------------------------

    def worker(wid):

        nonlocal total_jobs,total_ni

        if pin_mode:
            cpu = BEST_ORDER[wid % len(BEST_ORDER)]
            pin_current_thread(cpu)

        core=X512Core()

        local_jobs=0
        local_ni=0
        local_samples=[]

        while True:

            item=workq.get()

            if item is None:
                break

            jid,slot=item

            t0=time.perf_counter_ns()

            r,ni=core.tpop(
                bank.A[slot],
                bank.B[slot],
                bank.C[slot],
                prog="micro"
            )

            t1=time.perf_counter_ns()

            results[jid]=r

            local_jobs += 1
            local_ni += ni
            local_samples.append(t1-t0)

            bank.release(slot)

        with metrics_lock:
            total_jobs += local_jobs
            total_ni += local_ni
            worker_jobs[wid]=local_jobs
            tpop_samples.extend(local_samples)

    # --------------------------------------------------------
    # Two input generators
    # --------------------------------------------------------

    def generator(gid):

        nonlocal next_job

        # Generators deliberately use the slower end first,
        # leaving the best measured logical CPUs preferentially
        # available to X512 execution.
        if pin_mode and BEST_ORDER:
            cpu=BEST_ORDER[-1-(gid % min(2,len(BEST_ORDER)))]
            pin_current_thread(cpu)

        while True:

            with next_job_lock:
                if next_job >= NREQ:
                    return

                jid=next_job
                next_job += 1

            slot=bank.acquire()

            bank.fill(
                slot,
                SEED+jid
            )

            workq.put((jid,slot))

    # --------------------------------------------------------
    # Start X512 workers before timed processing
    # --------------------------------------------------------

    workers=[]

    for wid in range(WORKERS_A+WORKERS_B):
        t=threading.Thread(
            target=worker,
            args=(wid,)
        )
        workers.append(t)
        t.start()

    generators=[
        threading.Thread(
            target=generator,
            args=(i,)
        )
        for i in range(GENERATORS)
    ]

    t0=time.perf_counter()

    for g in generators:
        g.start()

    for g in generators:
        g.join()

    for _ in workers:
        workq.put(None)

    for t in workers:
        t.join()

    wall=time.perf_counter()-t0

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    a=np.asarray(
        tpop_samples,
        dtype=np.float64
    ) / 1000.0

    mean_us=float(np.mean(a))
    median_us=float(np.median(a))
    min_us=float(np.min(a))
    p95_us=float(np.percentile(a,95))
    p99_us=float(np.percentile(a,99))
    max_us=float(np.max(a))

    bitops=3.0*NREQ*NW*64

    gbit=bitops/wall/1e9
    req_s=NREQ/wall
    ginstr_wall=total_ni/wall/1e9

    tpop_total_s=sum(tpop_samples)/1e9

    ginstr_active=(
        total_ni/tpop_total_s/1e9
        if tpop_total_s
        else 0.0
    )

    sig=signature(results)

    valid=(
        sig==EXPECTED_SIG
        and total_jobs==NREQ
        and bank.generated==NREQ
    )

    return {
        "name":name,
        "wall":wall,
        "gbit":gbit,
        "req_s":req_s,
        "ginstr_wall":ginstr_wall,
        "ginstr_active":ginstr_active,
        "mean_us":mean_us,
        "median_us":median_us,
        "min_us":min_us,
        "p95_us":p95_us,
        "p99_us":p99_us,
        "max_us":max_us,
        "gen_s":bank.gen_s,
        "copy_s":bank.copy_s,
        "ni":total_ni,
        "jobs":total_jobs,
        "sig":sig,
        "valid":valid,
        "worker_min":min(worker_jobs),
        "worker_max":max(worker_jobs),
    }


# ============================================================
# A/B test
# ============================================================

print("="*78)
print("DELTA X512 — GLOBAL +100 / TARGET 4550")
print("="*78)

print(f"DELTA_BASE={DELTA_BASE_MHZ:.0f}_MHz")
print(f"DELTA_TARGET={DELTA_TARGET_MHZ:.0f}_MHz")
print(f"GLOBAL_STEP=+100_MHz")
print(f"REQUIRED_RATIO={RATIO:.8f}x")
print()

print(f"PIPELINE_BASE={BASE_PIPELINE_GBIT:.3f}_Gbitop_s")
print(f"PIPELINE_TARGET={TARGET_PIPELINE_GBIT:.3f}_Gbitop_s")

print(f"CORE_BASE_MEAN={BASE_CORE_MEAN_US:.3f}_us")
print(f"CORE_TARGET_MEAN={TARGET_CORE_MEAN_US:.3f}_us")

print(f"CORE_BASE_GINSTR={BASE_CORE_GINSTR:.6f}")
print(f"CORE_TARGET_GINSTR={TARGET_CORE_GINSTR:.6f}")
print()

print("RUNNING_REFERENCE_UNPINNED...")
reference=run_case(
    "UNPINNED",
    False
)

# Short rest: don't compare two runs back-to-back at exactly
# the same transient scheduling instant.
time.sleep(0.5)

print("RUNNING_TOPOLOGY_PINNED...")
pinned=run_case(
    "TOPOLOGY_PINNED",
    True
)

cases=[reference,pinned]

print()
print("="*78)
print("RESULTS")
print("="*78)

for r in cases:

    print()
    print(f"[{r['name']}]")

    print(f"WALL={r['wall']:.6f}s")
    print(f"REQUESTS_S={r['req_s']:.2f}")
    print(f"GBITOP_S={r['gbit']:.3f}")

    print(
        f"X512_GINSTR_S_WALL="
        f"{r['ginstr_wall']:.6f}"
    )

    print(
        f"X512_GINSTR_S_ACTIVE="
        f"{r['ginstr_active']:.6f}"
    )

    print(f"TPOP_MIN_US={r['min_us']:.3f}")
    print(f"TPOP_MEDIAN_US={r['median_us']:.3f}")
    print(f"TPOP_MEAN_US={r['mean_us']:.3f}")
    print(f"TPOP_P95_US={r['p95_us']:.3f}")
    print(f"TPOP_P99_US={r['p99_us']:.3f}")
    print(f"TPOP_MAX_US={r['max_us']:.3f}")

    print(f"GENERATION_SUM_S={r['gen_s']:.6f}")
    print(f"COPY_SUM_S={r['copy_s']:.6f}")

    print(f"WORKER_MIN_JOBS={r['worker_min']}")
    print(f"WORKER_MAX_JOBS={r['worker_max']}")

    print(f"SIG={r['sig']}")

    print(
        "VALIDATION="
        + ("PASS" if r["valid"] else "FAIL")
    )


# ============================================================
# Select only among valid results
# ============================================================

valid_cases=[
    r for r in cases
    if r["valid"]
]

if not valid_cases:
    print()
    print("GLOBAL_VALIDATION=FAIL")
    print("STATE=DELTA_4450_REMAINS_REFERENCE")
    raise SystemExit(1)

winner=max(
    valid_cases,
    key=lambda r:r["gbit"]
)

# Direct A/B improvement
ab_gain=(
    pinned["gbit"] /
    reference["gbit"]
    if reference["gbit"]
    else 0.0
)

ab_latency=(
    reference["mean_us"] /
    pinned["mean_us"]
    if pinned["mean_us"]
    else 0.0
)

pipeline_pass=(
    winner["gbit"] >=
    TARGET_PIPELINE_GBIT
)

# This latency target comes from the single-core baseline.
# Report separately; do not let it falsify pipeline validation.
latency_target_pass=(
    winner["mean_us"] <=
    TARGET_CORE_MEAN_US
)

print()
print("="*78)
print("GLOBAL +100 VERDICT")
print("="*78)

print(
    f"PINNING_GBIT_GAIN="
    f"{ab_gain:.6f}x"
)

print(
    f"PINNING_LATENCY_GAIN="
    f"{ab_latency:.6f}x"
)

print(
    f"WINNER={winner['name']}"
)

print(
    f"WINNER_GBITOP_S="
    f"{winner['gbit']:.3f}"
)

print(
    f"WINNER_TPOP_MEAN_US="
    f"{winner['mean_us']:.3f}"
)

print(
    f"WINNER_TPOP_MEDIAN_US="
    f"{winner['median_us']:.3f}"
)

print(
    f"PIPELINE_TARGET_GBITOP_S="
    f"{TARGET_PIPELINE_GBIT:.3f}"
)

print(
    "PIPELINE_4550_THRESHOLD="
    + ("PASS" if pipeline_pass else "FAIL")
)

print(
    f"LATENCY_TARGET_US="
    f"{TARGET_CORE_MEAN_US:.3f}"
)

print(
    "LATENCY_4550_THRESHOLD="
    + ("PASS" if latency_target_pass else "FAIL")
)

print(
    "CHECKSUM="
    + (
        "IDENTICAL"
        if winner["sig"]==EXPECTED_SIG
        else "FAIL"
    )
)

# 4550 global is only accepted when pipeline is validated.
# Latency remains a separately reported engineering target.
if (
    winner["valid"]
    and pipeline_pass
):
    state="DELTA_4550_GLOBAL_VALIDATED"
else:
    state="DELTA_4450_REMAINS_REFERENCE"

print(f"STATE={state}")
print("="*78)
