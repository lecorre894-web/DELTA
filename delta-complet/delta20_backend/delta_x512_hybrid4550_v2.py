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
# DELTA X512 HYBRID 4550 V2
# ============================================================

CPU = os.cpu_count() or 32

NREQ = 1024
NW = 16384 * 4
SEED = 894

WORKERS = 64
GENERATORS = 2
SLOTS = 16

EXPECTED_SIG = "5fbecb10a8ef936f"

DELTA_REFERENCE_MHZ = 4450.0
DELTA_TARGET_MHZ = 4550.0

BASELINE_GBIT = 12.442

TARGET_GBIT = (
    BASELINE_GBIT *
    DELTA_TARGET_MHZ /
    DELTA_REFERENCE_MHZ
)

# Ordre issu de notre mesure CORE-BOOST.
BEST_CPUS = [
    30,14,29,31,
    13,28,15,16,
    17,3,11,2,
    5,4,1,0,
    7,8,10,12,
    24,6,9,27,
    26,21,25,23,
    19,22,20,18
]

AVAILABLE = sorted(
    os.sched_getaffinity(0)
)

BEST_CPUS = [
    c for c in BEST_CPUS
    if c in AVAILABLE
]

# Correctif V2 : seulement 4 workers critiques épinglés.
# CPU 30,14,29,31 = quatre meilleures médianes mesurées.
PINNED_WORKERS = 4


def signature(values):
    return hashlib.sha256(
        repr(values).encode()
    ).hexdigest()[:16]


def pin(cpu):
    try:
        os.sched_setaffinity(
            0,
            {cpu}
        )
        return True
    except Exception:
        return False


class SharedBank:

    def __init__(self):

        self.A=[]
        self.B=[]
        self.C=[]

        for _ in range(SLOTS):

            self.A.append(
                aligne(
                    np.empty(
                        NW,
                        np.uint64
                    )
                )
            )

            self.B.append(
                aligne(
                    np.empty(
                        NW,
                        np.uint64
                    )
                )
            )

            self.C.append(
                aligne(
                    np.empty(
                        NW,
                        np.uint64
                    )
                )
            )

        self.free=queue.Queue()

        for i in range(SLOTS):
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

        a,b,c=vecs(
            seed,
            NW
        )

        t1=time.perf_counter()

        np.copyto(
            self.A[s],a
        )

        np.copyto(
            self.B[s],b
        )

        np.copyto(
            self.C[s],c
        )

        t2=time.perf_counter()

        with self.lock:

            self.generated += 1
            self.gen_s += t1-t0
            self.copy_s += t2-t1


def run_case(name, hybrid):

    bank=SharedBank()

    q=queue.Queue(
        maxsize=SLOTS
    )

    results=[None]*NREQ

    next_job=0
    job_lock=threading.Lock()

    metric_lock=threading.Lock()

    total_jobs=0
    total_ni=0

    all_samples=[]

    pinned_samples=[]
    free_samples=[]

    pinned_jobs=0
    free_jobs=0

    # ========================================================
    # X512 workers
    # ========================================================

    def worker(wid):

        nonlocal total_jobs
        nonlocal total_ni
        nonlocal pinned_jobs
        nonlocal free_jobs

        is_pinned=(
            hybrid
            and wid < PINNED_WORKERS
        )

        if is_pinned:

            cpu=BEST_CPUS[
                wid % len(BEST_CPUS)
            ]

            pin(cpu)

        core=X512Core()

        local_jobs=0
        local_ni=0
        local_samples=[]

        while True:

            item=q.get()

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

            dt=t1-t0

            local_samples.append(dt)

            local_jobs += 1
            local_ni += ni

            bank.release(slot)

        with metric_lock:

            total_jobs += local_jobs
            total_ni += local_ni

            all_samples.extend(
                local_samples
            )

            if is_pinned:

                pinned_jobs += local_jobs

                pinned_samples.extend(
                    local_samples
                )

            else:

                free_jobs += local_jobs

                free_samples.extend(
                    local_samples
                )

    # ========================================================
    # Generators — NEVER PINNED
    # ========================================================

    def generator():

        nonlocal next_job

        while True:

            with job_lock:

                if next_job >= NREQ:
                    return

                jid=next_job
                next_job += 1

            slot=bank.acquire()

            bank.fill(
                slot,
                SEED+jid
            )

            q.put(
                (jid,slot)
            )

    workers=[]

    for wid in range(WORKERS):

        t=threading.Thread(
            target=worker,
            args=(wid,)
        )

        workers.append(t)
        t.start()

    generators=[
        threading.Thread(
            target=generator
        )
        for _ in range(GENERATORS)
    ]

    t0=time.perf_counter()

    for g in generators:
        g.start()

    for g in generators:
        g.join()

    for _ in workers:
        q.put(None)

    for t in workers:
        t.join()

    wall=time.perf_counter()-t0

    # ========================================================
    # Statistics
    # ========================================================

    def stats(samples):

        if not samples:
            return None

        a=np.asarray(
            samples,
            dtype=np.float64
        ) / 1000.0

        return {
            "mean":float(np.mean(a)),
            "median":float(np.median(a)),
            "min":float(np.min(a)),
            "p95":float(
                np.percentile(a,95)
            ),
            "p99":float(
                np.percentile(a,99)
            ),
            "max":float(np.max(a))
        }

    allstat=stats(all_samples)
    pinstat=stats(pinned_samples)
    freestat=stats(free_samples)

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

    req_s=NREQ/wall

    ginstr_wall=(
        total_ni /
        wall /
        1e9
    )

    tpop_s=(
        sum(all_samples) /
        1e9
    )

    ginstr_active=(
        total_ni /
        tpop_s /
        1e9
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
        "all":allstat,
        "pinned":pinstat,
        "free":freestat,
        "pinned_jobs":pinned_jobs,
        "free_jobs":free_jobs,
        "gen_s":bank.gen_s,
        "copy_s":bank.copy_s,
        "sig":sig,
        "valid":valid
    }


print("="*78)
print("DELTA X512 — HYBRID4 LATENCY/THROUGHPUT 4550 V2")
print("="*78)

print(
    f"REFERENCE_DELTA="
    f"{DELTA_REFERENCE_MHZ:.0f}_MHz"
)

print(
    f"TARGET_DELTA="
    f"{DELTA_TARGET_MHZ:.0f}_MHz"
)

print(
    f"PIPELINE_TARGET="
    f"{TARGET_GBIT:.3f}_Gbitop_s"
)

print(
    f"TOTAL_WORKERS={WORKERS}"
)

print(
    f"HYBRID_PINNED_WORKERS="
    f"{PINNED_WORKERS}"
)

print(
    f"HYBRID_FREE_WORKERS="
    f"{WORKERS-PINNED_WORKERS}"
)

print(
    "GENERATORS=2_FREE"
)

print()


print("RUNNING_UNPINNED_REFERENCE...")

ref=run_case(
    "UNPINNED",
    False
)

time.sleep(0.5)

print(
    "RUNNING_HYBRID4..."
)

hyb=run_case(
    "HYBRID4",
    True
)

print()
print("="*78)
print("RESULTS")
print("="*78)

for r in [ref,hyb]:

    s=r["all"]

    print()
    print(
        f"[{r['name']}]"
    )

    print(
        f"WALL={r['wall']:.6f}s"
    )

    print(
        f"REQUESTS_S="
        f"{r['req_s']:.2f}"
    )

    print(
        f"GBITOP_S="
        f"{r['gbit']:.3f}"
    )

    print(
        f"GINSTR_WALL="
        f"{r['ginstr_wall']:.6f}"
    )

    print(
        f"GINSTR_ACTIVE="
        f"{r['ginstr_active']:.6f}"
    )

    print(
        f"TPOP_MIN_US="
        f"{s['min']:.3f}"
    )

    print(
        f"TPOP_MEDIAN_US="
        f"{s['median']:.3f}"
    )

    print(
        f"TPOP_MEAN_US="
        f"{s['mean']:.3f}"
    )

    print(
        f"TPOP_P95_US="
        f"{s['p95']:.3f}"
    )

    print(
        f"TPOP_P99_US="
        f"{s['p99']:.3f}"
    )

    print(
        f"TPOP_MAX_US="
        f"{s['max']:.3f}"
    )

    print(
        f"GENERATION_SUM_S="
        f"{r['gen_s']:.6f}"
    )

    print(
        f"COPY_SUM_S="
        f"{r['copy_s']:.6f}"
    )

    print(
        f"SIG={r['sig']}"
    )

    print(
        "VALIDATION="
        + (
            "PASS"
            if r["valid"]
            else "FAIL"
        )
    )

    if r["pinned"]:

        print()
        print("  PINNED SUBSET")

        print(
            f"  JOBS="
            f"{r['pinned_jobs']}"
        )

        print(
            f"  MEAN_US="
            f"{r['pinned']['mean']:.3f}"
        )

        print(
            f"  MEDIAN_US="
            f"{r['pinned']['median']:.3f}"
        )

        print(
            f"  P95_US="
            f"{r['pinned']['p95']:.3f}"
        )

    if (
        r["free"]
        and r["name"]=="HYBRID16"
    ):

        print()
        print("  FREE SUBSET")

        print(
            f"  JOBS="
            f"{r['free_jobs']}"
        )

        print(
            f"  MEAN_US="
            f"{r['free']['mean']:.3f}"
        )

        print(
            f"  MEDIAN_US="
            f"{r['free']['median']:.3f}"
        )

        print(
            f"  P95_US="
            f"{r['free']['p95']:.3f}"
        )


# ============================================================
# A/B verdict
# ============================================================

throughput_gain=(
    hyb["gbit"] /
    ref["gbit"]
)

latency_gain=(
    ref["all"]["mean"] /
    hyb["all"]["mean"]
)

p95_gain=(
    ref["all"]["p95"] /
    hyb["all"]["p95"]
)

target_pass=(
    hyb["gbit"] >= TARGET_GBIT
)

# HYBRID is accepted only if it doesn't trade throughput
# for latency and remains correct.
hybrid_better=(
    hyb["valid"]
    and
    hyb["gbit"] >= ref["gbit"]
    and
    hyb["all"]["mean"]
        <= ref["all"]["mean"]
)

print()
print("="*78)
print("HYBRID VERDICT")
print("="*78)

print(
    f"THROUGHPUT_GAIN="
    f"{throughput_gain:.6f}x"
)

print(
    f"MEAN_LATENCY_GAIN="
    f"{latency_gain:.6f}x"
)

print(
    f"P95_LATENCY_GAIN="
    f"{p95_gain:.6f}x"
)

print(
    "CHECKSUM="
    + (
        "IDENTICAL"
        if hyb["sig"]==EXPECTED_SIG
        else "FAIL"
    )
)

print(
    "HYBRID_NO_REGRESSION="
    + (
        "PASS"
        if hybrid_better
        else "FAIL"
    )
)

print(
    "4550_PIPELINE_THRESHOLD="
    + (
        "PASS"
        if target_pass
        else "FAIL"
    )
)

if hybrid_better:
    winner="HYBRID4"
else:
    winner="UNPINNED"

if (
    hybrid_better
    and target_pass
):
    state="DELTA_4550_VALIDATED"
else:
    state="DELTA_4450_REMAINS_REFERENCE"

print(
    f"WINNER={winner}"
)

print(
    f"STATE={state}"
)

print("="*78)
