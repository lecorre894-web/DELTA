#!/usr/bin/env python3

import os
import time
import queue
import hashlib
import threading
import numpy as np

from delta_x512 import X512Core, aligne, vecs

CPU = os.cpu_count() or 32

# ============================================================
# DELTA X512 DUAL-LINK + 2x TPOP
#
# X512-A : TPOP-A1 + TPOP-A2
# X512-B : TPOP-B1 + TPOP-B2
#
# Chaque banque TPOP possède CPU contextes X512 indépendants.
# Total logique = 4 * CPU contextes.
# ============================================================

TPOP_WORKERS = CPU

NREQ = 1024
NW = 16384 * 4
SEED = 894

MEMORY_SLOTS = 8

REFERENCE_SINGLE_GBIT = 5.982
REFERENCE_DUALLINK_GBIT = 7.411
REFERENCE_DUALLINK_GAIN = 1.2389


def signature(values):
    return hashlib.sha256(
        repr(values).encode()
    ).hexdigest()[:16]


# ============================================================
# SHARED MEMORY LINK
# ============================================================

class SharedBank:

    def __init__(self, slots, nw):

        self.slots = slots
        self.nw = nw

        self.A = []
        self.B = []
        self.C = []

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

        self.free = queue.Queue()

        for i in range(slots):
            self.free.put(i)

        self.generated = 0
        self.lock = threading.Lock()


    def acquire(self):
        return self.free.get()


    def release(self, slot):
        self.free.put(slot)


    def fill(self, slot, seed):

        a,b,c = vecs(
            seed,
            self.nw
        )

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

        with self.lock:
            self.generated += 1


# ============================================================
# ONE TPOP BANK
# ============================================================

class TPopBank:

    def __init__(
        self,
        name,
        workers,
        shared,
        workq,
        results
    ):

        self.name = name
        self.workers = workers
        self.shared = shared
        self.workq = workq
        self.results = results

        # IMPORTANT:
        # chaque voie possède ses propres contextes.
        self.cores = [
            X512Core()
            for _ in range(workers)
        ]

        self.threads = []

        self.jobs = 0
        self.ni = 0
        self.seconds = 0.0

        self.lock = threading.Lock()


    def worker(self, wid):

        core = self.cores[wid]

        while True:

            item = self.workq.get()

            if item is None:
                break

            job_id, slot = item

            A = self.shared.A[slot]
            B = self.shared.B[slot]
            C = self.shared.C[slot]

            t0 = time.perf_counter()

            result, ni = core.tpop(
                A,
                B,
                C,
                prog="micro"
            )

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            self.results[job_id] = result

            with self.lock:

                self.jobs += 1
                self.ni += ni
                self.seconds += elapsed

            self.shared.release(
                slot
            )


    def start(self):

        for wid in range(
            self.workers
        ):

            t = threading.Thread(
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
# CREATE SHARED LINK
# ============================================================

shared = SharedBank(
    MEMORY_SLOTS,
    NW
)

workq = queue.Queue(
    maxsize=MEMORY_SLOTS
)

results = [None] * NREQ


# ============================================================
# FOUR TPOP BANKS
# ============================================================

banks = [

    TPopBank(
        "TPOP-A1",
        TPOP_WORKERS,
        shared,
        workq,
        results
    ),

    TPopBank(
        "TPOP-A2",
        TPOP_WORKERS,
        shared,
        workq,
        results
    ),

    TPopBank(
        "TPOP-B1",
        TPOP_WORKERS,
        shared,
        workq,
        results
    ),

    TPopBank(
        "TPOP-B2",
        TPOP_WORKERS,
        shared,
        workq,
        results
    )
]

TOTAL_CONTEXTS = (
    len(banks)
    *
    TPOP_WORKERS
)


# ============================================================
# PRODUCER
# ============================================================

def producer():

    for job_id in range(NREQ):

        slot = shared.acquire()

        shared.fill(
            slot,
            SEED + job_id
        )

        workq.put(
            (job_id, slot)
        )

    # Un arrêt par consommateur.
    for _ in range(
        TOTAL_CONTEXTS
    ):
        workq.put(None)


# ============================================================
# RUN
# ============================================================

print("=" * 78)
print(" DELTA X512 DUAL-LINK + 2x TPOP PER PROCESSOR")
print("=" * 78)

print(
    f"CPU_LOGICAL={CPU}"
)

print(
    f"TPOP_A1_CONTEXTS={TPOP_WORKERS}"
)

print(
    f"TPOP_A2_CONTEXTS={TPOP_WORKERS}"
)

print(
    f"TPOP_B1_CONTEXTS={TPOP_WORKERS}"
)

print(
    f"TPOP_B2_CONTEXTS={TPOP_WORKERS}"
)

print(
    f"TOTAL_LOGICAL_CONTEXTS={TOTAL_CONTEXTS}"
)

print(
    f"SHARED_MEMORY_SLOTS={MEMORY_SLOTS}"
)

print(
    f"REQUESTS={NREQ}"
)

print(
    f"WORDS_PER_REQUEST={NW}"
)

print()


for bank in banks:
    bank.start()


P = threading.Thread(
    target=producer,
    name="X512-LINK-PRODUCER"
)


t0 = time.perf_counter()

P.start()

P.join()

for bank in banks:
    bank.join()

wall = (
    time.perf_counter()
    -
    t0
)


# ============================================================
# COLLECT
# ============================================================

total_jobs = sum(
    b.jobs
    for b in banks
)

total_ni = sum(
    b.ni
    for b in banks
)

sig = signature(
    results
)


# ============================================================
# INDEPENDENT CHECKSUM REFERENCE
#
# Recalcule séquentiellement la valeur attendue de chaque
# travail avec un X512Core distinct.
# ============================================================

ref_core = X512Core()

reference = []

for job_id in range(NREQ):

    a,b,c = vecs(
        SEED + job_id,
        NW
    )

    result, ni = ref_core.tpop(
        a,
        b,
        c,
        prog="micro"
    )

    reference.append(
        result
    )

ref_sig = signature(
    reference
)

checksum_ok = (
    results == reference
    and
    sig == ref_sig
)


# ============================================================
# PERFORMANCE
# ============================================================

bitops = (
    3.0
    *
    NREQ
    *
    NW
    *
    64
)

gbit = (
    bitops
    /
    wall
    /
    1e9
)

gain_vs_single = (
    gbit
    /
    REFERENCE_SINGLE_GBIT
)

gain_vs_v2 = (
    gbit
    /
    REFERENCE_DUALLINK_GBIT
)


# ============================================================
# VALIDATION
# ============================================================

generation_ok = (
    shared.generated
    ==
    NREQ
)

jobs_ok = (
    total_jobs
    ==
    NREQ
)

memory_ok = (
    None not in results
)

banks_ok = all(
    b.jobs > 0
    for b in banks
)

valid = (
    checksum_ok
    and
    generation_ok
    and
    jobs_ok
    and
    memory_ok
    and
    banks_ok
)


# ============================================================
# REPORT
# ============================================================

print("--------------- TPOP DISTRIBUTION ----------------")

for b in banks:

    print(
        f"{b.name:8s} "
        f"JOBS={b.jobs:4d} "
        f"NI={b.ni:10d} "
        f"CORE_S={b.seconds:.6f}"
    )

print()

print(
    f"TOTAL_JOBS={total_jobs}"
)

print(
    f"TOTAL_NI={total_ni}"
)

print(
    f"INPUT_GENERATIONS={shared.generated}"
)

print(
    f"SIG={sig}"
)

print(
    f"REFERENCE_SIG={ref_sig}"
)

print()


print("================ PERFORMANCE =================")

print(
    f"WALL={wall:.6f}s"
)

print(
    f"DUAL_LINK_2XTPOP_GBITOP_S="
    f"{gbit:.3f}"
)

print(
    f"REFERENCE_SINGLE_GBITOP_S="
    f"{REFERENCE_SINGLE_GBIT:.3f}"
)

print(
    f"REFERENCE_DUALLINK_V2_GBITOP_S="
    f"{REFERENCE_DUALLINK_GBIT:.3f}"
)

print(
    f"GAIN_VS_SINGLE="
    f"{gain_vs_single:.4f}x"
)

print(
    f"GAIN_VS_DUALLINK_V2="
    f"{gain_vs_v2:.4f}x"
)

print()


print("================ VALIDATION ==================")

print(
    "CHECKSUM="
    + (
        "IDENTICAL"
        if checksum_ok
        else "FAIL"
    )
)

print(
    "SHARED_MEMORY="
    + (
        "PASS"
        if memory_ok
        else "FAIL"
    )
)

print(
    "SINGLE_INPUT_GENERATION="
    + (
        "PASS"
        if generation_ok
        else "FAIL"
    )
)

for b in banks:

    print(
        f"{b.name}_STATUS="
        + (
            "PASS"
            if b.jobs > 0
            else "FAIL"
        )
    )

print(
    "DUAL_LINK_2XTPOP_VALIDATION="
    + (
        "PASS"
        if valid
        else "FAIL"
    )
)


# ============================================================
# WINNER — TOUJOURS EN DERNIER
# ============================================================

if not valid:

    winner = "INVALID"

elif (
    gbit >
    REFERENCE_DUALLINK_GBIT * 1.02
):

    winner = "X512_DUAL_LINK_2XTPOP"

elif (
    REFERENCE_DUALLINK_GBIT >
    REFERENCE_SINGLE_GBIT * 1.02
):

    winner = "X512_DUAL_LINK_V2"

else:

    winner = "X512_SINGLE"


print()
print("=" * 78)
print(
    "WINNER="
    + winner
)
print("=" * 78)
