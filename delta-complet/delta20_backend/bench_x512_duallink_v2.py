#!/usr/bin/env python3

import os
import time
import queue
import hashlib
import threading
import numpy as np

from delta_x512 import X512Core, aligne, vecs

CPU = os.cpu_count() or 32

# Deux processeurs X512 complets au sens demandé.
WORKERS_A = CPU
WORKERS_B = CPU
TOTAL_CONTEXTS = WORKERS_A + WORKERS_B

NREQ = 1024
NW = 16384 * 4
SEED = 894

# Mémoire commune bornée.
# 8 slots × 3 vecteurs × 65536 × 8 octets ~= 12 MiB.
MEMORY_SLOTS = 8


def signature(values):
    return hashlib.sha256(
        repr(values).encode()
    ).hexdigest()[:16]


# ============================================================
# SHARED MEMORY BANK
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
                aligne(np.empty(nw, dtype=np.uint64))
            )
            self.B.append(
                aligne(np.empty(nw, dtype=np.uint64))
            )
            self.C.append(
                aligne(np.empty(nw, dtype=np.uint64))
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

        a,b,c = vecs(seed, self.nw)

        # Une seule génération.
        np.copyto(self.A[slot], a)
        np.copyto(self.B[slot], b)
        np.copyto(self.C[slot], c)

        with self.lock:
            self.generated += 1


# ============================================================
# GENERATOR
#
# Produit chaque dataset UNE fois puis le place dans une file
# commune consommée par A ou B.
# ============================================================

def generator(bank, workq):

    for job_id in range(NREQ):

        slot = bank.acquire()

        bank.fill(
            slot,
            SEED + job_id
        )

        workq.put(
            (job_id, slot)
        )

    # 64 consommateurs au total.
    for _ in range(TOTAL_CONTEXTS):
        workq.put(None)


# ============================================================
# X512 PROCESSOR
#
# Chaque processeur possède exactement CPU X512Core indépendants.
# ============================================================

class X512Processor:

    def __init__(
        self,
        name,
        workers,
        bank,
        workq,
        results
    ):

        self.name = name
        self.workers = workers
        self.bank = bank
        self.workq = workq
        self.results = results

        self.cores = [
            X512Core()
            for _ in range(workers)
        ]

        self.threads = []

        self.jobs = 0
        self.ni = 0
        self.core_seconds = 0.0

        self.lock = threading.Lock()


    def worker(self, wid):

        # Le thread logique pilote son propre contexte X512.
        core = self.cores[wid]

        while True:

            item = self.workq.get()

            if item is None:
                break

            job_id, slot = item

            A = self.bank.A[slot]
            B = self.bank.B[slot]
            C = self.bank.C[slot]

            t0 = time.perf_counter()

            result, ni = core.tpop(
                A,
                B,
                C,
                prog="micro"
            )

            elapsed = (
                time.perf_counter() - t0
            )

            self.results[job_id] = result

            with self.lock:

                self.jobs += 1
                self.ni += ni
                self.core_seconds += elapsed

            # Seulement après consommation,
            # le slot peut être réutilisé.
            self.bank.release(slot)


    def start(self):

        for wid in range(self.workers):

            t = threading.Thread(
                target=self.worker,
                args=(wid,),
                name=f"X512-{self.name}-{wid}"
            )

            self.threads.append(t)
            t.start()


    def join(self):

        for t in self.threads:
            t.join()


# ============================================================
# REFERENCE SINGLE
#
# Même principe mémoire, mais seulement 32 contextes X512.
# Cela évite de comparer V2 à l'ancien chemin qui régénérait
# les données différemment.
# ============================================================

def run_configuration(
    dual=False
):

    bank = SharedBank(
        MEMORY_SLOTS,
        NW
    )

    workq = queue.Queue(
        maxsize=MEMORY_SLOTS
    )

    results = [None] * NREQ

    A = X512Processor(
        "A",
        WORKERS_A,
        bank,
        workq,
        results
    )

    B = None

    if dual:

        B = X512Processor(
            "B",
            WORKERS_B,
            bank,
            workq,
            results
        )

    # Nombre réel de consommateurs de cette configuration.
    consumers = (
        WORKERS_A +
        (WORKERS_B if dual else 0)
    )

    # Générateur local afin d'envoyer exactement
    # le bon nombre de marqueurs de fin.
    def produce():

        for job_id in range(NREQ):

            slot = bank.acquire()

            bank.fill(
                slot,
                SEED + job_id
            )

            workq.put(
                (job_id, slot)
            )

        for _ in range(consumers):
            workq.put(None)


    producer = threading.Thread(
        target=produce,
        name="X512-SHARED-MEMORY-PRODUCER"
    )

    t0 = time.perf_counter()

    A.start()

    if B is not None:
        B.start()

    producer.start()

    producer.join()

    A.join()

    if B is not None:
        B.join()

    wall = (
        time.perf_counter() - t0
    )

    jobs = A.jobs
    ni = A.ni
    core_seconds = A.core_seconds

    jobs_b = 0
    ni_b = 0
    core_seconds_b = 0.0

    if B is not None:

        jobs_b = B.jobs
        ni_b = B.ni
        core_seconds_b = B.core_seconds

        jobs += jobs_b
        ni += ni_b
        core_seconds += core_seconds_b

    return {
        "wall": wall,
        "results": results,
        "sig": signature(results),

        "generated": bank.generated,

        "jobs_a": A.jobs,
        "jobs_b": jobs_b,

        "ni_a": A.ni,
        "ni_b": ni_b,

        "core_a": A.core_seconds,
        "core_b": core_seconds_b,

        "jobs": jobs,
        "ni": ni,
        "core_seconds": core_seconds
    }


print("=" * 78)
print(" DELTA X512 DUAL-LINK V2 — SHARED DATA MEMORY")
print("=" * 78)

print(
    f"CPU_LOGICAL={CPU}"
)

print(
    f"X512_A_CONTEXTS={WORKERS_A}"
)

print(
    f"X512_B_CONTEXTS={WORKERS_B}"
)

print(
    f"DUAL_CONTEXTS={TOTAL_CONTEXTS}"
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


# ============================================================
# SINGLE
# ============================================================

single = run_configuration(
    dual=False
)


# ============================================================
# DUAL LINK
# ============================================================

dual = run_configuration(
    dual=True
)


# ============================================================
# VALIDATION
# ============================================================

same_results = (
    single["results"]
    ==
    dual["results"]
)

same_sig = (
    single["sig"]
    ==
    dual["sig"]
)

generation_ok = (
    single["generated"] == NREQ
    and
    dual["generated"] == NREQ
)

jobs_ok = (
    single["jobs"] == NREQ
    and
    dual["jobs"] == NREQ
)

ni_ok = (
    single["ni"]
    ==
    dual["ni"]
)

valid = (
    same_results
    and
    same_sig
    and
    generation_ok
    and
    jobs_ok
    and
    ni_ok
)


# ============================================================
# THROUGHPUT
# ============================================================

bitops = (
    3.0 *
    NREQ *
    NW *
    64
)

single_gbit = (
    bitops /
    single["wall"] /
    1e9
)

dual_gbit = (
    bitops /
    dual["wall"] /
    1e9
)

gain = (
    single["wall"]
    /
    dual["wall"]
)


# ============================================================
# OUTPUT
# ============================================================

print("--------------- SINGLE LINK-MEM ----------------")

print(
    f"CONTEXTS={WORKERS_A}"
)

print(
    f"WALL={single['wall']:.6f}s"
)

print(
    f"GBITOP_S={single_gbit:.3f}"
)

print(
    f"INPUT_GENERATIONS={single['generated']}"
)

print(
    f"JOBS={single['jobs']}"
)

print(
    f"NI={single['ni']}"
)

print(
    f"SIG={single['sig']}"
)

print()


print("-------------- DUAL LINK-MEM ------------------")

print(
    f"CONTEXTS={WORKERS_A}+{WORKERS_B}"
)

print(
    f"WALL={dual['wall']:.6f}s"
)

print(
    f"GBITOP_S={dual_gbit:.3f}"
)

print(
    f"INPUT_GENERATIONS={dual['generated']}"
)

print(
    f"X512_A_JOBS={dual['jobs_a']}"
)

print(
    f"X512_B_JOBS={dual['jobs_b']}"
)

print(
    f"TOTAL_JOBS={dual['jobs']}"
)

print(
    f"X512_A_NI={dual['ni_a']}"
)

print(
    f"X512_B_NI={dual['ni_b']}"
)

print(
    f"TOTAL_NI={dual['ni']}"
)

print(
    f"SIG={dual['sig']}"
)

print()


print("=================== VERDICT ===================")

print(
    "CHECKSUM="
    + (
        "IDENTICAL"
        if same_sig and same_results
        else "FAIL"
    )
)

print(
    "SHARED_INPUT_GENERATION="
    + (
        "PASS"
        if generation_ok
        else "FAIL"
    )
)

print(
    "INPUT_GENERATIONS_SINGLE="
    + str(single["generated"])
)

print(
    "INPUT_GENERATIONS_DUAL="
    + str(dual["generated"])
)

print(
    "X512_A_CONTEXTS="
    + str(WORKERS_A)
)

print(
    "X512_B_CONTEXTS="
    + str(WORKERS_B)
)

print(
    "TOTAL_DUAL_CONTEXTS="
    + str(TOTAL_CONTEXTS)
)

print(
    f"SINGLE_GBITOP_S={single_gbit:.3f}"
)

print(
    f"DUAL_LINK_GBITOP_S={dual_gbit:.3f}"
)

print(
    f"GAIN_LINK={gain:.4f}x"
)


if not valid:

    winner = "INVALID"

elif gain > 1.02:

    winner = "X512_DUAL_LINK_V2"

elif gain < 0.98:

    winner = "X512_SINGLE_LINK_MEM"

else:

    winner = "TIE"


print(
    "WINNER="
    + winner
)

print(
    "X512_DUAL_LINK_V2_VALIDATION="
    + (
        "PASS"
        if valid
        else "FAIL"
    )
)

print("================================================")
