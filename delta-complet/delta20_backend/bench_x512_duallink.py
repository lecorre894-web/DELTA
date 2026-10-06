#!/usr/bin/env python3

import os
import time
import hashlib
import threading
import numpy as np

from delta_x512 import DeltaX512

CPU = os.cpu_count() or 32

WORKERS_A = CPU
WORKERS_B = CPU

NREQ = 1024
NW = 16384 * 4
SEED = 894

# Mémoire commune bornée :
# catalogue unique des travaux, partagé par A et B.
SHARED_MEMORY_MAX = NREQ


def signature(values):
    return hashlib.sha256(
        repr(values).encode()
    ).hexdigest()[:16]


# ============================================================
# SHARED MEMORY / LINK
# ============================================================

class X512SharedMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.lock = threading.Lock()

        self.requests = [None] * capacity
        self.results = [None] * capacity

        self.next_job = 0

        self.jobs_a = 0
        self.jobs_b = 0


    def load(self):

        for i in range(self.capacity):

            key = hashlib.sha256(
                f"X512-LINK-{SEED}-{i}".encode()
            ).hexdigest()

            # Une seule description du travail existe.
            self.requests[i] = (
                key,
                SEED + i,
                NW
            )


    def acquire_block(self, block):

        with self.lock:

            if self.next_job >= self.capacity:
                return None

            start = self.next_job

            end = min(
                start + block,
                self.capacity
            )

            self.next_job = end

            return start, end, self.requests[start:end]


    def commit(self, engine, start, values):

        with self.lock:

            for offset, value in enumerate(values):
                self.results[start + offset] = value

            if engine == "A":
                self.jobs_a += len(values)
            else:
                self.jobs_b += len(values)


# ============================================================
# PROCESSOR WRAPPER
# ============================================================

class LinkedProcessor:

    def __init__(
        self,
        name,
        workers,
        memory,
        block=32
    ):

        self.name = name
        self.memory = memory
        self.block = block

        # X512 complet.
        self.x512 = DeltaX512(
            workers,
            "micro"
        )

        self.calc = 0
        self.ni = 0
        self.core_time = 0.0


    def run(self):

        while True:

            job = self.memory.acquire_block(
                self.block
            )

            if job is None:
                break

            start, end, reqs = job

            res, calc, ni, tc = \
                self.x512.compute(reqs)

            self.calc += calc
            self.ni += ni
            self.core_time += tc

            self.memory.commit(
                self.name,
                start,
                res
            )


    def close(self):
        self.x512.close()


# ============================================================
# DATASET COMMUN
# ============================================================

MEM = X512SharedMemory(
    SHARED_MEMORY_MAX
)

MEM.load()

ALL = MEM.requests


print("=" * 76)
print("DELTA X512 DUAL-LINK + SHARED MEMORY")
print("=" * 76)

print(f"CPU_LOGICAL={CPU}")
print(f"X512_A_WORKERS={WORKERS_A}")
print(f"X512_B_WORKERS={WORKERS_B}")
print(
    f"TOTAL_X512_WORKERS="
    f"{WORKERS_A + WORKERS_B}"
)
print(f"SHARED_MEMORY_JOBS={NREQ}")
print(f"WORDS_PER_JOB={NW}")

print()


# ============================================================
# REFERENCE SINGLE
# ============================================================

S = DeltaX512(
    CPU,
    "micro"
)

t0 = time.perf_counter()

single_res, single_calc, single_ni, single_tc = \
    S.compute(ALL)

single_wall = time.perf_counter() - t0

S.close()

single_sig = signature(single_res)


# ============================================================
# RESET LINK
# ============================================================

MEM.next_job = 0
MEM.results = [None] * NREQ
MEM.jobs_a = 0
MEM.jobs_b = 0


# ============================================================
# CREATE TWO COMPLETE X512 PROCESSORS
# ============================================================

A = LinkedProcessor(
    "A",
    WORKERS_A,
    MEM,
    block=32
)

B = LinkedProcessor(
    "B",
    WORKERS_B,
    MEM,
    block=32
)


# ============================================================
# LINK EXECUTION
# ============================================================

TA = threading.Thread(
    target=A.run,
    name="X512-A"
)

TB = threading.Thread(
    target=B.run,
    name="X512-B"
)

t0 = time.perf_counter()

TA.start()
TB.start()

TA.join()
TB.join()

link_wall = time.perf_counter() - t0


A.close()
B.close()


linked_res = MEM.results

linked_calc = A.calc + B.calc
linked_ni = A.ni + B.ni

linked_sig = signature(
    linked_res
)


# ============================================================
# VALIDATION
# ============================================================

identical = (
    single_res == linked_res
    and
    single_sig == linked_sig
    and
    single_calc == linked_calc
    and
    single_ni == linked_ni
)


bitops = (
    3.0 *
    NREQ *
    NW *
    64
)

single_gbit = (
    bitops /
    single_wall /
    1e9
)

link_gbit = (
    bitops /
    link_wall /
    1e9
)

gain = (
    single_wall /
    link_wall
)


# ============================================================
# RESULTS
# ============================================================

print("--------------- SINGLE X512 ----------------")

print(
    f"WALL={single_wall:.6f}s"
)

print(
    f"GBITOP_S={single_gbit:.3f}"
)

print(
    f"CALC={single_calc}"
)

print(
    f"NI={single_ni}"
)

print(
    f"SIG={single_sig}"
)

print()


print("----------- DUAL X512 LINKED ---------------")

print(
    f"WALL={link_wall:.6f}s"
)

print(
    f"GBITOP_S={link_gbit:.3f}"
)

print(
    f"X512_A_JOBS={MEM.jobs_a}"
)

print(
    f"X512_B_JOBS={MEM.jobs_b}"
)

print(
    f"TOTAL_JOBS="
    f"{MEM.jobs_a + MEM.jobs_b}"
)

print(
    f"X512_A_CALC={A.calc}"
)

print(
    f"X512_B_CALC={B.calc}"
)

print(
    f"TOTAL_CALC={linked_calc}"
)

print(
    f"NI={linked_ni}"
)

print(
    f"SIG={linked_sig}"
)

print()


print("================ VERDICT ===================")

print(
    "CHECKSUM="
    + (
        "IDENTICAL"
        if identical
        else "FAIL"
    )
)

print(
    "LINK_MEMORY="
    + (
        "PASS"
        if None not in linked_res
        else "FAIL"
    )
)

print(
    f"SINGLE_GBITOP_S="
    f"{single_gbit:.3f}"
)

print(
    f"DUAL_LINK_GBITOP_S="
    f"{link_gbit:.3f}"
)

print(
    f"GAIN_LINK="
    f"{gain:.4f}x"
)


if not identical:

    winner = "INVALID"

elif gain > 1.02:

    winner = "X512_DUAL_LINK"

elif gain < 0.98:

    winner = "X512_SINGLE"

else:

    winner = "TIE"


print(
    f"WINNER={winner}"
)

print(
    "DUAL_LINK_VALIDATION="
    + (
        "PASS"
        if identical
        else "FAIL"
    )
)

print("============================================")
