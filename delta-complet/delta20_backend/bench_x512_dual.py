#!/usr/bin/env python3

import os
import time
import hashlib
import numpy as np

from delta_x512 import DeltaX512
from delta_x512_dual import DeltaX512Dual

CPU = os.cpu_count() or 1

TOTAL_WORKERS = min(32, CPU)
WA = TOTAL_WORKERS // 2
WB = TOTAL_WORKERS - WA

NW = 16384 * 4
NREQ = 512
SEED = 894

def sig(x):
    return hashlib.sha256(
        repr(x).encode()
    ).hexdigest()[:16]

def make_requests():
    reqs = []

    # Toutes les clés sont uniques :
    # le benchmark mesure le calcul et non les hits du cache.
    for i in range(NREQ):
        key = hashlib.sha256(
            f"DUAL-X512-{SEED}-{i}".encode()
        ).hexdigest()

        reqs.append(
            (key, SEED+i, NW)
        )

    return reqs


reqs = make_requests()

print("=" * 72)
print("DELTA X512 — SINGLE vs DUAL")
print("=" * 72)
print(f"CPU_LOGICAL={CPU}")
print(f"TOTAL_WORKERS={TOTAL_WORKERS}")
print(f"DUAL_A={WA}")
print(f"DUAL_B={WB}")
print(f"REQUESTS={NREQ}")
print(f"WORDS_PER_REQUEST={NW}")
print()


# ------------------------------------------------------------
# SINGLE
# ------------------------------------------------------------

S = DeltaX512(TOTAL_WORKERS, "micro")

t0 = time.perf_counter()

single_res, single_calc, single_ni, single_tc = S.compute(reqs)

single_wall = time.perf_counter() - t0

S.close()

single_sig = sig(single_res)


# ------------------------------------------------------------
# DUAL
# ------------------------------------------------------------

D = DeltaX512Dual(WA, WB, "micro")

dual_res, dual_calc, dual_ni, dual_tc, dual_wall = D.compute(reqs)

D.close()

dual_sig = sig(dual_res)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

identical = (
    single_res == dual_res
    and single_sig == dual_sig
)

gain = (
    single_wall / dual_wall
    if dual_wall > 0
    else 0.0
)

single_req_s = (
    NREQ / single_wall
    if single_wall > 0
    else 0.0
)

dual_req_s = (
    NREQ / dual_wall
    if dual_wall > 0
    else 0.0
)

# Le noyau X512 existant compte 3 opérations bit
# pour chaque bit de ((A&B)^C).
bitops = 3.0 * NREQ * NW * 64

single_gbit = (
    bitops / single_wall / 1e9
    if single_wall > 0
    else 0.0
)

dual_gbit = (
    bitops / dual_wall / 1e9
    if dual_wall > 0
    else 0.0
)


print("--------------- SINGLE ----------------")
print(f"WALL={single_wall:.6f}s")
print(f"REQUESTS_S={single_req_s:.2f}")
print(f"GBITOP_S={single_gbit:.3f}")
print(f"CALC={single_calc}")
print(f"NI={single_ni}")
print(f"SIG={single_sig}")

print()

print("---------------- DUAL -----------------")
print(f"WALL={dual_wall:.6f}s")
print(f"REQUESTS_S={dual_req_s:.2f}")
print(f"GBITOP_S={dual_gbit:.3f}")
print(f"CALC={dual_calc}")
print(f"NI={dual_ni}")
print(f"SIG={dual_sig}")

print()

print("================ VERDICT ==============")
print(
    "CHECKSUM="
    + ("IDENTICAL" if identical else "FAIL")
)

print(f"GAIN_DUAL={gain:.4f}x")

if not identical:
    winner = "INVALID"
elif gain > 1.02:
    winner = "X512_DUAL"
elif gain < 0.98:
    winner = "X512_SINGLE"
else:
    winner = "TIE"

print(f"WINNER={winner}")
print(
    "DUAL_PROCESSOR_VALIDATION="
    + ("PASS" if identical else "FAIL")
)
print("=======================================")
