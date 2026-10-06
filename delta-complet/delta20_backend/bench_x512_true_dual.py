#!/usr/bin/env python3

import os
import time
import hashlib
import threading

from delta_x512 import DeltaX512

CPU = os.cpu_count() or 32

WORKERS_PER_PROCESSOR = CPU       # 32 + 32
NREQ_TOTAL = 1024
NREQ_EACH = NREQ_TOTAL // 2
NW = 16384 * 4
SEED = 894


def signature(values):
    return hashlib.sha256(
        repr(values).encode()
    ).hexdigest()[:16]


def make_requests(prefix, start, count):
    reqs=[]

    for i in range(count):
        n=start+i

        key=hashlib.sha256(
            f"DUCB-{prefix}-{SEED}-{n}".encode()
        ).hexdigest()

        reqs.append(
            (key, SEED+n, NW)
        )

    return reqs


# ============================================================
# IMPORTANT
#
# SINGLE doit effectuer exactement autant de travail total
# que A+B réunis.
#
# SINGLE : 1024 requêtes / 32 workers
#
# DUAL :
#    A : 512 requêtes / 32 workers
#    B : 512 requêtes / 32 workers
#
# => même dataset total, mêmes 1024 calculs.
# ============================================================

ALL = make_requests("COMMON",0,NREQ_TOTAL)

A_REQ = ALL[:NREQ_EACH]
B_REQ = ALL[NREQ_EACH:]


print("="*76)
print("DELTA X512 TRUE DUAL — SINGLE32 vs DUAL32+32")
print("="*76)

print(f"CPU_LOGICAL={CPU}")
print(f"SINGLE_WORKERS={WORKERS_PER_PROCESSOR}")
print(
    f"DUAL_WORKERS="
    f"{WORKERS_PER_PROCESSOR}+{WORKERS_PER_PROCESSOR}"
)
print(f"TOTAL_SOFTWARE_WORKERS={2*WORKERS_PER_PROCESSOR}")
print(f"TOTAL_REQUESTS={NREQ_TOTAL}")
print(f"WORDS_PER_REQUEST={NW}")
print()


# ============================================================
# SINGLE X512-A
# ============================================================

S=DeltaX512(
    WORKERS_PER_PROCESSOR,
    "micro"
)

t0=time.perf_counter()

sres,scalc,sni,stc=S.compute(ALL)

single_wall=time.perf_counter()-t0

S.close()

single_sig=signature(sres)


# ============================================================
# TRUE DUAL
#
# Deux DeltaX512 complets.
# Chacun crée ses 32 workers et ses 32 contextes X512.
# ============================================================

A=DeltaX512(
    WORKERS_PER_PROCESSOR,
    "micro"
)

B=DeltaX512(
    WORKERS_PER_PROCESSOR,
    "micro"
)

outA=None
outB=None
errA=None
errB=None


def runA():
    global outA,errA

    try:
        outA=A.compute(A_REQ)
    except Exception as e:
        errA=e


def runB():
    global outB,errB

    try:
        outB=B.compute(B_REQ)
    except Exception as e:
        errB=e


ta=threading.Thread(target=runA)
tb=threading.Thread(target=runB)

t0=time.perf_counter()

ta.start()
tb.start()

ta.join()
tb.join()

dual_wall=time.perf_counter()-t0


if errA:
    raise errA

if errB:
    raise errB


ares,acalc,ani,atc=outA
bres,bcalc,bni,btc=outB

A.close()
B.close()


# Reconstitution dans exactement le même ordre que SINGLE.
dual_res=ares+bres

dual_sig=signature(dual_res)

dual_calc=acalc+bcalc
dual_ni=ani+bni


# ============================================================
# VALIDATION
# ============================================================

identical=(
    sres==dual_res
    and
    single_sig==dual_sig
    and
    scalc==dual_calc
    and
    sni==dual_ni
)

gain=(
    single_wall/dual_wall
    if dual_wall>0
    else 0.0
)

single_req_s=NREQ_TOTAL/single_wall
dual_req_s=NREQ_TOTAL/dual_wall

bitops=3.0*NREQ_TOTAL*NW*64

single_gbit=bitops/single_wall/1e9
dual_gbit=bitops/dual_wall/1e9


print("--------------- SINGLE X512-A ----------------")

print(
    f"WORKERS={WORKERS_PER_PROCESSOR}"
)

print(
    f"WALL={single_wall:.6f}s"
)

print(
    f"REQUESTS_S={single_req_s:.2f}"
)

print(
    f"GBITOP_S={single_gbit:.3f}"
)

print(
    f"CALC={scalc}"
)

print(
    f"NI={sni}"
)

print(
    f"SIG={single_sig}"
)


print()

print("------------ TRUE DUAL X512-A + X512-B --------")

print(
    f"WORKERS="
    f"{WORKERS_PER_PROCESSOR}+{WORKERS_PER_PROCESSOR}"
)

print(
    f"WALL={dual_wall:.6f}s"
)

print(
    f"REQUESTS_S={dual_req_s:.2f}"
)

print(
    f"GBITOP_S={dual_gbit:.3f}"
)

print(
    f"A_CALC={acalc}"
)

print(
    f"B_CALC={bcalc}"
)

print(
    f"TOTAL_CALC={dual_calc}"
)

print(
    f"NI={dual_ni}"
)

print(
    f"SIG={dual_sig}"
)


print()

print("=================== VERDICT ===================")

print(
    "CHECKSUM="
    + ("IDENTICAL" if identical else "FAIL")
)

print(
    f"SINGLE_GBITOP_S={single_gbit:.3f}"
)

print(
    f"DUAL_GBITOP_S={dual_gbit:.3f}"
)

print(
    f"GAIN_DUAL={gain:.4f}x"
)

if not identical:
    winner="INVALID"

elif gain>1.02:
    winner="X512_TRUE_DUAL"

elif gain<0.98:
    winner="X512_SINGLE"

else:
    winner="TIE"

print(
    f"WINNER={winner}"
)

print(
    "TRUE_DUAL_VALIDATION="
    + ("PASS" if identical else "FAIL")
)

print("================================================")
