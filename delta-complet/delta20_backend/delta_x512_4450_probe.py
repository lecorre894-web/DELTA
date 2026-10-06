#!/usr/bin/env python3

import time
import statistics
import numpy as np

from delta_x512 import X512Core, aligne, vecs

# ============================================================
# DELTA X512 — PALIER 4450
# ============================================================

PHYSICAL_REFERENCE_MHZ = 4300.0

PREVIOUS_DELTA_MHZ = 4400.0
TARGET_DELTA_MHZ   = 4450.0

STEP_MHZ = TARGET_DELTA_MHZ - PREVIOUS_DELTA_MHZ
STEP_RATIO = TARGET_DELTA_MHZ / PREVIOUS_DELTA_MHZ

# Dernière mesure réelle
BASE_MEAN_US = 15.827
BASE_MEDIAN_US = 15.270
BASE_MIN_US = 14.810
BASE_P95_US = 17.396
BASE_P99_US = 28.993
BASE_MAX_US = 53.830

BASE_GINSTR_S = 2.329236

# Cible demandée : latence / 2
TARGET_MEAN_US   = BASE_MEAN_US / 2.0
TARGET_MEDIAN_US = BASE_MEDIAN_US / 2.0

# Si même quantité de travail en moitié moins de temps :
TARGET_GINSTR_S = BASE_GINSTR_S * 2.0

SEED = 894
NW = 16384 * 4

WARMUP = 128
SAMPLES = 5000

core = X512Core()

# ============================================================
# DATASET
# ============================================================

A,B,C = vecs(SEED,NW)

A=aligne(A)
B=aligne(B)
C=aligne(C)

# ============================================================
# WARM-UP
# ============================================================

for _ in range(WARMUP):
    core.tpop(
        A,B,C,
        prog="micro"
    )

# ============================================================
# LATENCY MEASUREMENT
# ============================================================

samples_ns=[]
ni_values=[]
results=[]

for _ in range(SAMPLES):

    t0=time.perf_counter_ns()

    r,ni=core.tpop(
        A,B,C,
        prog="micro"
    )

    t1=time.perf_counter_ns()

    samples_ns.append(t1-t0)
    ni_values.append(ni)
    results.append(r)

a=np.asarray(
    samples_ns,
    dtype=np.float64
)

us=a/1000.0

min_us=float(np.min(us))
median_us=float(np.median(us))
mean_us=float(np.mean(us))
p95_us=float(np.percentile(us,95))
p99_us=float(np.percentile(us,99))
max_us=float(np.max(us))
std_us=float(np.std(us))

mean_ms=mean_us/1000.0

total_ni=sum(ni_values)
total_s=sum(samples_ns)/1e9

ginstr_s=(
    total_ni /
    total_s /
    1e9
)

# ============================================================
# COMPARISONS
# ============================================================

latency_gain=(
    BASE_MEAN_US /
    mean_us
)

latency_reduction_pct=(
    (1.0 - mean_us/BASE_MEAN_US)
    *100.0
)

ginstr_gain=(
    ginstr_s /
    BASE_GINSTR_S
)

equivalent_from_latency=(
    PREVIOUS_DELTA_MHZ *
    latency_gain
)

equivalent_from_ginstr=(
    PREVIOUS_DELTA_MHZ *
    ginstr_gain
)

half_latency_reached=(
    mean_us <= TARGET_MEAN_US
)

target_4450_reached=(
    equivalent_from_ginstr
    >= TARGET_DELTA_MHZ
)

stable_result=(
    len(set(results))==1
)

stable_ni=(
    len(set(ni_values))==1
)

valid=(
    stable_result
    and stable_ni
)

# ============================================================
# REPORT
# ============================================================

print("="*76)
print("DELTA X512 — 4450 MHz EQUIVALENT PROCESSING PROBE")
print("="*76)

print(
    f"PHYSICAL_REFERENCE="
    f"{PHYSICAL_REFERENCE_MHZ:.0f}_MHz"
)

print(
    f"PREVIOUS_DELTA_TARGET="
    f"{PREVIOUS_DELTA_MHZ:.0f}_MHz"
)

print(
    f"NEW_DELTA_TARGET="
    f"{TARGET_DELTA_MHZ:.0f}_MHz"
)

print(
    f"DELTA_STEP="
    f"+{STEP_MHZ:.0f}_MHz"
)

print(
    f"STEP_RATIO="
    f"{STEP_RATIO:.8f}x"
)

print()

print("--------------- BASELINE ----------------")

print(
    f"BASE_MEAN_US="
    f"{BASE_MEAN_US:.3f}"
)

print(
    f"BASE_GINSTR_S="
    f"{BASE_GINSTR_S:.6f}"
)

print(
    f"HALF_LATENCY_TARGET_US="
    f"{TARGET_MEAN_US:.4f}"
)

print(
    f"DOUBLE_PROCESSING_TARGET_GINSTR_S="
    f"{TARGET_GINSTR_S:.6f}"
)

print()

print("--------------- NEW MEASUREMENT ----------")

print(
    f"SAMPLES={SAMPLES}"
)

print(
    f"MIN_US={min_us:.3f}"
)

print(
    f"MEDIAN_US={median_us:.3f}"
)

print(
    f"MEAN_US={mean_us:.3f}"
)

print(
    f"P95_US={p95_us:.3f}"
)

print(
    f"P99_US={p99_us:.3f}"
)

print(
    f"MAX_US={max_us:.3f}"
)

print(
    f"STDDEV_US={std_us:.3f}"
)

print(
    f"MEAN_MS={mean_ms:.6f}"
)

print()

print("--------------- PROCESSING ---------------")

print(
    f"NI_PER_CALL="
    f"{ni_values[0]}"
)

print(
    f"X512_GINSTR_S="
    f"{ginstr_s:.6f}"
)

print(
    f"LATENCY_GAIN="
    f"{latency_gain:.6f}x"
)

print(
    f"LATENCY_REDUCTION="
    f"{latency_reduction_pct:+.3f}%"
)

print(
    f"GINSTR_GAIN="
    f"{ginstr_gain:.6f}x"
)

print()

print("--------------- DELTA CLOCK --------------")

print(
    f"EQUIVALENT_FROM_LATENCY="
    f"{equivalent_from_latency:.1f}_MHz"
)

print(
    f"EQUIVALENT_FROM_GINSTR="
    f"{equivalent_from_ginstr:.1f}_MHz"
)

print(
    "TARGET_4450_REACHED="
    + (
        "YES"
        if target_4450_reached
        else "NO"
    )
)

print(
    "HALF_LATENCY_REACHED="
    + (
        "YES"
        if half_latency_reached
        else "NO"
    )
)

print()

print("--------------- VALIDATION ---------------")

print(
    "RESULT_STABILITY="
    + (
        "PASS"
        if stable_result
        else "FAIL"
    )
)

print(
    "NI_STABILITY="
    + (
        "PASS"
        if stable_ni
        else "FAIL"
    )
)

print(
    "VALIDATION="
    + (
        "PASS"
        if valid
        else "FAIL"
    )
)

print()

# Le palier 4450 n'est acquis que par mesure.
if valid and target_4450_reached:
    state="DELTA_4450_VALIDATED"
else:
    state="DELTA_4400_REMAINS_REFERENCE"

print(
    "STATE="+state
)

print("="*76)
