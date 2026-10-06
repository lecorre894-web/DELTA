#!/usr/bin/env python3

import time
import statistics
import numpy as np

from delta_x512 import X512Core, aligne, vecs

SEED = 894
NW = 16384 * 4

WARMUP = 32
SAMPLES = 1000

core = X512Core()

# Un même dataset déterministe et déjà préparé :
# ici on mesure TPOP, pas la génération des entrées.
A,B,C = vecs(SEED, NW)
A = aligne(A)
B = aligne(B)
C = aligne(C)

# Warm-up non compté
for _ in range(WARMUP):
    core.tpop(A,B,C,prog="micro")

samples_ns = []
ni_values = []
results = []

for _ in range(SAMPLES):

    t0 = time.perf_counter_ns()

    r,ni = core.tpop(
        A,B,C,
        prog="micro"
    )

    t1 = time.perf_counter_ns()

    samples_ns.append(t1-t0)
    ni_values.append(ni)
    results.append(r)

a = np.asarray(samples_ns,dtype=np.float64)

us = a / 1_000.0
ms = a / 1_000_000.0

mean_us   = float(np.mean(us))
median_us = float(np.median(us))
min_us    = float(np.min(us))
p95_us    = float(np.percentile(us,95))
p99_us    = float(np.percentile(us,99))
max_us    = float(np.max(us))
std_us    = float(np.std(us))

mean_ms = mean_us / 1000.0

stable_result = len(set(results)) == 1
stable_ni = len(set(ni_values)) == 1

total_ni = sum(ni_values)
total_seconds = sum(samples_ns) / 1e9

ginstr_s = (
    total_ni /
    total_seconds /
    1e9
)

print("="*72)
print("DELTA X512 OC100 — LATENCY µs / ms")
print("="*72)

print(f"SAMPLES={SAMPLES}")
print(f"WARMUP={WARMUP}")
print(f"WORDS={NW}")
print()

print("--------------- TPOP LATENCY ----------------")
print(f"MIN_US={min_us:.3f}")
print(f"MEDIAN_US={median_us:.3f}")
print(f"MEAN_US={mean_us:.3f}")
print(f"P95_US={p95_us:.3f}")
print(f"P99_US={p99_us:.3f}")
print(f"MAX_US={max_us:.3f}")
print(f"STDDEV_US={std_us:.3f}")
print()
print(f"MEAN_MS={mean_ms:.6f}")
print()

print("--------------- PROCESSING ------------------")
print(f"NI_PER_CALL={ni_values[0]}")
print(f"X512_GINSTR_S={ginstr_s:.6f}")
print()

print("--------------- VALIDATION ------------------")
print(
    "RESULT_STABILITY="
    + ("PASS" if stable_result else "FAIL")
)
print(
    "NI_STABILITY="
    + ("PASS" if stable_ni else "FAIL")
)

valid = stable_result and stable_ni

print(
    "LATENCY_PROBE="
    + ("PASS" if valid else "FAIL")
)

print("="*72)
