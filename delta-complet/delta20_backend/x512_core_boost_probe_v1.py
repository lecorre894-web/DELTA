#!/usr/bin/env python3

import os
import time
import statistics
import numpy as np

from delta_x512 import X512Core, aligne, vecs

CPU_COUNT = os.cpu_count() or 1

SEED = 894
NW = 16384 * 4

WARMUP = 64
SAMPLES = 300

print("="*78)
print("DELTA X512 — CORE-BOOST PROBE V1")
print("="*78)

print(f"CPU_LOGICAL={CPU_COUNT}")
print(f"WARMUP_PER_CPU={WARMUP}")
print(f"SAMPLES_PER_CPU={SAMPLES}")
print(f"WORDS={NW}")
print()

# ------------------------------------------------------------
# Dataset unique : aucune génération dans les mesures.
# ------------------------------------------------------------

A,B,C = vecs(SEED,NW)

A = aligne(A)
B = aligne(B)
C = aligne(C)

core = X512Core()

# ------------------------------------------------------------
# CPU MHz reporté par Linux
# ------------------------------------------------------------

def cpu_mhz(cpu):

    try:
        with open(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        ) as f:

            return float(f.read().strip()) / 1000.0

    except Exception:
        pass

    try:

        current = -1

        with open("/proc/cpuinfo") as f:

            for line in f:

                if line.startswith("processor"):

                    current = int(
                        line.split(":")[1]
                    )

                elif (
                    current == cpu
                    and line.lower().startswith("cpu mhz")
                ):

                    return float(
                        line.split(":")[1]
                    )

    except Exception:
        pass

    return 0.0


# ------------------------------------------------------------
# Affinité initiale
# ------------------------------------------------------------

original_affinity = os.sched_getaffinity(0)

cpus = sorted(original_affinity)

print(
    "AVAILABLE_CPUS="
    + ",".join(map(str,cpus))
)

print()
print(
    "CPU | MHz(rep.) | MIN us | MEDIAN us | MEAN us | "
    "P95 us | GInstr/s"
)

print("-"*78)

rows=[]

reference_result=None
reference_ni=None

for cpu in cpus:

    # --------------------------------------------------------
    # PIN sur UN CPU logique
    # --------------------------------------------------------

    os.sched_setaffinity(
        0,
        {cpu}
    )

    # Petite stabilisation
    time.sleep(0.02)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warm_result=None
    warm_ni=None

    for _ in range(WARMUP):

        warm_result,warm_ni = core.tpop(
            A,B,C,
            prog="micro"
        )

    # --------------------------------------------------------
    # Mesure
    # --------------------------------------------------------

    samples=[]
    ni_total=0

    stable=True

    for _ in range(SAMPLES):

        t0=time.perf_counter_ns()

        result,ni=core.tpop(
            A,B,C,
            prog="micro"
        )

        t1=time.perf_counter_ns()

        samples.append(
            t1-t0
        )

        ni_total += ni

        if reference_result is None:
            reference_result=result
            reference_ni=ni

        if (
            result != reference_result
            or
            ni != reference_ni
        ):
            stable=False

    arr=np.asarray(
        samples,
        dtype=np.float64
    ) / 1000.0

    minimum=float(
        np.min(arr)
    )

    median=float(
        np.median(arr)
    )

    mean=float(
        np.mean(arr)
    )

    p95=float(
        np.percentile(
            arr,
            95
        )
    )

    total_s=(
        sum(samples) /
        1e9
    )

    ginstr=(
        ni_total /
        total_s /
        1e9
    )

    mhz=cpu_mhz(cpu)

    rows.append({
        "cpu":cpu,
        "mhz":mhz,
        "min":minimum,
        "median":median,
        "mean":mean,
        "p95":p95,
        "ginstr":ginstr,
        "stable":stable
    })

    print(
        f"{cpu:3d} | "
        f"{mhz:8.1f} | "
        f"{minimum:6.3f} | "
        f"{median:9.3f} | "
        f"{mean:7.3f} | "
        f"{p95:6.3f} | "
        f"{ginstr:8.4f}"
    )


# ------------------------------------------------------------
# Restauration affinité
# ------------------------------------------------------------

os.sched_setaffinity(
    0,
    original_affinity
)


# ------------------------------------------------------------
# Classement
# ------------------------------------------------------------

valid_rows=[
    r for r in rows
    if r["stable"]
]

ranking=sorted(
    valid_rows,
    key=lambda r:r["median"]
)

print()
print("="*78)
print("TOP 8 — LOWEST MEDIAN LATENCY")
print("="*78)

for rank,r in enumerate(
    ranking[:8],
    1
):

    print(
        f"RANK={rank} "
        f"CPU={r['cpu']} "
        f"MEDIAN_US={r['median']:.3f} "
        f"MEAN_US={r['mean']:.3f} "
        f"MIN_US={r['min']:.3f} "
        f"P95_US={r['p95']:.3f} "
        f"GINSTR_S={r['ginstr']:.6f} "
        f"MHZ_REPORTED={r['mhz']:.1f}"
    )


# ------------------------------------------------------------
# Winner
# ------------------------------------------------------------

if ranking:

    winner=ranking[0]

    best_cpu=winner["cpu"]
    best_median=winner["median"]
    best_mean=winner["mean"]
    best_ginstr=winner["ginstr"]

    print()
    print("--------------- WINNER -----------------------")

    print(
        f"BEST_CPU={best_cpu}"
    )

    print(
        f"BEST_MEDIAN_US={best_median:.3f}"
    )

    print(
        f"BEST_MEAN_US={best_mean:.3f}"
    )

    print(
        f"BEST_GINSTR_S={best_ginstr:.6f}"
    )

    print(
        f"BEST_MHZ_REPORTED="
        f"{winner['mhz']:.1f}"
    )

    # Comparaison avec dernière moyenne mesurée
    old_mean=15.827

    gain=(
        old_mean /
        best_mean
    )

    reduction=(
        (1.0-best_mean/old_mean)
        *100.0
    )

    print(
        f"VS_OLD_MEAN_GAIN="
        f"{gain:.6f}x"
    )

    print(
        f"VS_OLD_MEAN_REDUCTION="
        f"{reduction:+.3f}%"
    )

    # Seuil correspondant au +50 MHz 4400 -> 4450
    required_ratio=4450.0/4400.0

    required_mean=(
        old_mean /
        required_ratio
    )

    print(
        f"4450_REQUIRED_MEAN_US="
        f"{required_mean:.3f}"
    )

    reached=(
        best_mean <= required_mean
    )

    print(
        "4450_LATENCY_THRESHOLD="
        + (
            "PASS"
            if reached
            else "FAIL"
        )
    )

    print()
    print(
        "CORE_BOOST_VALIDATION=PASS"
        if all(r["stable"] for r in rows)
        else
        "CORE_BOOST_VALIDATION=FAIL"
    )

    print(
        f"WINNER=CPU_{best_cpu}"
    )

else:

    print(
        "CORE_BOOST_VALIDATION=FAIL"
    )

print("="*78)
