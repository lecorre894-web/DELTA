#!/usr/bin/env python3

import os
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor

MASK = 0xFFFFFFFFFFFFFFFF
TOTAL_WORK = 6400000
REPEATS = 3

CLUSTERS = [1, 2, 4, 8, 16, 32, 64]

HOST_CPUS = os.cpu_count() or 1


def work_unit(args):
    cluster_id, start, count = args

    x = (
        0x9E3779B97F4A7C15
        ^ ((cluster_id + 1) * 0xBF58476D1CE4E5B9)
    ) & MASK

    end = start + count

    for i in range(start, end):
        x ^= (
            i
            + cluster_id
            + 0x9E3779B97F4A7C15
        ) & MASK

        x = (
            x * 0xBF58476D1CE4E5B9
        ) & MASK

        x ^= x >> 29

    return x & MASK


def run_once(cluster_count):
    base = TOTAL_WORK // cluster_count
    rem = TOTAL_WORK % cluster_count

    jobs = []
    pos = 0

    for c in range(cluster_count):
        count = base + (1 if c < rem else 0)

        jobs.append(
            (c, pos, count)
        )

        pos += count

    t0 = time.perf_counter()
    c0 = time.process_time()

    # Les grappes sont logiques.
    # Le nombre de workers physiques est limité au host réel.
    workers = min(
        cluster_count,
        HOST_CPUS
    )

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="delta-grappe"
    ) as ex:
        signatures = list(
            ex.map(work_unit, jobs)
        )

    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0

    checksum = 0

    for x in signatures:
        checksum ^= x

    return {
        "wall_s": wall,
        "cpu_s": cpu,
        "throughput": TOTAL_WORK / wall,
        "checksum": checksum,
        "workers": workers
    }


print("=== DELTA CLUSTER BENCH V1 ===")
print("HOST_LOGICAL_CPUS =", HOST_CPUS)
print("TOTAL_WORK        =", TOTAL_WORK)
print("REPEATS           =", REPEATS)
print()

results = []

for clusters in CLUSTERS:

    runs = []

    print(
        "--- GRAPPES = %d ---"
        % clusters
    )

    for r in range(REPEATS):

        x = run_once(clusters)
        runs.append(x)

        print(
            "RUN %d  WALL=%.6f s  "
            "DEBIT=%.3f iter/s  "
            "WORKERS=%d"
            % (
                r + 1,
                x["wall_s"],
                x["throughput"],
                x["workers"]
            )
        )

    walls = [
        x["wall_s"]
        for x in runs
    ]

    throughputs = [
        x["throughput"]
        for x in runs
    ]

    med_wall = statistics.median(walls)
    med_tp = statistics.median(throughputs)

    result = {
        "clusters": clusters,
        "physical_workers":
            min(clusters, HOST_CPUS),
        "median_wall_s": med_wall,
        "median_iterations_per_second":
            med_tp,
        "runs": runs
    }

    results.append(result)

    print(
        "MEDIAN WALL = %.6f s"
        % med_wall
    )

    print(
        "MEDIAN DEBIT = %.3f iter/s"
        % med_tp
    )

    print()


baseline = results[0][
    "median_iterations_per_second"
]

for r in results:
    r["speedup_vs_1_cluster"] = (
        r["median_iterations_per_second"]
        / baseline
    )


best = max(
    results,
    key=lambda x:
        x["median_iterations_per_second"]
)


print("=== RESULTAT ===")

for r in results:

    print(
        "GRAPPES=%2d  "
        "WORKERS=%d  "
        "DEBIT=%12.3f  "
        "SPEEDUP=%6.3fx"
        % (
            r["clusters"],
            r["physical_workers"],
            r[
                "median_iterations_per_second"
            ],
            r["speedup_vs_1_cluster"]
        )
    )


print()
print(
    "BEST_CLUSTER_COUNT      =",
    best["clusters"]
)

print(
    "BEST_PHYSICAL_WORKERS   =",
    best["physical_workers"]
)

print(
    "BEST_THROUGHPUT         = %.3f"
    % best[
        "median_iterations_per_second"
    ]
)

print(
    "BEST_SPEEDUP            = %.3fx"
    % best["speedup_vs_1_cluster"]
)


out = {
    "host_logical_cpus": HOST_CPUS,
    "total_work": TOTAL_WORK,
    "repeats": REPEATS,
    "results": results,
    "best": best
}

with open(
    "delta_cluster_bench_result.json",
    "w"
) as f:
    json.dump(
        out,
        f,
        indent=2
    )


print(
    "RESULT_FILE             = "
    "delta_cluster_bench_result.json"
)

print(
    "DELTA_CLUSTER_BENCH     = PASSED"
)
