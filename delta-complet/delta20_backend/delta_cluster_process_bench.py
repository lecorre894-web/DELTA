#!/usr/bin/env python3

import os
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

MASK = 0xFFFFFFFFFFFFFFFF
TOTAL_WORK = 6400000
REPEATS = 5
CLUSTERS = [1, 2, 4, 8, 16, 32, 64]

try:
    HOST_CPUS = len(os.sched_getaffinity(0))
except AttributeError:
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


def make_jobs(cluster_count):
    base = TOTAL_WORK // cluster_count
    rem = TOTAL_WORK % cluster_count

    jobs = []
    pos = 0

    for c in range(cluster_count):
        count = base + (1 if c < rem else 0)
        jobs.append((c, pos, count))
        pos += count

    return jobs


def run_once(cluster_count, mode):
    jobs = make_jobs(cluster_count)

    workers = min(cluster_count, HOST_CPUS)

    Executor = (
        ThreadPoolExecutor
        if mode == "THREAD"
        else ProcessPoolExecutor
    )

    t0 = time.perf_counter()
    c0 = time.process_time()

    with Executor(max_workers=workers) as ex:
        signatures = list(ex.map(work_unit, jobs))

    wall = time.perf_counter() - t0
    parent_cpu = time.process_time() - c0

    checksum = 0
    for x in signatures:
        checksum ^= x

    return {
        "wall_s": wall,
        "parent_cpu_s": parent_cpu,
        "throughput": TOTAL_WORK / wall,
        "checksum": checksum,
        "workers": workers
    }


def bench_mode(mode):
    results = []

    print()
    print("========================================")
    print("MODE =", mode)
    print("========================================")

    for clusters in CLUSTERS:
        runs = []

        print()
        print("--- GRAPPES = %d ---" % clusters)

        for r in range(REPEATS):
            x = run_once(clusters, mode)
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

        walls = [x["wall_s"] for x in runs]
        throughputs = [x["throughput"] for x in runs]

        med_wall = statistics.median(walls)
        med_tp = statistics.median(throughputs)

        results.append({
            "mode": mode,
            "clusters": clusters,
            "physical_workers": min(clusters, HOST_CPUS),
            "median_wall_s": med_wall,
            "median_iterations_per_second": med_tp,
            "runs": runs
        })

        print(
            "MEDIAN WALL = %.6f s"
            % med_wall
        )

        print(
            "MEDIAN DEBIT = %.3f iter/s"
            % med_tp
        )

    baseline = results[0]["median_iterations_per_second"]

    for x in results:
        x["speedup_vs_1_cluster"] = (
            x["median_iterations_per_second"]
            / baseline
        )

    return results


if __name__ == "__main__":

    print("=== DELTA CLUSTER THREAD/PROCESS BENCH V2 ===")
    print("HOST_AVAILABLE_CPUS =", HOST_CPUS)
    print("TOTAL_WORK          =", TOTAL_WORK)
    print("REPEATS             =", REPEATS)

    thread_results = bench_mode("THREAD")
    process_results = bench_mode("PROCESS")

    thread_best = max(
        thread_results,
        key=lambda x: x["median_iterations_per_second"]
    )

    process_best = max(
        process_results,
        key=lambda x: x["median_iterations_per_second"]
    )

    ratio = (
        process_best["median_iterations_per_second"]
        / thread_best["median_iterations_per_second"]
    )

    print()
    print("========================================")
    print("=== COMPARAISON FINALE ===")
    print("========================================")

    print(
        "THREAD_BEST_CLUSTERS     =",
        thread_best["clusters"]
    )

    print(
        "THREAD_BEST_WORKERS      =",
        thread_best["physical_workers"]
    )

    print(
        "THREAD_BEST_THROUGHPUT   = %.3f"
        % thread_best["median_iterations_per_second"]
    )

    print()

    print(
        "PROCESS_BEST_CLUSTERS    =",
        process_best["clusters"]
    )

    print(
        "PROCESS_BEST_WORKERS     =",
        process_best["physical_workers"]
    )

    print(
        "PROCESS_BEST_THROUGHPUT  = %.3f"
        % process_best["median_iterations_per_second"]
    )

    print()

    print(
        "PROCESS_VS_THREAD        = %.3fx"
        % ratio
    )

    winner = (
        "PROCESS"
        if ratio > 1.0
        else "THREAD"
    )

    print(
        "MEASURED_WINNER          =",
        winner
    )

    out = {
        "host_available_cpus": HOST_CPUS,
        "total_work": TOTAL_WORK,
        "repeats": REPEATS,
        "thread": thread_results,
        "process": process_results,
        "thread_best": thread_best,
        "process_best": process_best,
        "process_vs_thread": ratio,
        "measured_winner": winner
    }

    with open(
        "delta_cluster_process_bench_result.json",
        "w"
    ) as f:
        json.dump(out, f, indent=2)

    print(
        "RESULT_FILE              = "
        "delta_cluster_process_bench_result.json"
    )

    print(
        "DELTA_CLUSTER_PROCESS_BENCH=PASSED"
    )
