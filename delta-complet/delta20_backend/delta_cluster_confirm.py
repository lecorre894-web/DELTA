#!/usr/bin/env python3

import os
import time
import json
import statistics
from concurrent.futures import ProcessPoolExecutor

MASK = 0xFFFFFFFFFFFFFFFF

TOTAL_WORK = 32000000
REPEATS = 10
CLUSTERS = [2, 4]

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


def run_once(cluster_count):
    jobs = make_jobs(cluster_count)
    workers = min(cluster_count, HOST_CPUS)

    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        signatures = list(ex.map(work_unit, jobs))

    wall = time.perf_counter() - t0

    checksum = 0
    for x in signatures:
        checksum ^= x

    return {
        "wall_s": wall,
        "throughput": TOTAL_WORK / wall,
        "checksum": checksum,
        "workers": workers
    }


def summarize(values):
    ordered = sorted(values)

    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "q1": statistics.median(ordered[:len(ordered)//2]),
        "q3": statistics.median(ordered[(len(ordered)+1)//2:])
    }


if __name__ == "__main__":

    print("=== DELTA CLUSTER CONFIRMATION V1 ===")
    print("HOST_AVAILABLE_CPUS =", HOST_CPUS)
    print("TOTAL_WORK          =", TOTAL_WORK)
    print("REPEATS             =", REPEATS)
    print("CANDIDATES          =", CLUSTERS)
    print()

    all_results = {}

    for clusters in CLUSTERS:

        print("========================================")
        print("GRAPPES =", clusters)
        print("========================================")

        runs = []

        for r in range(REPEATS):
            x = run_once(clusters)
            runs.append(x)

            print(
                "RUN %2d  WALL=%8.6f s  "
                "DEBIT=%12.3f iter/s  "
                "WORKERS=%d"
                % (
                    r + 1,
                    x["wall_s"],
                    x["throughput"],
                    x["workers"]
                )
            )

        throughputs = [
            x["throughput"]
            for x in runs
        ]

        walls = [
            x["wall_s"]
            for x in runs
        ]

        tp_stats = summarize(throughputs)
        wall_stats = summarize(walls)

        all_results[str(clusters)] = {
            "clusters": clusters,
            "workers": min(clusters, HOST_CPUS),
            "throughput": tp_stats,
            "wall": wall_stats,
            "runs": runs
        }

        print()
        print(
            "MEDIAN_DEBIT = %.3f"
            % tp_stats["median"]
        )

        print(
            "MEAN_DEBIT   = %.3f"
            % tp_stats["mean"]
        )

        print(
            "STDEV_DEBIT  = %.3f"
            % tp_stats["stdev"]
        )

        print(
            "MIN_DEBIT    = %.3f"
            % tp_stats["min"]
        )

        print(
            "MAX_DEBIT    = %.3f"
            % tp_stats["max"]
        )

        print()

    r2 = all_results["2"]["throughput"]
    r4 = all_results["4"]["throughput"]

    median_ratio = r4["median"] / r2["median"]
    mean_ratio = r4["mean"] / r2["mean"]

    median_gain = (median_ratio - 1.0) * 100.0
    mean_gain = (mean_ratio - 1.0) * 100.0

    winner_median = (
        4 if median_ratio > 1.0 else 2
    )

    winner_mean = (
        4 if mean_ratio > 1.0 else 2
    )

    print("========================================")
    print("=== CONFIRMATION FINALE ===")
    print("========================================")

    print(
        "2_GRAPPES_MEDIAN         = %.3f"
        % r2["median"]
    )

    print(
        "4_GRAPPES_MEDIAN         = %.3f"
        % r4["median"]
    )

    print(
        "4_VS_2_MEDIAN            = %.4fx"
        % median_ratio
    )

    print(
        "4_VS_2_MEDIAN_GAIN       = %+.3f%%"
        % median_gain
    )

    print()

    print(
        "2_GRAPPES_MEAN           = %.3f"
        % r2["mean"]
    )

    print(
        "4_GRAPPES_MEAN           = %.3f"
        % r4["mean"]
    )

    print(
        "4_VS_2_MEAN              = %.4fx"
        % mean_ratio
    )

    print(
        "4_VS_2_MEAN_GAIN         = %+.3f%%"
        % mean_gain
    )

    print()

    print(
        "MEDIAN_WINNER            =",
        winner_median,
        "GRAPPES"
    )

    print(
        "MEAN_WINNER              =",
        winner_mean,
        "GRAPPES"
    )

    # Règle conservatrice :
    # on ne déclare 4 grappes confirmées que si
    # moyenne ET médiane gagnent toutes deux.
    confirmed = (
        median_ratio > 1.0
        and mean_ratio > 1.0
    )

    print(
        "4_CLUSTER_CONFIRMATION   =",
        "PASSED" if confirmed else "NOT_CONFIRMED"
    )

    output = {
        "host_available_cpus": HOST_CPUS,
        "total_work": TOTAL_WORK,
        "repeats": REPEATS,
        "results": all_results,
        "comparison": {
            "median_ratio_4_vs_2": median_ratio,
            "median_gain_percent": median_gain,
            "mean_ratio_4_vs_2": mean_ratio,
            "mean_gain_percent": mean_gain,
            "median_winner": winner_median,
            "mean_winner": winner_mean,
            "four_cluster_confirmed": confirmed
        }
    }

    with open(
        "delta_cluster_confirm_result.json",
        "w"
    ) as f:
        json.dump(output, f, indent=2)

    print(
        "RESULT_FILE              = "
        "delta_cluster_confirm_result.json"
    )

    print(
        "DELTA_CLUSTER_CONFIRM    = PASSED"
    )
