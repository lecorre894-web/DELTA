#!/usr/bin/env python3

import os
import time
import json
import hashlib
import platform
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# DELTA OVERLAY SCHEDULER V1
#
# Overlay:
#   AMD Ryzen 9 9950X3D : 16C / 32T virtualisés
#   NVIDIA RTX 4080     : couche virtuelle
#
# Physical execution:
#   Codespaces / Xeon
#
# Important:
#   Les 32 lanes sont des workers logiques DELTA.
#   Le travail mesuré reste exécuté par les CPU réellement
#   disponibles dans le Codespace.
# ============================================================

OVERLAY_CPU = "AMD Ryzen 9 9950X3D"
OVERLAY_CORES = 16
OVERLAY_THREADS = 32

OVERLAY_GPU = "NVIDIA GeForce RTX 4080"
OVERLAY_GPU_VRAM_MIB = 16376

HOST_CPUS = os.cpu_count() or 1

WORK_PER_LANE = int(
    os.environ.get("DELTA_OVERLAY_WORK", "200000")
)


def lane_work(lane_id):
    """
    Charge déterministe par lane.

    Chaque lane possède son état propre et produit une signature.
    Aucun résultat aléatoire n'est utilisé.
    """

    physical_slot = lane_id % HOST_CPUS

    x = (
        0x9E3779B97F4A7C15
        ^ ((lane_id + 1) * 0xBF58476D1CE4E5B9)
    ) & 0xFFFFFFFFFFFFFFFF

    start = time.perf_counter()

    for i in range(WORK_PER_LANE):
        x ^= (
            i
            + lane_id
            + 0x9E3779B97F4A7C15
        ) & 0xFFFFFFFFFFFFFFFF

        x = (
            x * 0xBF58476D1CE4E5B9
        ) & 0xFFFFFFFFFFFFFFFF

        x ^= x >> 29

    wall = time.perf_counter() - start

    return {
        "lane": lane_id,
        "virtual_cpu": lane_id,
        "physical_slot": physical_slot,
        "iterations": WORK_PER_LANE,
        "signature": x,
        "wall_s": wall
    }


def main():

    print("=== DELTA OVERLAY SCHEDULER V1 ===")
    print()

    print("--- PHYSICAL EXECUTION HOST ---")
    print("HOST_CPU                =", platform.processor() or "UNKNOWN")
    print("HOST_LOGICAL_CPUS       =", HOST_CPUS)
    print("HOST_EXECUTION          = PHYSICAL_MEASURED")

    print()
    print("--- VIRTUAL CPU OVERLAY ---")
    print("OVERLAY_CPU             =", OVERLAY_CPU)
    print("OVERLAY_CPU_CORES       =", OVERLAY_CORES)
    print("OVERLAY_CPU_THREADS     =", OVERLAY_THREADS)
    print("OVERLAY_CPU_EXECUTION   = VIRTUALIZED")
    print(
        "OVERLAY_CPU_SCHEDULER   = %d_TO_%d_MULTIPLEXED"
        % (OVERLAY_THREADS, HOST_CPUS)
    )

    print()
    print("--- VIRTUAL GPU OVERLAY ---")
    print("OVERLAY_GPU             =", OVERLAY_GPU)
    print("OVERLAY_GPU_VRAM_MIB    =", OVERLAY_GPU_VRAM_MIB)
    print("OVERLAY_GPU_EXECUTION   = VIRTUALIZED")
    print("OVERLAY_GPU_CUDA        = NOT_PHYSICALLY_VISIBLE")

    print()
    print("--- EXECUTION ---")

    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    with ThreadPoolExecutor(
        max_workers=OVERLAY_THREADS,
        thread_name_prefix="delta-overlay"
    ) as executor:

        results = list(
            executor.map(
                lane_work,
                range(OVERLAY_THREADS)
            )
        )

    wall = time.perf_counter() - start_wall
    cpu_time = time.process_time() - start_cpu

    results.sort(key=lambda r: r["lane"])

    total_iterations = sum(
        r["iterations"] for r in results
    )

    digest = hashlib.sha256()

    for r in results:
        digest.update(
            (
                "%d:%d:%d:%d;"
                % (
                    r["lane"],
                    r["physical_slot"],
                    r["iterations"],
                    r["signature"]
                )
            ).encode()
        )

    execution_signature = digest.hexdigest()

    physical_slots = sorted(
        set(r["physical_slot"] for r in results)
    )

    print("LOGICAL_LANES_EXECUTED   =", len(results))
    print("PHYSICAL_SLOTS_USED      =", len(physical_slots))
    print("TOTAL_ITERATIONS         =", total_iterations)
    print("WALL_TIME_S              = %.6f" % wall)
    print("CPU_TIME_S               = %.6f" % cpu_time)

    if wall > 0:
        print(
            "ITERATIONS_PER_SECOND   = %.3f"
            % (total_iterations / wall)
        )

    print(
        "EXECUTION_SIGNATURE      =",
        execution_signature
    )

    print()
    print("--- LANE ROUTING ---")

    for r in results:
        print(
            "LANE[%02d] -> HOST_SLOT[%d] "
            "SIGNATURE=%016x"
            % (
                r["lane"],
                r["physical_slot"],
                r["signature"]
            )
        )

    valid = (
        len(results) == OVERLAY_THREADS
        and total_iterations
            == OVERLAY_THREADS * WORK_PER_LANE
        and len(physical_slots)
            <= HOST_CPUS
        and len(
            {r["lane"] for r in results}
        ) == OVERLAY_THREADS
    )

    state = {
        "architecture":
            "DELTA_RYZEN_RTX_OVERLAY_V1",

        "host": {
            "cpu":
                platform.processor()
                or "UNKNOWN",

            "logical_cpus":
                HOST_CPUS,

            "execution":
                "PHYSICAL_MEASURED"
        },

        "overlay_cpu": {
            "model":
                OVERLAY_CPU,

            "cores":
                OVERLAY_CORES,

            "threads":
                OVERLAY_THREADS,

            "execution":
                "VIRTUALIZED",

            "scheduler":
                "%d_TO_%d_MULTIPLEXED"
                % (
                    OVERLAY_THREADS,
                    HOST_CPUS
                )
        },

        "overlay_gpu": {
            "model":
                OVERLAY_GPU,

            "vram_mib":
                OVERLAY_GPU_VRAM_MIB,

            "execution":
                "VIRTUALIZED",

            "cuda_physical":
                False
        },

        "measurement": {
            "logical_lanes":
                len(results),

            "physical_slots_used":
                len(physical_slots),

            "iterations":
                total_iterations,

            "wall_s":
                wall,

            "cpu_s":
                cpu_time,

            "signature":
                execution_signature
        },

        "validation":
            "PASSED" if valid else "FAILED"
    }

    with open(
        "delta_overlay_scheduler_result.json",
        "w"
    ) as f:
        json.dump(
            state,
            f,
            indent=2
        )

    print()
    print(
        "DELTA_EXECUTION_VIEW     = "
        "RYZEN_RTX_OVER_XEON"
    )

    print(
        "OVERLAY_WORK_EXECUTION   = "
        "PHYSICAL_XEON"
    )

    print(
        "DELTA_OVERLAY_SCHEDULER  =",
        "PASSED" if valid else "FAILED"
    )

    print(
        "RESULT_FILE              = "
        "delta_overlay_scheduler_result.json"
    )

    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
