#!/usr/bin/env python3

import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

DSPC = 1
RESULT = HERE / "delta_world_benchmark_fresh.json"

world_start = time.perf_counter_ns()
jobs_total = 0
results = []


def duration(ns):
    return {
        "ns": ns,
        "us": ns / 1_000.0,
        "ms": ns / 1_000_000.0,
    }


def extract_gflops(text):
    patterns = [
        r'([0-9]+(?:\.[0-9]+)?)\s*GFLOP/s',
        r'([0-9]+(?:\.[0-9]+)?)\s*GFLOPS',
        r'([0-9]+(?:\.[0-9]+)?)\s*GF\b',
    ]

    values = []

    for pat in patterns:
        for x in re.findall(pat, text, re.I):
            try:
                values.append(float(x))
            except ValueError:
                pass

    return max(values) if values else None


def extract_job_ids(text):
    patterns = [
        r'IBM_JOB_ID\s*=\s*([A-Za-z0-9_-]+)',
        r'JOB_ID\s*=\s*([A-Za-z0-9_-]+)',
        r'job[_ ]?id["\':=\s]+([A-Za-z0-9_-]+)',
    ]

    ids = []

    for pat in patterns:
        ids += re.findall(pat, text, re.I)

    return list(dict.fromkeys(ids))


def execute(label, classification, candidates, env=None):
    global jobs_total

    target = None

    for name in candidates:
        p = HERE / name
        if p.exists():
            target = p
            break

    if target is None:
        r = {
            "label": label,
            "classification": classification,
            "status": "NOT_FOUND",
            "jobs": 0,
            "gflops": None
        }

        results.append(r)
        return r

    jobs_total += 1

    cmd = (
        [sys.executable, str(target)]
        if target.suffix == ".py"
        else [str(target)]
    )

    e = os.environ.copy()

    if env:
        e.update(env)

    t0 = time.perf_counter_ns()

    p = subprocess.run(
        cmd,
        cwd=HERE,
        env=e,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    t1 = time.perf_counter_ns()

    text = p.stdout + "\n" + p.stderr
    d = duration(t1 - t0)

    r = {
        "label": label,
        "classification": classification,
        "program": str(target.relative_to(HERE)),
        "jobs": 1,
        "time_ns": d["ns"],
        "time_us": d["us"],
        "time_ms": d["ms"],
        "gflops": extract_gflops(text),
        "job_ids": extract_job_ids(text),
        "returncode": p.returncode,
        "status": "PASSED" if p.returncode == 0 else "FAILED",
        "stdout": p.stdout[-6000:],
        "stderr": p.stderr[-3000:]
    }

    results.append(r)
    return r


print()
print("============================================================")
print(" DELTA WORLD — FRESH CHECK-UP BENCHMARK")
print(" AUCUNE ANCIENNE MESURE REPRISE")
print("============================================================")
print("DSPC_ACTIVE     = 1")
print("DSPC_CHANNELS   = 1")
print("MODE            = FRESH_RUN")
print()


# ------------------------------------------------------------
# 1 — XEON PHYSIQUE
# delta_platform effectue une nouvelle campagne CPU.
# ------------------------------------------------------------

execute(
    "XEON",
    "PHYSICAL_MEASURED",
    [
        "delta_platform.py",
        "delta_gpu.py"
    ]
)


# ------------------------------------------------------------
# 2 — DELTA CORE
# Nouvelle exécution de D.compute().
# ------------------------------------------------------------

jobs_total += 1

t0 = time.perf_counter_ns()

try:
    from delta_core import DeltaCore

    D = DeltaCore()

    cr = D.compute(
        name="DELTA_WORLD_FRESH",
        work=200000
    )

    t1 = time.perf_counter_ns()
    d = duration(t1 - t0)

    # D.compute actuel mesure un kernel entier déterministe.
    # Ce n'est pas un kernel FLOP défini :
    # aucun faux GFLOPS n'est fabriqué.

    results.append({
        "label": "DELTA",
        "classification": "PHYSICAL_MEASURED_ON_HOST",
        "program": "delta_core.py::DeltaCore.compute",
        "jobs": 1,
        "time_ns": d["ns"],
        "time_us": d["us"],
        "time_ms": d["ms"],
        "gflops": None,
        "iterations_per_second":
            cr.get("iterations_per_second"),
        "signature": cr.get("signature"),
        "status": cr.get("validation", "PASSED")
    })

except Exception as exc:
    t1 = time.perf_counter_ns()
    d = duration(t1 - t0)

    results.append({
        "label": "DELTA",
        "classification": "PHYSICAL_MEASURED_ON_HOST",
        "jobs": 1,
        "time_ns": d["ns"],
        "time_us": d["us"],
        "time_ms": d["ms"],
        "gflops": None,
        "status": "FAILED",
        "error": repr(exc)
    })


# ------------------------------------------------------------
# 3 — RYZEN 9950X3D — MONDE VIRTUEL DELTA
# ------------------------------------------------------------

execute(
    "RYZEN 9 9950X3D",
    "EMULATED_ON_XEON",
    [
        "delta_ryzen_emu.py",
        "delta_ryzen_qpu.py"
    ]
)


# ------------------------------------------------------------
# 4 — RTX 4080 — MONDE VIRTUEL DELTA
# ------------------------------------------------------------

execute(
    "RTX 4080",
    "EMULATED_ON_XEON",
    [
        "delta_rtx_emu_fast.py",
        "delta_rtx_emu.py"
    ]
)


# ------------------------------------------------------------
# 5 — IBM QPU PHYSIQUE
# EXACTEMENT UN APPEL ORCHESTRATEUR
# ------------------------------------------------------------

qpu = execute(
    "IBM QPU",
    "PHYSICAL_QPU_REQUESTED",
    [
        "delta_backend/ibm_qpu_execute.py"
    ]
)

if qpu["status"] == "PASSED":
    if qpu.get("job_ids"):
        qpu["classification"] = "PHYSICAL_MEASURED"
    else:
        qpu["classification"] = "EXECUTED_NO_JOB_ID_DETECTED"


# ------------------------------------------------------------
# RAPPORT
# ------------------------------------------------------------

world_end = time.perf_counter_ns()
wd = duration(world_end - world_start)

passed = all(
    r["status"] == "PASSED"
    for r in results
)

report = {
    "benchmark": "DELTA_WORLD_FRESH_V1",
    "fresh_measurements_only": True,

    "dspc": {
        "channels": DSPC,
        "active": 1,
        "mode": "SINGLE_DSPC"
    },

    "jobs_total": jobs_total,

    "world_time": {
        "ns": wd["ns"],
        "us": wd["us"],
        "ms": wd["ms"]
    },

    "results": results,

    "validation":
        "PASSED"
        if passed
        else "PARTIAL_OR_FAILED"
}

with RESULT.open("w") as f:
    json.dump(report, f, indent=2)


print()
print("============================================================")
print(" RESULTATS FRAIS")
print("============================================================")

for r in results:

    print()
    print(r["label"])
    print("-" * 60)

    print(
        "CLASSIFICATION =",
        r.get("classification")
    )

    print(
        "JOBS           =",
        r.get("jobs", 0)
    )

    if "time_us" in r:
        print(
            "TEMPS_US       = %.3f us"
            % r["time_us"]
        )

        print(
            "TEMPS_MS       = %.6f ms"
            % r["time_ms"]
        )

    gf = r.get("gflops")

    if gf is None:
        print(
            "GFLOPS         = N/A "
            "(kernel FLOP non défini)"
        )
    else:
        print(
            "GFLOPS         = %.6f"
            % gf
        )

    if r.get("iterations_per_second") is not None:
        print(
            "ITER/S         = %.3f"
            % r["iterations_per_second"]
        )

    if r.get("job_ids"):
        print(
            "JOB_ID         =",
            ",".join(r["job_ids"])
        )

    print(
        "STATUS         =",
        r.get("status")
    )


print()
print("============================================================")
print(" MONDE VIRTUEL DELTA")
print("============================================================")

print("DSPC_CHANNELS   =", DSPC)
print("DSPC_ACTIVE     = 1")
print("JOBS_TOTAL      =", jobs_total)

print(
    "WORLD_TIME_US   = %.3f us"
    % wd["us"]
)

print(
    "WORLD_TIME_MS   = %.6f ms"
    % wd["ms"]
)

print(
    "FINAL_VALIDATION=",
    report["validation"]
)

print(
    "RESULT_FILE     =",
    RESULT.name
)

print("============================================================")
