#!/usr/bin/env python3

import os
import re
import json
import time
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

DSPC_CHANNELS = 1
OUTFILE = HERE / "delta_final_benchmark_result.json"

R = {
    "benchmark": "DELTA_FINAL_BENCHMARK_V1",
    "dspc_channels": DSPC_CHANNELS,
    "components": {},
    "validation": "UNKNOWN"
}


def run_python(name, filename, classification):
    path = HERE / filename

    if not path.exists():
        return {
            "name": name,
            "classification": classification,
            "status": "NOT_FOUND"
        }

    t0 = time.perf_counter()

    p = subprocess.run(
        ["python3", str(path)],
        cwd=HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "name": name,
        "classification": classification,
        "wall_ms": elapsed_ms,
        "returncode": p.returncode,
        "status": "PASSED" if p.returncode == 0 else "FAILED",
        "stdout_tail": p.stdout[-3000:],
        "stderr_tail": p.stderr[-1500:]
    }


def read_xeon_fma():
    path = HERE / "bench_fma" / "RESULTATS.md"

    out = {
        "name": "XEON_FMA",
        "classification": "PHYSICAL_MEASURED",
        "source": "bench_fma/RESULTATS.md",
        "status": "NOT_FOUND"
    }

    if not path.exists():
        return out

    txt = path.read_text(errors="replace")

    # Ligne validée observée :
    # x16 chaines ... : 9182 ms, 182.72 GF
    m = re.search(
        r"x16.*?:\s*([0-9.]+)\s*ms,\s*([0-9.]+)\s*GF",
        txt,
        re.I
    )

    if m:
        out.update({
            "wall_ms": float(m.group(1)),
            "gflops": float(m.group(2)),
            "status": "PASSED"
        })

    return out


def run_delta_engine():
    """
    Charge déterministe courte exécutée réellement maintenant.
    Ce résultat n'est PAS comparé directement au FMA comme
    s'il s'agissait du même kernel.
    """
    total = 1_000_000
    mask = 0xFFFFFFFFFFFFFFFF

    t0 = time.perf_counter()

    x = 0x9E3779B97F4A7C15

    for i in range(total):
        x ^= (i + 0xBF58476D1CE4E5B9) & mask
        x = (x * 0x94D049BB133111EB) & mask
        x ^= x >> 29

    ms = (time.perf_counter() - t0) * 1000.0

    return {
        "name": "DELTA_ENGINE",
        "classification": "PHYSICAL_MEASURED_ON_HOST",
        "task": "DETERMINISTIC_INTEGER_KERNEL",
        "iterations": total,
        "wall_ms": ms,
        "iterations_per_second": total / (ms / 1000.0),
        "signature": str(x),
        "status": "PASSED"
    }


def run_qpu():
    enabled = os.environ.get("DELTA_FINAL_QPU") == "1"

    if not enabled:
        return {
            "name": "IBM_QPU",
            "classification": "PHYSICAL_QPU_NOT_EXECUTED_THIS_RUN",
            "status": "SKIPPED",
            "reason": "DELTA_FINAL_QPU != 1"
        }

    # Utilise le batch existant sans modifier son code.
    t0 = time.perf_counter()

    p = subprocess.run(
        ["python3", str(HERE / "delta_qpu_batch.py")],
        cwd=HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    total_ms = (time.perf_counter() - t0) * 1000.0

    job_ids = re.findall(
        r"(?:job|JOB)[^A-Za-z0-9_-]*([A-Za-z0-9_-]{8,})",
        p.stdout
    )

    return {
        "name": "IBM_QPU",
        "classification":
            "PHYSICAL_MEASURED"
            if p.returncode == 0
            else "PHYSICAL_EXECUTION_FAILED",
        "total_observed_ms": total_ms,
        "qpu_execution_ms": None,
        "note":
            "total_observed_ms inclut transport/file/attente; "
            "qpu_execution_ms reste inconnu si IBM ne le fournit pas",
        "job_ids_detected": job_ids,
        "returncode": p.returncode,
        "status": "PASSED" if p.returncode == 0 else "FAILED",
        "stdout_tail": p.stdout[-4000:],
        "stderr_tail": p.stderr[-2000:]
    }


print()
print("======================================================")
print(" DELTA FINAL BENCHMARK V1")
print(" DSPC UNIQUE")
print("======================================================")
print("DSPC_CHANNELS =", DSPC_CHANNELS)
print()

# ------------------------------------------------------
# 1. XEON physique : résultat FMA validé existant
# ------------------------------------------------------

R["components"]["xeon"] = read_xeon_fma()

# ------------------------------------------------------
# 2. DELTA : mesure réellement exécutée maintenant
# ------------------------------------------------------

R["components"]["delta"] = run_delta_engine()

# ------------------------------------------------------
# 3. Ryzen : émulation exécutée sur l'hôte
# ------------------------------------------------------

R["components"]["ryzen"] = run_python(
    "RYZEN_9950X3D",
    "delta_ryzen_emu.py",
    "EMULATED_ON_XEON"
)

# ------------------------------------------------------
# 4. RTX : émulation exécutée sur l'hôte
# ------------------------------------------------------

R["components"]["rtx"] = run_python(
    "RTX_4080",
    "delta_rtx_emu_fast.py",
    "EMULATED_ON_XEON"
)

# ------------------------------------------------------
# 5. QPU IBM
# ------------------------------------------------------

R["components"]["qpu"] = run_qpu()

# ------------------------------------------------------
# Validation
# ------------------------------------------------------

required = ["xeon", "delta", "ryzen", "rtx"]

ok = all(
    R["components"][x].get("status") == "PASSED"
    for x in required
)

qpu_status = R["components"]["qpu"].get("status")

if qpu_status == "FAILED":
    ok = False

R["validation"] = "PASSED" if ok else "FAILED"


def fmt_ms(x):
    if x is None:
        return "N/A"
    return "%.3f ms" % x


def line(title, component):
    print(title)

    print(
        "  CLASSIFICATION :",
        component.get("classification", "UNKNOWN")
    )

    if "wall_ms" in component:
        print(
            "  TEMPS          :",
            fmt_ms(component["wall_ms"])
        )

    if "total_observed_ms" in component:
        print(
            "  TEMPS TOTAL    :",
            fmt_ms(component["total_observed_ms"])
        )

    if "gflops" in component:
        print(
            "  PERFORMANCE    : %.3f GFLOPS"
            % component["gflops"]
        )

    if "iterations_per_second" in component:
        print(
            "  DEBIT          : %.3f iter/s"
            % component["iterations_per_second"]
        )

    print(
        "  STATUS         :",
        component.get("status", "UNKNOWN")
    )

    print()


print()
print("================ RESULTATS =================")
print()

line("XEON", R["components"]["xeon"])
line("DELTA", R["components"]["delta"])
line("RYZEN 9950X3D", R["components"]["ryzen"])
line("RTX 4080", R["components"]["rtx"])

q = R["components"]["qpu"]

print("IBM QPU")
print("  CLASSIFICATION :", q["classification"])

if q.get("total_observed_ms") is not None:
    print(
        "  TEMPS OBSERVE  :",
        fmt_ms(q["total_observed_ms"])
    )

if q.get("qpu_execution_ms") is not None:
    print(
        "  TEMPS QPU      :",
        fmt_ms(q["qpu_execution_ms"])
    )
else:
    print("  TEMPS QPU      : N/A")

print("  STATUS         :", q["status"])
print()

print("================ DSPC ======================")
print("DSPC_CHANNELS       =", DSPC_CHANNELS)
print("DSPC_ACTIVE         =", 1)
print("DSPC_MODE           = SINGLE_CHANNEL")
print()

print("================ VALIDATION ================")
print("FINAL_VALIDATION    =", R["validation"])
print("RESULT_FILE         =", OUTFILE.name)

with OUTFILE.open("w") as f:
    json.dump(R, f, indent=2)

print("======================================================")
