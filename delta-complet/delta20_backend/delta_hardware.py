#!/usr/bin/env python3
"""
DELTA - reconnaissance matérielle ponctuelle.
- aucune écriture disque
- aucun service
- aucun thread supplémentaire
- aucune surveillance en arrière-plan
- résultat conservé uniquement en RAM par le processus appelant
"""

from __future__ import annotations

import os
import platform
import re


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def _cpu_linux():
    text = _read_text("/proc/cpuinfo")

    model = ""
    for label in ("model name", "Hardware", "Processor"):
        m = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$",
                      text, flags=re.MULTILINE | re.IGNORECASE)
        if m:
            model = m.group(1).strip()
            break

    flags = set()
    for label in ("flags", "Features"):
        for value in re.findall(
                rf"^{label}\s*:\s*(.+)$",
                text, flags=re.MULTILINE | re.IGNORECASE):
            flags.update(value.lower().split())

    return model or platform.processor() or "UNKNOWN", flags


def _memory_linux():
    text = _read_text("/proc/meminfo")

    def kb(name):
        m = re.search(rf"^{name}:\s*(\d+)\s+kB",
                      text, flags=re.MULTILINE)
        return int(m.group(1)) if m else None

    total = kb("MemTotal")
    avail = kb("MemAvailable")

    return (
        round(total / 1024 / 1024, 3) if total else None,
        round(avail / 1024 / 1024, 3) if avail else None,
    )


def detect():
    system = platform.system()
    machine = platform.machine()
    logical = os.cpu_count() or 1

    model = platform.processor() or "UNKNOWN"
    flags = set()

    if system == "Linux":
        model, flags = _cpu_linux()

    mem_total = None
    mem_available = None
    if system == "Linux":
        mem_total, mem_available = _memory_linux()

    codespaces = bool(os.environ.get("CODESPACES"))
    container = (
        os.path.exists("/.dockerenv")
        or "container" in _read_text("/proc/1/cgroup").lower()
        or codespaces
    )

    avx2 = "avx2" in flags
    avx512 = any(x.startswith("avx512") for x in flags)

    if avx512:
        profile = "X512_AVX512"
    elif avx2:
        profile = "X512_AVX2"
    else:
        profile = "X512_PORTABLE"

    return {
        "cpu_model": model,
        "architecture": machine,
        "logical_cpus": logical,
        "avx2": avx2,
        "avx512": avx512,
        "memory_total_gib": mem_total,
        "memory_available_gib": mem_available,
        "codespaces": codespaces,
        "container": container,
        "profile": profile,
        "persistent_storage_used": False,
        "background_service": False,
    }


def report():
    h = detect()

    print("DELTA HARDWARE DISCOVERY")
    print(f"CPU_MODEL={h['cpu_model']}")
    print(f"ARCHITECTURE={h['architecture']}")
    print(f"CPU_LOGICAL={h['logical_cpus']}")
    print(f"ISA_AVX2={'YES' if h['avx2'] else 'NO'}")
    print(f"ISA_AVX512={'YES' if h['avx512'] else 'NO'}")
    print(f"MEMORY_TOTAL_GiB={h['memory_total_gib']}")
    print(f"MEMORY_AVAILABLE_GiB={h['memory_available_gib']}")
    print(f"ENV_CODESPACES={'YES' if h['codespaces'] else 'NO'}")
    print(f"ENV_CONTAINER={'YES' if h['container'] else 'NO'}")
    print(f"DELTA_PROFILE={h['profile']}")
    print("DETECTION_MODE=ON_DEMAND")
    print("BACKGROUND_SERVICE=NO")
    print("PERSISTENT_CACHE=NO")
    print("DISK_DATA_CREATED=NO")
    print("HARDWARE_DETECTION=PASSED")

    return h


if __name__ == "__main__":
    report()
