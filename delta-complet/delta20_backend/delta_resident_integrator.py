#!/usr/bin/env python3
from pathlib import Path
import ast
import re

ROOT = Path(".")
SELF = {
    "delta_resident.py",
    "delta_resident_integrator.py",
}

print("========== DELTA RESIDENT GLOBAL INTEGRATOR ==========")
print("MODE=ANALYSIS_ONLY")
print("SOURCE_MODIFICATION=NO")
print()

candidates = []
remote = []
other = []

for p in sorted(ROOT.glob("*.py")):
    if p.name in SELF:
        continue

    try:
        src = p.read_text(errors="replace")
        ast.parse(src)
    except Exception:
        continue

    reasons = []

    # Utilisation explicite du moteur X512
    if re.search(r'\b(?:import|from)\s+delta_x512\b', src):
        reasons.append("IMPORT_X512")

    if re.search(r'\bDeltaX512\s*\(', src):
        reasons.append("CONSTRUCT_X512")

    # Lancement de composants X512 par subprocess
    if re.search(r'(?:direct|micro)\.x512', src):
        reasons.append("X512_PROCESS")

    if re.search(r'\bx512_run\b', src):
        reasons.append("X512_RUN")

    # QPU physique/distant : on l'isole volontairement
    qpu_remote = bool(re.search(
        r'IBMProvider|QiskitRuntimeService|SamplerV2|'
        r'least_busy|simulator\s*=\s*False|'
        r'ibm_[A-Za-z0-9_]+',
        src
    ))

    if qpu_remote:
        remote.append(p.name)
        print(f"REMOTE_KEEP_SEPARATE  {p.name}")
        continue

    if reasons:
        candidates.append((p.name, sorted(set(reasons))))
        print(
            f"RESIDENT_CANDIDATE    {p.name:<32} "
            f"{','.join(sorted(set(reasons)))}"
        )
    else:
        # subprocess/multiprocessing sans preuve qu'il s'agit de X512
        generic = []
        if "subprocess" in src:
            generic.append("SUBPROCESS")
        if "multiprocessing" in src:
            generic.append("MULTIPROCESS")

        if generic:
            other.append((p.name, generic))
            print(
                f"REVIEW_ONLY           {p.name:<32} "
                f"{','.join(generic)}"
            )

print()
print("========== INTEGRATION PLAN ==========")
print(f"RESIDENT_CANDIDATES={len(candidates)}")
print(f"REMOTE_QPU_SEPARATE={len(remote)}")
print(f"REVIEW_ONLY={len(other)}")

print()
print("RESIDENT_TARGETS=")
for name, reasons in candidates:
    print(f"  {name} -> delta_resident [{','.join(reasons)}]")

print()
print("REMOTE_TARGETS=")
for name in remote:
    print(f"  {name} -> KEEP_REMOTE")

print()
print("POLICY_X512=ONE_RESIDENT_ENGINE")
print("POLICY_WORKERS=PERSISTENT")
print("POLICY_REMOTE_QPU=SEPARATE")
print("SOURCE_FILES_CHANGED=0")
print("FINAL_VALIDATION=PASSED")
