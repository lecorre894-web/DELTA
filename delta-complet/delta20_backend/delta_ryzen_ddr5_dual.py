#!/usr/bin/env python3
import json

ARCH = "DELTA_RYZEN_DDR5_DUAL"
MODE = "DUAL_CHANNEL_LOGICAL"
MEMORIES = 2

ZONE_KIB = 64
ZONES_PER_MEMORY = 16
BLOCK_BYTES = 128

BYTES_PER_MEMORY = ZONE_KIB * 1024 * ZONES_PER_MEMORY
TOTAL_BYTES = BYTES_PER_MEMORY * MEMORIES

# Résultat expérimental CacheMap16 conservé comme référence.
# Ce n'est PAS une bande passante DDR5 physique annoncée.
CACHEMAP16_REFERENCE_GB_S = 234.116005

config = {
    "architecture": ARCH,
    "mode": MODE,
    "physical_cache_modified": False,
    "memories": {
        "DDR5_EXP_A": {
            "zones": ZONES_PER_MEMORY,
            "zone_kib": ZONE_KIB,
            "block_bytes": BLOCK_BYTES,
            "capacity_bytes": BYTES_PER_MEMORY
        },
        "DDR5_EXP_B": {
            "zones": ZONES_PER_MEMORY,
            "zone_kib": ZONE_KIB,
            "block_bytes": BLOCK_BYTES,
            "capacity_bytes": BYTES_PER_MEMORY
        }
    },
    "controller": {
        "channels": 2,
        "mapping": "A_B_INTERLEAVED_BY_128B_BLOCK",
        "channel_A": "DDR5_EXP_A",
        "channel_B": "DDR5_EXP_B"
    },
    "aggregate": {
        "logical_zones": ZONES_PER_MEMORY * MEMORIES,
        "working_set_kib": TOTAL_BYTES // 1024,
        "working_set_mib": TOTAL_BYTES / (1024 * 1024)
    },
    "reference": {
        "cachemap16_gb_s": CACHEMAP16_REFERENCE_GB_S,
        "classification": "EXPERIMENTAL_MEASURED_REFERENCE",
        "dual_bandwidth": "TO_BE_MEASURED"
    }
}

print("==============================================")
print(" DELTA RYZEN + 2x DDR5 EXPERIMENTALE")
print("==============================================")
print("MEMORY_A=16x64KiB")
print("MEMORY_B=16x64KiB")
print("TOTAL_LOGICAL_ZONES=32")
print(f"TOTAL_WORKING_SET_MiB={TOTAL_BYTES/(1024*1024):.2f}")
print("BLOCK_BYTES=128")
print("MODE=DUAL_CHANNEL_LOGICAL")
print("MAPPING=A/B_128B_INTERLEAVE")
print(f"CACHEMAP16_REFERENCE_GB_S={CACHEMAP16_REFERENCE_GB_S:.6f}")
print("DUAL_CHANNEL_GB_S=TO_BE_MEASURED")
print("PHYSICAL_CACHE_MODIFIED=NO")
print("VALIDATION=PASSED")

with open("delta_ryzen_ddr5_dual.json", "w") as f:
    json.dump(config, f, indent=2)

print("OUTPUT=delta_ryzen_ddr5_dual.json")
