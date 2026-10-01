#!/bin/bash
cd "$(dirname "$0")"
LOG="fullstack_$(date +%Y%m%d_%H%M%S).log"
{
echo "=== DELTA FULL STACK BENCHMARK $(date -Iseconds) ==="
echo "--- LAYER 0 : HARDWARE ---"
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket"
gcc -O2 -pthread dspc_pingpong.c -o dspc_pingpong && gcc -O2 cache_l200.c -o cache_l200 && gcc -O2 layers_bench.c -o layers_bench || { echo "BUILD=FAIL"; exit 1; }
echo "--- LAYER 1 : CORE BRIDGE / DSPC PING-PONG (3 runs) ---"
for i in 1 2 3; do ./dspc_pingpong 0 1 | grep -E "RTT_NS|EXCHANGES|VALIDATION"; done
echo "--- LAYER 2-3 : CLOUD 1T ADDRESSING + VIRTUAL SILICON ---"
./layers_bench
echo "--- LAYER 4 : CACHE L1-L200 ---"
./cache_l200
echo "--- LAYER 5 : IBM QPU ATTESTATION (live) ---"
if [ -n "$IQP_API_TOKEN" ] && [ -n "$IQP_INSTANCE_CRN" ]; then
 T0=$(date +%s%N); OUT=$(python3 delta_backend/ibm_hardware_attest.py 2>&1 | tail -1); T1=$(date +%s%N)
 echo "$OUT" | python3 -c "import sys,json;d=json.loads(sys.stdin.read());print('IBM_BACKEND=%s PHYSICAL_QUBITS=%s ATTESTATION=%s'%(d.get('quantumBackend'),d.get('physicalQubits'),d.get('attestation')))" 2>/dev/null || echo "IBM_RAW=$OUT"
 echo "IBM_ATTESTATION_WALL_MS=$(( (T1-T0)/1000000 ))"
else echo "IBM_ATTESTATION=SKIPPED (IQP_* absent)"; fi
echo "--- SUMMARY ---"
} 2>&1 | tee "$LOG"
F=$(grep -c "FAIL" "$LOG"); echo "FULLSTACK_VALIDATION=$([ "$F" = "0" ] && echo OK || echo "FAIL($F)")" | tee -a "$LOG"
echo "LOG=$LOG"
