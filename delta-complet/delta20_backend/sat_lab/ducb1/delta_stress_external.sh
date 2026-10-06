#!/usr/bin/env bash
set -u

LOG="delta_stress_external_$(date +%Y%m%d_%H%M%S).log"

echo "====================================================================" | tee "$LOG"
echo " EXTERNAL STRESS-NG — MULTI-SUPPORT BASELINE"                         | tee -a "$LOG"
echo "====================================================================" | tee -a "$LOG"

echo "DATE=$(date -Iseconds)"                                               | tee -a "$LOG"
echo "CPU=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"      | tee -a "$LOG"
echo "LOGICAL_CPUS=$(nproc)"                                                | tee -a "$LOG"
echo "MEMORY:"                                                              | tee -a "$LOG"
free -h                                                                      | tee -a "$LOG"

echo                                                                    | tee -a "$LOG"
stress-ng --version                                                       | tee -a "$LOG"

for N in 1 2 4 8 16 32
do
    echo                                                                    | tee -a "$LOG"
    echo "====================================================================" | tee -a "$LOG"
    echo " STAGE=${N}_WORKERS"                                               | tee -a "$LOG"
    echo "====================================================================" | tee -a "$LOG"

    START=$(date +%s.%N)

    stress-ng \
        --cpu "$N" \
        --cpu-method all \
        --timeout 60s \
        --verify \
        --metrics \
        --times \
        --timestamp \
        2>&1 | tee -a "$LOG"

    RC=${PIPESTATUS[0]}

    END=$(date +%s.%N)

    WALL=$(python - <<PY
a=float("$START")
b=float("$END")
print(f"{b-a:.6f}")
PY
)

    echo "STAGE_WORKERS=$N"       | tee -a "$LOG"
    echo "STAGE_RC=$RC"           | tee -a "$LOG"
    echo "STAGE_WALL_SEC=$WALL"   | tee -a "$LOG"

    if [ "$RC" -ne 0 ]; then
        echo "STAGE_STATUS=FAIL"  | tee -a "$LOG"
        echo "STOP_ON_FAILURE=YES"| tee -a "$LOG"
        break
    else
        echo "STAGE_STATUS=PASS"  | tee -a "$LOG"
    fi

    sleep 5
done

echo                                                                    | tee -a "$LOG"
echo "====================================================================" | tee -a "$LOG"
echo " EXTERNAL_STRESS_COMPLETE"                                            | tee -a "$LOG"
echo "====================================================================" | tee -a "$LOG"

echo "LOG=$LOG"
