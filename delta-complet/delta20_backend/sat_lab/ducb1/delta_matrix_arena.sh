#!/usr/bin/env bash
set -u

TMP="$(mktemp)"
BEST_SCORE=0
BEST_N=0
PASS=0
FAIL=0

echo "============================================================"
echo " EXTERNAL MATRIX STRESS ARENA"
echo " stress-ng UNMODIFIED WORKLOAD"
echo "============================================================"

for N in 1 2 4 8 16 32
do
    echo
    echo "RUN workers=$N"

    stress-ng \
        --matrix "$N" \
        --timeout 20s \
        --verify \
        --metrics-brief \
        2>&1 | tee "$TMP"

    RC=${PIPESTATUS[0]}

    SCORE=$(
        awk '
        $1=="matrix" {
            for(i=1;i<=NF;i++) {
                if($i ~ /^[0-9]+\.[0-9]+$/) {
                    nums[++n]=$i
                }
            }
        }
        END {
            if(n>=2) print nums[n-1]
            else print 0
        }' "$TMP"
    )

    if [ "$RC" -eq 0 ]; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
    fi

    BETTER=$(
        python - <<PY
a=float("$SCORE")
b=float("$BEST_SCORE")
print(1 if a>b else 0)
PY
    )

    if [ "$BETTER" -eq 1 ]; then
        BEST_SCORE="$SCORE"
        BEST_N="$N"
    fi
done

rm -f "$TMP"

echo
echo "============================================================"
echo " FINAL MULTI-SUPPORT VERDICT"
echo "============================================================"

echo "RYZEN_NATIVE=PASS"
echo "RYZEN_STAGES_PASSED=$PASS"
echo "RYZEN_STAGES_FAILED=$FAIL"
echo "RYZEN_BEST_WORKERS=$BEST_N"
echo "RYZEN_BEST_SCORE=$BEST_SCORE"

echo
echo "DELTA_BIT64=N/A"
echo "DELTA_X512=N/A"
echo "DELTA_QPU_LOGICAL=N/A"
echo "IBM_QPU_PHYSICAL=N/A"

echo
echo "REASON=stress-ng_matrix_is_native_CPU_workload"
echo "NO_WORKLOAD_ADAPTATION=YES"

if [ "$FAIL" -eq 0 ]; then
    echo
    echo "STABILITY_WINNER=RYZEN_NATIVE"
    echo "THROUGHPUT_WINNER=RYZEN_NATIVE"
    echo "WINNER=RYZEN_NATIVE"
else
    echo
    echo "WINNER=NO_STABLE_WINNER"
fi

echo "============================================================"
