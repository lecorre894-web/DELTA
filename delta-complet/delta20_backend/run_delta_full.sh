#!/usr/bin/env bash
set -e

CC="${CC:-cc}"
CFLAGS="-O3"

echo "======================================================"
echo "       DELTA FULL CHAIN — V10 → V15"
echo "======================================================"
echo "CPU_CORE_ROLE=BRIDGE"
echo "V15_QBIT_ROLE=EXECUTION_LAYER"
echo

run_component () {
    VERSION="$1"
    SOURCE="$2"
    BINARY="$3"

    echo
    echo "------------------------------------------------------"
    echo " DELTA $VERSION"
    echo " SOURCE=$SOURCE"
    echo "------------------------------------------------------"

    if [ ! -f "$SOURCE" ]; then
        echo "ERROR: source absent: $SOURCE"
        exit 1
    fi

    echo "[BUILD] $SOURCE"

    "$CC" $CFLAGS "$SOURCE" -o "$BINARY" -lm

    echo "[RUN] $BINARY"
    "./$BINARY"

    RC=$?

    if [ "$RC" -ne 0 ]; then
        echo "DELTA_COMPONENT=$VERSION FAILED"
        exit "$RC"
    fi

    echo "DELTA_COMPONENT=$VERSION PASSED"
}

run_component \
    "V10 EVENT STATE" \
    "delta_v10_10m_event_state.VALIDATED.c" \
    ".delta_full_v10"

run_component \
    "V12.1 FULL ENGINE" \
    "delta_v12_1_FULL_ENGINE.VALIDATED.c" \
    ".delta_full_v12_1"

run_component \
    "V14 CLOUD" \
    "delta_v14_cloud.c" \
    ".delta_full_v14_cloud"

run_component \
    "V14 QUANTUM CLOUD" \
    "delta_v14_quantum_cloud.c" \
    ".delta_full_v14_quantum"

run_component \
    "V14 WORLD" \
    "delta_v14_world.c" \
    ".delta_full_v14_world"

run_component \
    "V15 1T QBITS" \
    "delta_v15_1T_qbits.c" \
    ".delta_full_v15"

run_component \
    "V15 1T QBITS METRICS" \
    "delta_v15_1T_qbits_metrics.c" \
    ".delta_full_v15_metrics"

echo
echo "======================================================"
echo " DELTA FULL CHAIN = PASSED"
echo " V10 -> V12.1 -> V14 -> V15"
echo "======================================================"
