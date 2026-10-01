#!/usr/bin/env bash
set -e

echo "======================================"
echo "       DELTA V14 / V15 LAUNCHER"
echo "======================================"
echo
echo "1 - V14 Cloud"
echo "2 - V14 Quantum Cloud"
echo "3 - V14 World"
echo "4 - V15 1T Qbits"
echo "5 - V15 1T Qbits Metrics"
echo
printf "Choix : "
read choice

case "$choice" in

1)
    echo "Compilation V14 Cloud..."
    cc -O3 delta_v14_cloud.c -o delta_v14_cloud -lm
    echo
    ./delta_v14_cloud
    ;;

2)
    echo "Compilation V14 Quantum Cloud..."
    cc -O3 delta_v14_quantum_cloud.c -o delta_v14_quantum_cloud -lm
    echo
    ./delta_v14_quantum_cloud
    ;;

3)
    echo "Compilation V14 World..."
    cc -O3 delta_v14_world.c -o delta_v14_world -lm
    echo
    ./delta_v14_world
    ;;

4)
    echo "Compilation V15 1T Qbits..."
    cc -O3 delta_v15_1T_qbits.c -o delta_v15_1T_qbits -lm
    echo
    ./delta_v15_1T_qbits
    ;;

5)
    echo "Compilation V15 Metrics..."
    cc -O3 delta_v15_1T_qbits_metrics.c -o delta_v15_1T_qbits_metrics -lm
    echo
    ./delta_v15_1T_qbits_metrics
    ;;

*)
    echo "Choix invalide."
    exit 1
    ;;
esac
