#!/bin/bash
CORES=$(nproc --all)
echo "Cœurs détectés par le container : $CORES"

export OMP_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export NUMEXPR_NUM_THREADS=$CORES
export MAKEFLAGS="-j$CORES"
export CMAKE_BUILD_PARALLEL_LEVEL=$CORES

echo "export OMP_NUM_THREADS=$CORES" >> ~/.bashrc
echo "export OPENBLAS_NUM_THREADS=$CORES" >> ~/.bashrc
echo "export MKL_NUM_THREADS=$CORES" >> ~/.bashrc
echo "export NUMEXPR_NUM_THREADS=$CORES" >> ~/.bashrc
echo "export MAKEFLAGS="-j$CORES"" >> ~/.bashrc
echo "export CMAKE_BUILD_PARALLEL_LEVEL=$CORES" >> ~/.bashrc

ulimit -u unlimited 2>/dev/null

echo "--- Vérification finale ---"
nproc --all
nproc
lscpu | grep -E "^CPU(s)|Thread|Core|Socket"
