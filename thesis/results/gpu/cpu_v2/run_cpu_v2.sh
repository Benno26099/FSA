#!/bin/bash
# CPU baseline v2 on guppy: warm-up, single-threaded main run, then one default-threads run.
# Run in ~/fsa-gpu (run_cases_gpu.py, cases.py, results/*.jsonl); outputs go to out/cpu_v2/
set -u
cd ~/fsa-gpu
source ~/gpu-env/bin/activate
mkdir -p out/cpu_v2
{ date; uptime; who; top -bn1 -o %CPU | head -15; } > out/cpu_v2/machine_state_before.txt

one() {   # one() <suffix> : all 8 case files
    for N in 4 8 16 32; do
        for s in scaling_cases_sa$N blas3_cases_sa$N; do
            python run_cases_gpu.py --measure cpu --modes fp32 --cases results/$s.jsonl \
                --out out/cpu_v2/${s}_cpu_v2$1.csv | grep -v ' done$'
        done
    done
}

# warm-up (CPU clocks, caches, page cache): one throwaway pass over the 32x32 scaling cases
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_cases_gpu.py --measure cpu \
    --modes fp32 --cases results/scaling_cases_sa32.jsonl --out out/cpu_v2/warmup.csv > /dev/null

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 one ""
one "_defthreads"

{ date; uptime; who; top -bn1 -o %CPU | head -15; } > out/cpu_v2/machine_state_after.txt
echo ALL DONE
