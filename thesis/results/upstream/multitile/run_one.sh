#!/bin/bash
# usage: run_one.sh <fork|upstream> <n> <seq> <outroot>
set -e
ver=$1; n=$2; seq=$3; out=$4/${ver}_${n}x${n}_seq${seq}
PY=/home/ben/chipyard/chipyard-fsa/generators/fsa/python/.venv/bin/python
SIM=/home/ben/chipyard/chipyard-fsa/sims/verilator
if [ $ver = fork ]; then dir=/home/ben/chipyard/chipyard-fsa/generators/fsa/python; cfg=FSA${n}X${n}Fp16Config
else dir=/home/ben/fsa_upstream_f4051d9/python; cfg=FSA${n}X${n}Fp16UpstreamConfig; fi
mkdir -p $out
cd $dir
$PY -c "import fsa; print('fsa package:', fsa.__file__)" > $out.log
$PY main.py --config $cfg --seq_q $seq --seq_kv $seq --output_dir $out --build_dir $SIM \
  --simulator_bin $SIM/simulator-chipyard.harness-${cfg}-debug >> $out.log 2>&1
