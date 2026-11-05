#!/bin/bash
S=$(dirname "$0")
for n in 4 8 16 32; do for m in 2 4; do seq=$((n*m))
  for v in upstream fork; do
    t0=$(date +%s); "$S/run_one.sh" $v $n $seq "$S" || echo "RUN FAILED $v $n $seq"
    echo "$v ${n}x${n} seq$seq done in $(( $(date +%s)-t0 ))s"
  done
done; done
echo ALL DONE
