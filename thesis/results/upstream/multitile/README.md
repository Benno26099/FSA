<!-- AI assistance: run and summarised by Claude (Anthropic; Claude Code, Claude Opus 5.5) at Ben Colbourne's request, 2026-11-05. -->
# Multi-tile FlashAttention: extended vs original FSA

`main.py --seq_q S --seq_kv S` with S = 2n and 4n at n = 4, 8, 16, 32, on the extended build
(`FSA{n}X{n}Fp16Config-debug`, 14 Oct) and the original build (`FSA{n}X{n}Fp16UpstreamConfig-debug`,
f4051d9, 20 Oct, run with f4051d9's own Python), same seed (0) and driver. Scripts: `run_one.sh`,
`run_all.sh`. Single-tile 4x4 reproduced the thesis table first (89 / 96 cycles, same MAE).

Result (`multitile_attention.csv`): output memory dumps byte-identical in all 8 configurations.
Timing difference +3 to +20 cycles (0.4-4.6%); mxActive difference = number of LOAD_STATIONARY
instructions at 8x8 and above (+4 for 2x2 blocks, +16 for 4x4), one fewer at 4x4.

Anomaly: at 32x32, S = 128 (4x4 blocks) BOTH builds give MAE 0.0605, MaxErr 11.5: 24 of 4,096
outputs wrong, all in Q block 1 (rows 44-63), columns 11 and 27, about 22x the reference.
Identical bytes in both builds, so it is present in the original FSA; not investigated.
