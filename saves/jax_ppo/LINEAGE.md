# Level 3 PPO training lineage

Recovered on 2026-11-21 from each run's `train.log`, the `systemd --user` journal, and snapshot
`metadata.json`. Every stage was policy-initialized from the previous one (optimizer, EMA, PRNG,
environment reset). Results marked `~` are 500-episode training validations; the others are the
5000-episode deterministic evaluation (seeds from 30000) of the final EMA policy.

| # | Profile | Initialized from | Budget | Result |
|---|---|---|---|---|
| 1 | level3_v4_commit | random parameters | 50M (`level3_v4_commit_pilot50m`) | — |
| 2 | level3_v4_commit | #1 `latest` (raw) | snapshot at 100M: `level3_v4_commit_r1_raw100m` | ~21% |
| 3 | level3_v6_scaled | #2 snapshot (raw), `--rescale-critic 0.02` | snapshot at 120M: `level3_v6_scaled_r1_ema120m` | ~50% |
| 4 | level3_v7_margin | #3 snapshot (EMA) | best at 30M: `level3_v7_margin_r1_ema_best` | ~55% |
| 5 | level3_v8_clearance | #4 snapshot (EMA) | at 30M: `level3_v8_diag_ckpt` | ~56% |
| 6 | level3_v9_slow | #5 snapshot (EMA) | best at 40M: `level3_v9_ema_best` | ~57% |
| 7 | level3_v10_full | #6 snapshot (EMA) | best at 50M: `level3_v10_raw_best` | ~59% |
| 8 | level3_v11_explore | #7 snapshot (raw) | best at 60M: `level3_v11_ema_best` | ~62% |
| 9 | level3_v12_refine | #8 snapshot (EMA) | 500M: `level3_sr93.4_lap7.66` (80M copy: `level3_sr73.6_lap8.42`) | 93.4% / 7.66 s |
| 10 | level3_v13_fast | #9 final EMA | 150M: `level3_sr91.8_lap6.28` | 91.8% / 6.28 s |
| 11 | level3_v14_faster | #10 final EMA | 200M: `level3_sr91.8_lap5.78` | 91.8% / 5.78 s |
| 12 | level3_v15_race | #11 final EMA | 250M: `level3_sr91.5_lap5.42` | 91.5% / 5.42 s |
| 13 | level3_v16_lap | #12 final EMA | 300M: `level3_sr92.3_lap5.32` | 92.3% / 5.32 s |
| 14 | level3_v17_sprint | #13 final EMA | 400M: `level3_sr86.4_lap4.76` | 86.4% / 4.76 s |
| 15a | level3_v18_refine | #14 final EMA | 300M: `level3_sr90.2_lap4.90` | 90.2% / 4.90 s |
| 15b | level3_v18_plr (branch) | #14 final EMA | 300M: `level3_sr91.9_lap5.32` | 91.9% / 5.32 s |
| 16 | level3_v19_tight | #15a final EMA | 400M: `level3_sr90.7_lap4.80` | 90.7% / 4.80 s |
| 17 | level3_v20_memory (25-frame history, zero-padded from 4) | #16 final EMA | 400M: `level3_sr91.1_lap4.74` | 91.1% / 4.74 s |
| 18 | level3_v21_fullinfo (sensor range 10 m: true track visible from reset; diagnostic only, not Level 3 rules; metadata profile name `level3_v21_nominal`) | #17 final EMA | 400M: `level3_fullinfo_sr91.3_lap4.32` | 91.3% / 4.32 s (full-info evaluation) |
| 19 | level3_v20_memory, 5-pass track `[1, 2, 3, 4, 2]`, Crazyflow 0.4 | #17 final EMA (62.7% / 6.20 s on this track before training) | 400M: `level3_v20_memory_5pass` | 87.7% / 6.12 s |
| 20 | level3_v20_memory, 5-pass track | #19 final EMA | 400M: `level3_v20_memory_5pass_r2` | 89.0% / 6.10 s |

Dead ends: `level3_v2_explore_pilot50m` (random init, entropy blow-up) and `level3_v5_safe_r1`
(from #2, optimizer froze); #3 re-branched from #2.

Logs record the pre-rename folder names: `level3_v12_refine_r1` = #9, `level3_v13_fast_r1` = #10,
`level3_v14_faster_r1` = #11, `level3_v15_race_r1` = #12, `level3_v16_lap_r1` = #13,
`level3_v17_sprint_r1` = #14, `level3_v18_refine_r1` = #15a, `level3_v18_plr_r1` = #15b,
`level3_v19_tight_r1` = #16, `level3_v20_memory_r1` = #17, `level3_v21_nominal_r1` = #18, `level3_v12_ema80m` = the 80M copy of #9.

Snapshots #2-#8 are copies of a run's `latest`/`best_*`; their profile and step count match the
source run, but the copy itself was not logged. Runs #1-#15 used uncommitted code at the time;
GPU training is not bitwise deterministic. Runs #19-#20 used commit `429ecdb`.
