# Level 3 ablations (seed 42, single run each)

All numbers: deterministic 5000-episode evaluation of the final EMA policy (seeds from 30000).
Single seed: differences below ~1 percentage point of success or ~0.05 s of median lap are noise.

## Fine-tune reward ablations

Start: `level3_sr91.1_lap4.74` EMA, 200M steps, V19 reward with 25-frame history; one term removed.

| Run | Success | CI low | Median lap | Gate hits | Pole hits |
|---|---|---|---|---|---|
| ft_base (all terms) | 90.5% | 89.7% | 4.76 s | 7.9% | 1.5% |
| no route_progress | 90.8% | 90.0% | 4.76 s | 7.9% | 1.2% |
| no gate_risk | 91.1% | 90.3% | 4.78 s | 7.6% | 1.3% |
| no obstacle_risk | 91.0% | 90.1% | 4.72 s | 7.4% | 1.6% |
| no obstacle_clearance | 90.9% | 90.1% | 4.76 s | 7.6% | 1.4% |
| no overspeed | 91.5% | 90.7% | 4.78 s | 6.8% | 1.6% |
| no time | 92.8% | 92.1% | 5.12 s | 6.4% | 0.7% |

## From-scratch baselines

| Run | Reward / settings | Success | Notes |
|---|---|---|---|
| base_v10_scratch | V10 reliability reward, curriculum off | 0.0% | 99.8% timeouts: hovers; crash (250) costs more than timeout (60) |
| base | v4 settings (V4 reward, curriculum on) | 30.1% | 2.48 gates/episode; 40.7% gate and 29.1% pole collisions |

## From-scratch observation ablations (v4 settings, 300M steps, actor input zeroed)

| Run | Success | CI low | Median lap | Gates/episode | Notes |
|---|---|---|---|---|---|
| base | 30.1% | 28.9% | 6.54 s | 2.48 | |
| no_history | 31.9% | 30.6% | 6.44 s | 2.53 | same as base |
| no_direct_geometry | 18.0% | 17.0% | 6.64 s | 2.26 | slower learning curve throughout |
| no_obstacles | 0.0% | 0.0% | — | 0.00 | never passed a gate; entropy 11.5, value loss ~0: collapsed |
| near_gates | 0.0% | 0.0% | — | 0.00 | never passed a gate; entropy 11.4: collapsed |
| no_visited | 30.3% | 29.0% | 6.62 s | 2.46 | same as base |

`no_obstacles` and `near_gates` never passed even the first gate, although `near_gates` still
sees the current gate. Both collapsed into a no-learning-signal regime (policy entropy exploded,
value loss near zero) before 40M steps, while every other run passed gates by 40M. This is
consistent with a failed start rather than with the removed information being necessary; it needs
more seeds before any conclusion.

## From-scratch reward ablations (v4 settings, one v4 change reverted)

| Run | Change from base | Success | CI low | Median lap | Gates/episode | Gate hits | Pole hits |
|---|---|---|---|---|---|---|---|
| base | — | 30.1% | 28.9% | 6.54 s | 2.48 | 40.7% | 29.1% |
| gate10 | gate reward 100 -> 10 | 41.6% | 40.2% | 6.06 s | 2.73 | 28.0% | 30.0% |
| failure_per_gate | failure_per_gate 0 -> 30 | 31.7% | 30.4% | 7.12 s | 2.49 | 40.0% | 27.9% |
| timeout_bootstrap | timeout no longer terminal | 23.4% | 22.2% | 6.24 s | 2.33 | 48.5% | 28.0% |
| no_time | time penalty 0.03 -> 0 | 28.0% | 26.8% | 6.76 s | 2.40 | 42.7% | 29.1% |

No single reverted reward change stops learning from scratch, and a smaller gate reward learned
better. The V2 runs that stayed at 0% also used different PPO settings (learning rate 5e-5 -> 1e-5,
target KL 0.01) than the v4 profile (3e-4 -> 3e-5, target KL 0.03), so the v4 breakthrough is not
attributable to the reward changes alone.

## Extra seeds (from scratch, 300M steps)

| Run | Seed 42 | Seed 1 | Seed 2 |
|---|---|---|---|
| base | 30.1% | 0.0% | 21.2% |
| no_obstacles | 0.0% | 0.0% | 0.0% |
| near_gates | 0.0% | 0.0% | 0.0% |
| gate10 | 41.6% | 0.0% | — |
| v2_reward (V2 reward, v4 PPO settings) | 14.1% | 0.0% (1.24 gates/episode) | — |

From-scratch starts fail outright on some seeds (base seed 1 never passed a gate), so single-seed
differences between scratch runs are not reliable, including no_direct_geometry (18.0%) and gate10
(41.6%). no_obstacles and near_gates failed on all three seeds, so removing that information at
least makes the start much harder. The V2 reward does learn with v4 PPO settings (14.1%), so the
original V2 0% was largely due to the PPO settings.
