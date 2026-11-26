# Agent Guide: Level 2 baseline and Level 3 structured JAX PPO

## Scope and current objective

This file applies to the entire repository. Read it before changing the reinforcement-learning
implementation.

The active development branch is `rl-jax-dev`. It is based on `065a6ec` and contains the fused JAX
PPO implementation added in `3ba81b9`, full-budget training support added in `ae8d2a2`, and the
Level 2 controller configuration added in `6bd683c`.

The active research objective is now a reliability-first structured V2 policy for the original,
fully randomized Level 3 environment. The frozen design and current measured status are in
`lsy_drone_racing/rl_training/LEVEL3_V2.md`; treat that document as authoritative for V2 feature,
reward, curriculum, network, and evaluation semantics. Endpoint-only segment curricula preserved
the learned crossing primitive but failed to teach approach and turn acquisition. The active
experiment is `level3_v3_route_corridor`: policy-initialize it from the raw mixed-curriculum
checkpoint, reset all other state and the 500M clock, and run only its authorized 382-update 50M
pilot before deciding whether to continue. The existing Level 2 V1 baseline remains supported and
must not regress.

## User constraints

- Do not merge `25ws-main`. It is a reference for behavior and hyperparameters only.
- Do not push branches, create a pull request, or publish checkpoints unless explicitly requested.
- Preserve existing PyTorch code and dependencies, but the production JAX PPO path must not import
  Torch.
- Do not migrate or convert the old PyTorch checkpoint.
- Store checkpoints, logs, evaluations, and performance reports only under ignored
  `saves/jax_ppo/` paths.
- Do not silently change the competition track, randomization bounds, disturbances, sensor range,
  or success definition to inflate the result.
- Do not claim a success rate from rollout metrics or a 100-episode selection run. The headline
  result must come from a deterministic, fixed-seed evaluation with confidence intervals.
- The user requested no early stopping for full-budget experiments unless they explicitly change
  that decision.
- Future long-running jobs must be launched as a detached `systemd --user` service so closing SSH
  or Codex does not terminate training. Record the unit name, PID, command, log paths, checkpoint
  path, and stop command before leaving it running.
- QuotaHalo was intentionally removed from this machine. Do not reintroduce its hooks, service, or
  files.

## Repository and runtime facts

- Repository: `/media/ymlee/yuming_data/repos/lsy_drone_racing`
- Git remotes: `origin` is the user's fork; `upstream` is `utiasDSL/lsy_drone_racing`.
- Supported Crazyflow range: `>=0.4.0,<0.5.0`; the current lock resolves Crazyflow 0.4.0.
- Training hardware used so far: one NVIDIA RTX 4070.
- Production training environment: `gpu`.
- CPU test environment: `tests`.
- CUDA test environment: `gpu-tests`.
- PyTorch/JAX comparison environment: `rl-benchmark-gpu`.
- The repository is on an external Samsung T7. The machine was configured to use `usb-storage`
  instead of UAS for this device after prior I/O failures. Treat new filesystem or Git object
  errors as storage issues first; do not use destructive Git recovery commands.

The dependency and task definitions are in `pyproject.toml`; keep `pixi.lock` synchronized when
dependencies change.

## Code map

- `lsy_drone_racing/rl_training/config.py`: immutable PPO profiles and reward coefficients.
- `lsy_drone_racing/rl_training/observations.py`: pure-JAX observation construction and temporal
  state.
- `lsy_drone_racing/rl_training/rewards.py`: pure-JAX shaping, termination classification, and
  episode accounting.
- `lsy_drone_racing/rl_training/networks.py`: independent Flax actor/critic MLPs and squashed
  Gaussian utilities.
- `lsy_drone_racing/rl_training/algorithm.py`: GAE, Optax optimizer, and compiled PPO update.
- `lsy_drone_racing/rl_training/environment.py`: fused RaceCore transition, reset, rollout, and
  aggregation logic.
- `lsy_drone_racing/rl_training/runtime.py`: environment/model construction.
- `lsy_drone_racing/rl_training/training.py`: CLI, checkpointing, validation, early-stop behavior,
  and logging.
- `lsy_drone_racing/rl_training/evaluation.py`: deterministic fixed-seed evaluation and Wilson
  intervals.
- `lsy_drone_racing/rl_training/checkpoint.py`: exact-resume and policy-only serialization.
- `lsy_drone_racing/rl_training/benchmark.py`: JAX throughput measurement and 2x speed gate.
- `lsy_drone_racing/rl_training/torch_reference.py`: isolated local PyTorch reference only.
- `lsy_drone_racing/control/attitude_jax_ppo.py`: deterministic deployment/simulation controller.
- `lsy_drone_racing/rl_training/level3_*.py`: versioned Level 3 V2 config, structured
  observations/network, asymmetric PPO, reward, fused environment, checkpoint, evaluation,
  benchmark, and training lifecycle.
- `lsy_drone_racing/rl_training/level3_segment_reset.py`: pure-JAX training-only mixed original/
  gate-segment state generation.
- `lsy_drone_racing/control/attitude_jax_ppo_level3.py`: V2 Actor-only deterministic controller.
- `lsy_drone_racing/rl_training/LEVEL3_V2.md`: exact V2 implementation contract and current gates.
- `config/level2.toml`: currently selects the JAX controller, `fast/latest`, and attitude mode.
- `tests/unit/rl_training/test_jax_ppo.py`: mathematical and serialization tests.
- `tests/integration/rl_training/test_jax_ppo_environment.py`: CPU and 1024-world GPU smoke tests.

## Hot-path architecture: preserve these invariants

The speedup comes from keeping the entire rollout on the accelerator, not merely replacing a Torch
MLP with a Flax MLP.

- Use `VecDroneRaceEnv`/`RaceCoreEnv` functional data, reset, and step logic directly.
- Keep policy inference, sampling, action scaling, simulation, observation construction, reward,
  episode state, terminal bootstrap, and mask reset inside a compiled `jax.lax.scan`.
- Keep reverse GAE and PPO epoch/minibatch loops compiled with `jax.lax.scan`.
- Do not add Gymnasium wrapper chains, NumPy, SciPy, Torch, per-environment Python loops, callbacks,
  printing, or host synchronization to the rollout/update hot path.
- Only transfer aggregated metrics to the host at logging/validation boundaries.
- Keep shape-changing values (`num_envs`, `num_steps`, minibatches, history length, model shape)
  static during a compiled run.
- Rollout and optimizer are intentionally separate XLA executables. A second outer `jit` was
  measured to increase memory use and reduce RTX 4070 throughput.
- Preserve bounded `tanh` actions and the matching Jacobian-corrected PPO log probability.
- Preserve the distinction between `terminated` and time-limit `truncated`: true termination does
  not bootstrap, truncation bootstraps from the terminal observation, and GAE never propagates
  across either boundary.
- Masked reset must reset environment data, observation history, previous action, reward history,
  and episode accumulators only for completed worlds.

Crazyflow helpers beginning with `_` are currently used to reproduce the exact functional step
order. Re-audit those calls when upgrading Crazyflow; do not assume private API compatibility past
0.4.x.

## Current policy and action semantics

The actor and critic are independent two-layer, 256-unit Tanh MLPs with orthogonal initialization.
The actor has a learnable four-dimensional log standard deviation. Its output is a pre-Tanh mean;
training samples a diagonal Gaussian and applies Tanh, while evaluation/controller execution uses
`tanh(mean)` deterministically.

Normalized actions are mapped to physical attitude actions in this order:

1. roll
2. pitch
3. yaw
4. collective thrust

The initial thrust bias represents nominal hover thrust. Training, evaluation, and the controller
must use the same action bounds, bias, scaling, observation version, history semantics, and model
shape.

## Current 88-dimensional observation (V1)

The exact layout is:

| Slice | Size | Meaning |
| --- | ---: | --- |
| `0:1` | 1 | world-frame altitude |
| `1:4` | 3 | body-frame linear velocity |
| `4:7` | 3 | angular velocity |
| `7:16` | 9 | body rotation matrix |
| `16:28` | 12 | current gate's four corners in the body frame |
| `28:40` | 12 | next gate's four corners in the body frame |
| `40:44` | 4 | previous normalized action |
| `44:56` | 12 | four nearest obstacle relative positions in the body frame |
| `56:88` | 32 | two 16-value history entries: altitude, rotation, velocity, angular velocity |

Gate order is taken from `gate_sequence`; reverse gates are encoded by mirroring ordered local
corner coordinates using `gate_sequence_direction`. Obstacles are sorted by Euclidean distance.
For cylindrical obstacles, the relative vertical coordinate is clipped so avoidance is mostly an
XY problem.

Important V1 limitations:

- The environment reports nominal object poses until objects enter sensor range, then persistent
  true poses. V1 discards `gates_visited` and `obstacles_visited`, so the policy cannot explicitly
  distinguish nominal from observed geometry.
- Missing obstacle slots use a value of 10 while most orientation inputs lie in [-1, 1]. Inputs are
  not explicitly normalized by semantic scale.
- A flat MLP must infer gate frame, signed plane distance, crossing time, crossing point, and frame
  clearance from four corners.
- Two history samples are weak memory for handling a sudden nominal-to-measured pose update.

## Current reward (V1)

Default `fast` coefficients are:

| Component | Coefficient/behavior |
| --- | --- |
| progress | 60 times clipped decrease in Euclidean distance to current gate center |
| gate | 20 times the one-based gate progress on a pass |
| finish | +100 |
| align | 0.5 times velocity along the directed gate normal |
| collision | -20 once at termination |
| ground | -0.1 per low-altitude step |
| time, smooth, spin, angle | disabled in `fast` |

`safe_fast` currently differs only by enabling `angle=1.0` and using different evaluation targets.

V1's main reward problem is scale and geometry. Successful episodes commonly accumulate roughly
220 progress, 146-150 gate, 49 alignment, and 67-70 average finish contribution in rollout
aggregates, while a collision costs only 20. Euclidean center progress and unconditional normal
velocity can reward a fast approach even when the predicted crossing point is on the frame.

## Completed baseline and artifacts

The full `fast` run used 1024 environments, 96 rollout steps, 8 minibatches, 12 PPO epochs, learning
rate `1e-4`, and approximately 500M environment steps.

- Final checkpoint: `saves/jax_ppo/fast/latest` at 499,974,144 steps.
- Selection checkpoint: `saves/jax_ppo/fast/best` at 490,045,440 steps.
- Final fixed-seed 500-scenario result for `latest`: 331/500, success 66.2%, Wilson 95% CI
  [61.94%, 70.21%], successful mean time 3.499 s, collision 33.8%, truncation 0%.
- The best 100-scenario validation observed 76% at 490,045,440 steps, but this is a selection metric
  and is not evidence of 76% over the final 500-scenario protocol.
- Final rollout success was about 67.2%; rollout metrics are training diagnostics only.
- Validation fluctuated substantially late in training (for example 45% at about 480M and 76% at
  about 490M), indicating update instability and/or a noisy selection set.

Performance reports under `saves/jax_ppo/performance/` record the 10M-step A/B on the RTX 4070:

- adapted PyTorch reference: 239,874.94 end-to-end SPS
- fused JAX PPO: 519,002.40 end-to-end SPS
- speedup: 2.16364x (gate passed)
- JAX rollout: 1,075,851 SPS
- JAX optimizer: 992,915 sample SPS
- median GPU utilization: 99%
- peak GPU memory: 11,109 MiB
- XLA compilation: 4.854 s rollout and 0.337 s optimizer, excluded from steady throughput

Do not commit anything in `saves/`; it is intentionally ignored. Do not delete these artifacts
without explicit permission because they are the only local trained policies and reports.

## Level 3 V2 implemented design and current gate

Level 3 V2 is implemented behind separate modules/profile/checkpoints; it does not alter V1 policy
files or the configured Level 2 controller. Key invariants:

- Actor input is deployment-safe structured belief: 28 ego values, four 16-value history frames,
  one 23-value gate token per gate pass, four 15-value obstacle tokens, and a 31-value direct geometry skip.
- Actor and critic parameters are independent. The critic re-encodes Actor belief and adds actual
  geometry plus eight dynamics/disturbance/difficulty values.
- Shared per-entity encoders feed parameter-free four-head ego-query attention pooling. The direct
  geometry path is never compressed by pooling.
- Reward uses directed route potential x30, safe-only gate speed x0.3, gate +10, finish +100,
  gate/obstacle/boundary risk coefficients 0.15/0.10/0.05, collision `-90 - 30*n_passed`, and
  timeout `-60 - 30*n_passed`. Reliability phase has no time or action-delta reward.
- Curriculum samples per world but always uses a fixed 500M clock; a short run's stop budget must
  not compress it. Track layout randomization remains fully enabled at every difficulty.
- PPO uses 1024x128, eight minibatches, four epochs, LR `5e-5 -> 1e-5`, target KL 0.01, and EMA
  0.99. Raw and EMA validation/checkpoint selection are separate. Full training never early-stops.
- Final evaluation is deterministic, original full Level 3, 5,000 independently seeded episodes.

The final 10M same-architecture benchmark measured 84,114.36 PyTorch SPS and 184,234.72 JAX SPS
(2.19029x); JAX rollout was 449,674.70 SPS and the performance gate passed. Reports are ignored in
`saves/jax_ppo/performance/level3_v2_final_{torch,jax}_10m.json`.

`Linger=yes` is enabled with user authorization. A real transient service was stopped by SIGTERM
at 6,291,456 steps, saved `latest` after the current PPO update, and restored exactly at update 48.
All later 50M pilots ran as detached user services and exited successfully.

The selected reward4 50M pilot is `saves/jax_ppo/level3_v2_reward4_pilot50m/latest`. Its final
fixed 500-scenario validation had 0 full successes; raw/EMA collision was 59.4%/57.6%, raw/EMA
passed gate one in 333/360 scenarios, and 29/25 reached gate three. This passes the required
gate-completion/learning-direction pilot gate, but is not evidence of a successful Level 3 policy.
The recommended full run is policy-only initialization from its EMA, with optimizer/environment/
PRNG/curriculum reset. This recommendation is superseded by the mixed-start experiment below.

`level3_v2_segment` changes only the training reset distribution. Each reset independently samples
the original start or a state just after gate one, two, or three with probability 0.25. The ordinary
RaceCore reset still generates the entire random track, uncertainty, dynamics and curriculum; the
pure-JAX transform then initializes directed exit pose/velocity, elapsed time, visited flags,
reward potential/history and runner start progress. Training metrics are split by start progress.
Validation/evaluation always force original full starts. Those historical format-3 checkpoints
remain policy-only compatible after start-kind telemetry was added; format 4 (Crazyflow 0.3, 4-pass
route) is policy-only too, and current format 5 exactly restores the runner and records
`gate_sequence`. Selection is lexicographic by successes, total gates passed, fewer collisions,
and mean return.

The segment experiment must start from random parameters. Its 382-update pilot is 50,069,504 steps
of the unchanged 500M LR and curriculum clocks. CPU and 1024-world GPU rollout/update tests pass.
The segment 10M benchmark measured 181,639.64 steady SPS and 2.15944x the 84,114.36 SPS PyTorch
reference, passing both the 171k and 2x gates; its ignored report is
`saves/jax_ppo/performance/level3_v2_segment_jax_10m.json`. After the pilot, require all three
segment target-gate pass rates >=50%, start-2 remaining-track success >=5%, and start-3
remaining-track success >=20%, with no numerical instability. Continuing from the pilot checkpoint
to 500M requires explicit user confirmation and must not early-stop.

The 50M segment pilot failed: original-start and all segment pass rates ended at zero. Diagnostic
evaluation found 98.2% current-gate visibility but zero passes, 0.014 m mean minimum gate-plane
distance, and 0.709 m near-plane lateral error. The root cause is reward geometry, not insufficient
sensor visibility: V2 allowed progress near the plane outside the opening and disabled gate risk
when crossing prediction became invalid. `level3_v3_reward` preserves
Obs/network/PPO/reset/evaluation, reduces route progress to 12, raises gate reward to 30, gates
route phase transitions on current aperture alignment, and adds current-clearance near-plane/stall
penalties. Its 10M performance benchmark passed at 181,357.14 SPS and 2.15608x. The 77-update
pilot completed 10,092,544 steps stably at about 185k SPS, but target-gate pass rates peaked at only
about 0-1.5% and full-start raw/EMA evaluation remained 0/500. Reward V3 removed the known reward
exploit but did not teach the crossing primitive from post-gate starts.

`level3_v3_pregate` is the next diagnostic profile. It keeps Reward V3 and samples the target gate
uniformly, but initializes every training episode 0.45-0.65 m on the directed entrance side with
small 0.03 m lateral/vertical offsets and 0.20-0.45 m/s gate-directed velocity. It is deliberately
a micro-curriculum for determining whether PPO can learn the crossing primitive, not a production
training distribution. Evaluation remains full-start. Run only a 77-update 10M pilot from random
parameters before designing a mixed full/post/pre-gate curriculum. Its RTX 4070 benchmark passed
at 182,103.90 end-to-end SPS, 2.16496x the same-environment PyTorch baseline, with 9,511 MiB peak
GPU memory. Its 10M pilot nevertheless failed: target-gate pass rates peaked at only 0.9-3.3% and
ended at 0-2.0%. A reachability probe found that the initialized deterministic/stochastic policies
passed only 3/1102 and 0/1813 completed episodes respectively, so this reset was still too far from
the gate to seed the crossing behavior.

`level3_v3_crossing_seed` is the corrected diagnostic profile. It places training starts only
0.12-0.18 m before the directed gate plane, uses +/-0.015 m offsets, and seeds 0.60-0.90 m/s forward
speed. Before learning, the same deterministic initialization passed 89.8-91.6% per target gate
and stochastic sampling passed 77.4-81.5%, proving that positive crossing samples are reachable.
Use a separate output directory and run only a 77-update 10M pilot; do not overwrite the failed
`level3_v3_pregate` artifacts or expand to the full budget based on injected-start pass rates alone.
The pilot completed 10,092,544 steps stably at 184,089.88 final SPS. Final target-gate pass rates
were 100%, 98.39%, 100%, and 98.55% for starts 0-3. Start-2 remaining-track completion peaked at
3.33% but ended at zero; start-3 completion ended at 98.55%. Full-start raw/EMA evaluation remained
0/500 with zero gates passed. Therefore the crossing primitive is now established, but approach,
exit recovery, and transitions between gates are not. Do not continue this injected-only profile
to 500M. The next experiment must mix full, post-gate, and progressively widened pre-gate starts.

`level3_v3_mixed_curriculum` is that next experiment and is the only segment profile allowed to
policy-initialize. It loads the raw crossing-seed policy while resetting optimizer, EMA, PRNG,
environment and the 500M curriculum clock. Full/post/pre probabilities move through
10/20/70% at progress 0, 20/25/55% at 10%, 45/30/25% at 40%, and 70/15/15% at 80% and later.
Pre-gate geometry expands by 40% progress from 0.12-0.18 m, +/-0.015 m and 0.60-0.90 m/s to
0.60-1.00 m, +/-0.12 m and 0.15-0.50 m/s. Post-gate geometry stays unchanged. Runner/checkpoint
formats 4+ record start kind separately from start progress, and rollout metrics report full,
post-gate and pre-gate outcomes independently. Formal evaluation remains original full-start.
The RTX 4070 benchmark passed at 180,928.00 end-to-end SPS and 2.15098x the same-environment
PyTorch baseline, with 9,511 MiB peak memory. Start with a 382-update 50M pilot; do not launch the
remaining 450M until full-start and multi-segment diagnostics are reviewed.

The mixed 50M pilot completed 50,069,504 steps stably at 188,939.33 final SPS. At the final
rollout, pre-gate target passes remained 79.80% and directly seeded gate-4 completion was 65.63%,
but full-start and post-gate target passes were both zero. Full-start raw/EMA evaluation remained
0/500 with zero gates passed and 99.0%/99.4% truncation. Do not continue this checkpoint to 500M:
the policy retains the crossing primitive but cannot turn from a previous-gate exit or acquire the
first gate from the race start. The next bounded experiment should sample a continuous approach
corridor from the original/previous-gate exit to the next-gate entrance, including increasing
heading and lateral errors, rather than adding more endpoint-only full/post/pre training.

`level3_v3_route_corridor` implements that bounded experiment without changing Obs V2, Reward V3,
PPO, random track generation, collision termination, or formal evaluation. At each transformed
reset it samples target gates 1-4 uniformly and places the vehicle continuously between the
original start/previous directed gate exit and the next directed gate entrance. During the first
10% of the unchanged 500M clock, corridor coverage expands from the final 35% to the full route;
entrance distance, lateral/vertical error, velocity noise, attitude error, and angular velocity
expand at the same time. Full/corridor/crossing-seed probabilities are 10/60/30% initially,
20/65/15% at 10%, 45/50/5% at 40%, and 70/30/0% from 80%. Start kind 3 identifies corridor
episodes in checkpoints and metrics. Evaluation still forces original starts. Its RTX 4070 10M
benchmark passed at 180,497.43 SPS, 2.14888x the 83,996.06 SPS stored PyTorch reference, with
438,513.02 rollout SPS, 9,511 MiB peak memory, and 11.395/1.367 seconds rollout/optimizer compile
time. The ignored report is
`saves/jax_ppo/performance/level3_v3_route_corridor_jax_10m.json`.

`level3_v22_yaw_lock` is `level3_v20_memory` plus `lock_yaw=True` and
`yaw_anneal_timesteps=120_000_000`: every Level 3 transition multiplies the normalized yaw by
`clip(1 - steps / 120M, 0, 1)` before scaling and stores the scaled action in observation memory and
reward state; yaw stays in PPO log-probabilities and entropy only while that scale is positive.
Validation/evaluation run at progress 1 and are fully locked (0 rad world yaw), like the controller,
which honors `lock_yaw` from checkpoint metadata. See `LEVEL3_V2.md` for semantics.

## Historical Level 2 V2 research notes

The following list predates the selected Level 3 design. Do not implement it over the authoritative
Level 3 contract, but retain it as context for the earlier Level 2 analysis.

Treat the following as the recommended starting hypothesis, not as already-proven design:

1. First add diagnostics that identify the failed gate/progress index and whether contact was with
   a gate, obstacle, or ground. Evaluate the current policy on a large fixed scenario set and use
   this failure breakdown to drive changes.
2. Add current/next gate visited flags and obstacle visited flags. Encode an unobserved relative
   vector as zero plus an explicit validity/visited mask instead of a magic value of 10.
3. Add explicit directed current-gate-frame features: signed plane distance, lateral/vertical
   offset, relative velocity, gate normal, predicted time to plane, predicted crossing point, and
   clearance from the opening boundary.
4. Add a compact next-gate representation for exit planning and an explicit race-progress feature.
5. Represent obstacle risk using relative position/velocity, radius or clearance, and time to
   closest approach. Prefer ordering by collision risk rather than distance. A shared obstacle
   encoder plus symmetric pooling/attention is preferable if it does not materially hurt speed.
6. Normalize every feature using fixed physical scales and clip only after choosing meaningful
   ranges. Document those scales and use them identically in training and deployment.
7. Compare a longer fixed history against a small recurrent policy. Do not add recurrence until
   exact reset behavior and checkpoint/controller state are tested.
8. Replace Euclidean center progress with a potential along an approach-point -> gate-plane -> exit
   route or another directed gate-frame potential.
9. Add a dense pre-collision penalty based on predicted frame clearance and obstacle time to
   collision. Gate-normal speed should only be rewarded when the predicted crossing point is safe.
10. Rebalance terminal rewards so every collided trajectory is clearly worse than a completed
    trajectory. A reasonable initial experiment is gate +10, finish +50, collision -60, with all
    dense terms scaled to keep typical total returns around 100; tune from measured component
    distributions rather than these numbers alone.
11. Add a small time penalty only after reliability improves. Add action jerk/spin penalties only
    if logs show unsafe oscillation; large smoothness penalties can prevent necessary racing
    maneuvers.
12. Stabilize PPO with a decaying learning rate, 4-6 update epochs, target KL around 0.01, and
    optional parameter EMA. Ablate each change rather than changing everything in one run.
13. Use a real environment curriculum: reduced pose randomization first, then full Level 2 pose
    randomization, then full disturbances/obstacles. Curriculum changes must stay on device or be
    staged through compatible checkpoints without introducing dynamic shapes.

An observation or network shape change is intentionally incompatible with V1 policy files. Create
a versioned V2 profile/output directory (for example `saves/jax_ppo/v2_fast/`) and do not load V1
parameters unless a deliberately tested compatible encoder preserves all parameter shapes. Keep
the V1 controller usable while developing V2.

## Experimental protocol for a 90% claim

- Establish one frozen set of training changes before a long run.
- Run unit/CPU/GPU integration tests and a throughput benchmark after changing hot-path shapes.
- Run a short pilot that proves learning direction and checks numerical stability before a long
  budget.
- Select checkpoints on a fixed validation suite large enough to reduce noise; 100 episodes is too
  noisy for fine ranking near 90%.
- Run the final deterministic evaluation on at least 2,000 scenarios, preferably 5,000, using a
  separately fixed seed range. Report successes, Wilson 95% interval, successful mean/median/p95
  time, collision rate, truncation rate, and failures by gate/contact type.
- A point estimate above 90% is not enough for a strong claim. Prefer a Wilson lower bound above
  90%, or clearly state when only the point estimate exceeds 90%.
- Evaluate `latest`, selected checkpoints, and EMA separately. Never overwrite or relabel the
  result of one checkpoint as another.
- Preserve the Level 2 configuration and deterministic policy mode during comparison.

## Commands

Training and exact resume:

```bash
pixi run -e gpu train-jax-ppo --profile fast
pixi run -e gpu train-jax-ppo --profile fast --disable-early-stopping
pixi run -e gpu train-jax-ppo \
  --profile fast --resume saves/jax_ppo/fast/latest --disable-early-stopping
```

Deterministic policy evaluation:

```bash
pixi run -e gpu eval-jax-ppo \
  --checkpoint saves/jax_ppo/fast/latest --profile fast --episodes 500 --seed 30000
```

Controller simulation using the configured Level 2 policy:

```bash
pixi run -e gpu python scripts/sim.py --config level2.toml
```

The script defaults to `level0.toml`, so always pass `--config level2.toml`. Do not override it with
`--controller state_controller.py`. A green reference curve/red setpoint means the old state
controller is running; the JAX controller does not draw a trajectory. The first rendered steps may
show timing warnings due to one-time JAX compilation.

CPU unit and integration tests:

```bash
pixi run -e tests python -m pytest -q \
  tests/unit/rl_training/test_jax_ppo.py \
  tests/integration/rl_training/test_jax_ppo_environment.py -m 'not gpu'
```

GPU 1024-world rollout/update test:

```bash
pixi run -e gpu-tests python -m pytest -q \
  tests/integration/rl_training/test_jax_ppo_environment.py::test_gpu_1024_world_rollout_and_full_update
```

Formatting and static checks:

```bash
pixi run ruff check lsy_drone_racing/rl_training tests/unit/rl_training \
  tests/integration/rl_training
pixi run ruff format --check lsy_drone_racing/rl_training tests/unit/rl_training \
  tests/integration/rl_training
git diff --check
```

Performance A/B:

```bash
pixi run -e rl-benchmark-gpu benchmark-torch-ppo \
  --timesteps 10000000 --output saves/jax_ppo/performance/torch_10m.json
pixi run -e gpu benchmark-jax-ppo \
  --timesteps 10000000 \
  --pytorch-report saves/jax_ppo/performance/torch_10m.json \
  --output saves/jax_ppo/performance/jax_10m.json
```

Level 3 V2 training, evaluation, and same-architecture A/B:

```bash
pixi run -e gpu train-level3-jax-ppo --profile level3_v2
pixi run -e gpu eval-level3-jax-ppo \
  --checkpoint saves/jax_ppo/level3_v2/best_ema --episodes 5000 --seed 30000
pixi run -e rl-benchmark-gpu benchmark-level3-torch-ppo \
  --timesteps 10000000 --output saves/jax_ppo/performance/level3_v2_final_torch_10m.json
pixi run -e gpu benchmark-level3-jax-ppo \
  --timesteps 10000000 \
  --pytorch-report saves/jax_ppo/performance/level3_v2_final_torch_10m.json \
  --output saves/jax_ppo/performance/level3_v2_final_jax_10m.json

pixi run -e gpu benchmark-level3-jax-ppo \
  --profile level3_v2_segment --timesteps 10000000 \
  --pytorch-report saves/jax_ppo/performance/level3_v2_final_torch_10m.json \
  --output saves/jax_ppo/performance/level3_v2_segment_jax_10m.json
pixi run -e gpu train-level3-jax-ppo \
  --profile level3_v2_segment --max-updates 382 --skip-final-evaluation
pixi run -e gpu train-level3-jax-ppo \
  --profile level3_v2_segment \
  --resume saves/jax_ppo/level3_v2_segment/latest
pixi run -e gpu eval-level3-jax-ppo \
  --profile level3_v2_segment \
  --checkpoint saves/jax_ppo/level3_v2_segment/best_ema \
  --episodes 5000 --seed 30000

# Continuous approach-corridor 50M pilot; initialize raw policy only.
pixi run -e gpu train-level3-jax-ppo \
  --profile level3_v3_route_corridor \
  --initialize-policy saves/jax_ppo/level3_v3_mixed_curriculum/latest \
  --max-updates 382 --skip-final-evaluation
# Yaw-annealed fine-tune of the 5-pass v20 policy.
pixi run -e gpu train-level3-jax-ppo --profile level3_v22_yaw_lock \
  --initialize-policy saves/jax_ppo/level3_v20_memory_5pass_r2/latest --initialize-policy-ema \
  --total-timesteps 400000000 --output saves/jax_ppo/level3_v22_yaw_lock
```

Compilation time must be reported separately from steady throughput. If a V2 change fails the 2x
same-environment PyTorch gate, diagnose synchronization, recompilation, launch count, layout, and
memory pressure before beginning a long run.

## Detached long-run procedure

Before launching, verify that the user manager survives logout with
`loginctl show-user ymlee -p Linger`. If lingering is disabled, do not assume the run is SSH-safe;
ask the user before enabling it. Prefer a named transient user service over `nohup`, `tmux`, or a
process owned by the Codex terminal. A representative launch is:

```bash
systemd-run --user \
  --unit=lsy-jax-ppo-level3-segment-pilot \
  --description='LSY Level 3 mixed-segment JAX PPO 50M pilot' \
  --property=WorkingDirectory=/media/ymlee/yuming_data/repos/lsy_drone_racing \
  --collect \
  /home/ymlee/.pixi/bin/pixi run -e gpu train-level3-jax-ppo \
  --profile level3_v2_segment --max-updates 382 --skip-final-evaluation
```

Before reporting the job as detached, verify all of the following:

```bash
systemctl --user status lsy-jax-ppo-level3-segment-pilot
journalctl --user -u lsy-jax-ppo-level3-segment-pilot -n 100 --no-pager
nvidia-smi
```

Use `journalctl --user -u lsy-jax-ppo-level3-segment-pilot -f` to follow logs and
`systemctl --user stop lsy-jax-ppo-level3-segment-pilot` for a graceful stop. The Level 3 CLI
catches SIGINT and SIGTERM, requests shutdown after the current PPO update, and saves `latest` in
`finally`, but a real transient-service stop/save/resume test passed on 2026-08-12 at 6,291,456
steps. Exact restore continued from update 48. Keep this behavior covered when changing the
lifecycle.

## Required tests for V2 changes

In addition to existing regression tests, add tests for:

- exact observation layout/version and fixed normalization ranges
- nominal versus visited gate/obstacle encoding
- signed/reverse gate-frame geometry
- predicted crossing point and opening clearance
- obstacle ordering/pooling and zero-obstacle padding/masks
- history/recurrent state reset for only completed worlds
- reward component signs, magnitudes, and potential telescoping
- no reward for unsafe normal velocity
- collision, success, true termination, and time-limit bootstrap boundaries
- training/evaluation/controller feature parity
- old V1 and new V2 checkpoint rejection with a clear error
- checkpoint exact resume, optimizer state, PRNG, and recurrent state if used
- finite rollout/update values and parameter changes
- 1024-world GPU rollout, masked reset, full PPO update, and no production Torch import
- mixed-start sampling ratio, positive/reverse exit-side geometry, candidate safety, state ranges,
  visited/nominal semantics, elapsed time, and per-start reward/history initialization
- segment masked reset and diagnostics, full-start equivalence, evaluation full-start enforcement,
  format-2/3/4 policy-only compatibility/format-5 exact resume, and lexicographic checkpoint selection
- continuous corridor sampling from original/previous-exit sources, directed/reverse target
  entrance geometry, curriculum coverage expansion, start-kind diagnostics, and full-start fallback

## Git and change discipline

- Inspect `git status` before editing. Preserve unrelated user changes.
- Use focused commits; do not amend or rewrite existing commits unless requested.
- Do not modify `main` for this research. Stay on `rl-jax-dev` or create a new explicitly requested
  branch from it.
- Do not run destructive Git commands or delete ignored artifacts.
- Keep V1 behavior and controller available while introducing V2 behind explicit profiles/files.
- Update this document whenever architecture, checkpoint compatibility, commands, measured
  baselines, or long-running job procedures materially change.
