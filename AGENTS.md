# Agent Guide: Level 2 End-to-End JAX PPO

## Scope and current objective

This file applies to the entire repository. Read it before changing the reinforcement-learning
implementation.

The active development branch is `rl-jax-dev`. It is based on `065a6ec` and contains the fused JAX
PPO implementation added in `3ba81b9`, full-budget training support added in `ae8d2a2`, and the
Level 2 controller configuration added in `6bd683c`.

The next research objective is to improve deterministic Level 2 success from the current 66.2% to
at least 90%, primarily through a versioned observation/reward redesign and more stable PPO
updates. Do not start another long training run until the redesign, tests, diagnostics, and a short
performance/training pilot have passed.

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
- Supported Crazyflow range: `>=0.3.0,<0.4.0`; the current lock resolves Crazyflow 0.3.0.
- Training hardware used so far: one NVIDIA RTX 4070.
- Production training environment: `jax-rl-gpu`.
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
0.3.x.

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

## Observation/reward V2 research direction

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
pixi run -e jax-rl-gpu train-jax-ppo --profile fast
pixi run -e jax-rl-gpu train-jax-ppo --profile fast --disable-early-stopping
pixi run -e jax-rl-gpu train-jax-ppo \
  --profile fast --resume saves/jax_ppo/fast/latest --disable-early-stopping
```

Deterministic policy evaluation:

```bash
pixi run -e jax-rl-gpu eval-jax-ppo \
  --checkpoint saves/jax_ppo/fast/latest --profile fast --episodes 500 --seed 30000
```

Controller simulation using the configured Level 2 policy:

```bash
pixi run -e jax-rl-gpu python scripts/sim.py --config level2.toml
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
pixi run -e jax-rl-gpu benchmark-jax-ppo \
  --timesteps 10000000 \
  --pytorch-report saves/jax_ppo/performance/torch_10m.json \
  --output saves/jax_ppo/performance/jax_10m.json
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
  --unit=lsy-jax-ppo-v2 \
  --description='LSY Level 2 JAX PPO V2 training' \
  --property=WorkingDirectory=/media/ymlee/yuming_data/repos/lsy_drone_racing \
  --collect \
  /home/ymlee/.pixi/bin/pixi run -e jax-rl-gpu train-jax-ppo \
  --profile v2_fast --disable-early-stopping
```

The V2 profile name above is a placeholder until that profile is implemented. Before reporting the
job as detached, verify all of the following:

```bash
systemctl --user status lsy-jax-ppo-v2
journalctl --user -u lsy-jax-ppo-v2 -n 100 --no-pager
nvidia-smi
```

Use `journalctl --user -u lsy-jax-ppo-v2 -f` to follow logs and
`systemctl --user stop lsy-jax-ppo-v2` for a graceful stop. Confirm that SIGTERM/interrupt handling
writes an exact `latest` checkpoint before relying on the stop command; the current training CLI
only explicitly catches `KeyboardInterrupt`, so this must be tested or fixed for a systemd run.

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

## Git and change discipline

- Inspect `git status` before editing. Preserve unrelated user changes.
- Use focused commits; do not amend or rewrite existing commits unless requested.
- Do not modify `main` for this research. Stay on `rl-jax-dev` or create a new explicitly requested
  branch from it.
- Do not run destructive Git commands or delete ignored artifacts.
- Keep V1 behavior and controller available while introducing V2 behind explicit profiles/files.
- Update this document whenever architecture, checkpoint compatibility, commands, measured
  baselines, or long-running job procedures materially change.
