# Level 3 V2 structured JAX PPO

This document freezes the first reliability-first Level 3 experiment. It is an implementation
contract for training, deterministic evaluation, and deployment. The original `config/level3.toml`
track generator, randomization limits, disturbances, sensing rules, episode termination, and
success definition remain unchanged.

## Objective and evidence standard

- Train on fully randomized Level 3 tracks using direct attitude actions.
- Reliability is the only optimization target in this phase. Do not add a time reward or tune for
  lap time until the user explicitly approves it.
- Collision terminates immediately.
- Full-budget training has no early stopping.
- The headline result uses deterministic Actor execution on 5,000 independently seeded, original
  Level 3 episodes. Report the point estimate, Wilson 95% interval, successful mean/median/p95
  time, collision/truncation rates, collision categories, and failure gate.
- A point estimate of at least 90% is the requested target. State separately whether the Wilson
  lower bound also exceeds 90%.

## Actor observation contract

All features are `float32`, normalized with fixed physical scales, and clipped only where the
feature table specifies. The Actor receives only information available to the deployed controller:
nominal geometry until an object is sensed, then its persistent measured geometry. An object pose
never silently tells the Actor whether it is nominal or measured; explicit visited flags do that.

| Tensor | Shape | Contents |
| --- | --- | --- |
| `ego` | `[B, 28]` | altitude `/2.5`, body velocity `/4`, angular velocity `/10`, 3x3 body rotation, normalized distance to six arena faces, race progress, remaining episode fraction, previous normalized action |
| `history` | `[B, 4, 16]` | body velocity `/4`, angular velocity `/10`, previous action, body-velocity delta `/2`, angular-velocity delta `/5` |
| `gate_tokens` | `[B, 4, 23]` | sequence-ordered relative centre `/3`, directed normal/horizontal/vertical axes, measured-minus-nominal position `/0.20`, rotation correction `/0.25`, visited/passed/current/future flags, relative sequence offset |
| `obstacle_tokens` | `[B, 4, 15]` | relative position `/3`, measured correction `/0.20`, relative velocity `/4`, visited, route-corridor distance, time to closest approach, closest clearance, collision risk, valid mask |
| `direct_geometry` | `[B, 31]` | current directed gate-frame position/velocity, time and predicted point at the plane, opening clearances, prediction/correct-side flags, approach/exit vectors, next-turn sin/cos, current-gate visited, highest-risk obstacle geometry, nearest arena-boundary clearance |

Obstacle slots are padded to four with zero features and a false valid mask. A magic sentinel is
never used. Gate tokens follow `gate_sequence`; the directed normal and horizontal axes are
mirrored for reverse gates. All temporal and nominal-map state resets only for completed worlds.

## Critic contract

The critic has parameters independent from the Actor. It re-encodes the complete Actor belief and
adds privileged features that are available only during simulation training:

- actual four-gate tokens, four-obstacle tokens, and 31-value direct geometry;
- actual-versus-nominal corrections contained in those tokens;
- mass ratio, three diagonal inertia ratios, current external force, and curriculum difficulty
  (`[B, 8]`).

It never receives future RNG values, future disturbances, success labels, or collision labels.
Evaluation and deployment call only `model.actor`; privileged features are not required.

## Network

Actor and critic belief encoders each have independent two-layer Tanh encoders for ego, history,
gates, and obstacles. Entity width is 128. Gate and obstacle embeddings are pooled with
parameter-free four-head ego-query scaled dot-product attention. The pooling is permutation
invariant and avoids redundant Q/K/V/O projections; the 31 critical geometry values bypass it
unchanged. Actor and critic fusion heads are two independent 256-unit Tanh layers. The Actor emits
a four-value pre-Tanh mean plus learned log standard deviation. Training uses a
Jacobian-corrected tanh-squashed diagonal Gaussian; deterministic execution uses `tanh(mean)`.

Normalized actions map identically in training and deployment to:

1. roll
2. pitch
3. yaw
4. collective thrust

The initial thrust bias is physical hover thrust.

## Safety geometry

- Crazyflie collision sphere radius: 0.086 m.
- Gate collision inner boundary: 0.20 m from centre.
- Target centre half-clearance: 0.10 m horizontally and vertically (0.014 m robustness reserve).
- Gate risk buffer: 0.04 m.
- Obstacle physical radius: 0.015 m; target centre safety radius: 0.13 m; risk buffer: 0.12 m.
- Arena boundary risk buffer: 0.20 m.
- Route approach and exit offsets: 0.50 m and 0.35 m.
- Gate and obstacle prediction horizons: 1.0 s.

## Reward

The dense progress term is the clipped decrease in a smooth directed route potential through
`approach -> gate plane -> exit`, multiplied by 30. A single step is clipped to 0.20 m before the
coefficient. Gate passes add +10 and completion adds +100. Directed gate-normal speed contributes
at most `0.3 * 2.0` per step, and only when the predicted gate-plane intersection has positive
opening clearance.

Gate-frame, obstacle-TCA, and arena-boundary risks are capped at -0.15, -0.10, and -0.05 per step.
The action-delta coefficient remains zero because the pilots did not show oscillation that
justified suppressing necessary manoeuvres. There is no unconditional normal-velocity reward,
global angle penalty, or time penalty. Collision costs `-90 - 30*n_passed`; timeout costs
`-60 - 30*n_passed`, making late failure progressively worse without making hovering preferable
to completion. Gate, finish, collision, and timeout events use actual geometry/state, while the
Actor remains belief-only.

## Mixed-difficulty curriculum

Every reset independently samples an uncertainty multiplier `d`. Full track-layout generation is
always active; `d` scales initial pose, local gate/obstacle pose error, inertial randomization,
action noise, and dynamic force. The curriculum clock is always 500M environment steps, even for a
short pilot, so changing a run's stopping budget cannot accidentally compress the curriculum.

| Progress | Mixture |
| --- | --- |
| 0-15% | 60% `U(0,.3)`, 20% `U(.3,.8)`, 20% `d=1` |
| 15-45% | 25% `U(0,.4)`, 45% `U(.4,.9)`, 30% `d=1` |
| 45-75% | 10% `U(.3,.6)`, 30% `U(.6,1)`, 60% `d=1` |
| 75-100% | all `d=1` |

Validation always uses `d=1` and the original Level 3 environment, independent of curriculum.

## Mixed gate/segment reset profile

`level3_v2_segment` is a reset-distribution experiment. It deliberately reuses the exact V2
observation, reward, Actor/Critic, PPO update, collision termination, curriculum, random Level 3
track generator, and success definition. It must start with random parameters; policy-only
initialization is rejected. Exact resume is supported only from its own format-3 checkpoint.

After the ordinary reset has generated the complete track, uncertainty, dynamics, and clean
controller state, each training world independently samples one of four start-progress values with
probability 0.25: the original start, or just after gate one, two, or three. Segment states are
placed in the real previous-gate exit frame. The randomized exit distance is 0.25-0.35 m with
0.04 m lateral/vertical perturbations; unsafe candidates fall back to centred 0.20 m and 0.12 m
exit candidates. Initial velocity, attitude, angular velocity, passed/visited flags, potential,
reward history, previous action, state derivatives, and simulated elapsed time are initialized
consistently. Segment worlds retain the episode's pre-transform nominal map rather than treating
the deliberately visited passed gates as newly nominal geometry.

Training rollouts report episode count, target-gate pass rate, remaining-track completion,
collision rate, timeout rate, and final progress separately for starts 0-3. Validation and the
evaluation CLI force `full_start`; a segment profile can never turn a shortened episode into a
headline Level 3 success. Best checkpoints use the frozen lexicographic order: success count,
total gates passed, fewer collisions, then mean return.

## PPO and lifecycle

- 1,024 environments x 128 steps, eight minibatches, four epochs.
- gamma 0.995, GAE lambda 0.97, clip 0.2.
- Adam learning rate decays linearly from `5e-5` to `1e-5` over the configured training budget.
- target KL 0.01 stops remaining epochs at epoch boundaries.
- gradient norm 0.5; parameter EMA 0.99. Validation records raw and EMA independently and stores
  separate `best_raw` and `best_ema` checkpoints.
- Checkpoints are V2/version-checked. Format 3 exact resume includes raw/EMA parameters,
  optimizer, PRNG, environment, curriculum progress, start progress, belief anchors, and history.
  Existing format-2 V2 checkpoints remain policy-only compatible but are explicitly rejected for
  exact resume; V1 policies are rejected.
- SIGINT and SIGTERM request a save after the current PPO update.

Commands:

```bash
pixi run -e jax-rl-gpu train-level3-jax-ppo --profile level3_v2
pixi run -e jax-rl-gpu train-level3-jax-ppo \
  --profile level3_v2 --resume saves/jax_ppo/level3_v2/latest
pixi run -e jax-rl-gpu train-level3-jax-ppo \
  --profile level3_v2 \
  --initialize-policy saves/jax_ppo/level3_v2_reward4_pilot50m/latest \
  --initialize-policy-ema
pixi run -e jax-rl-gpu eval-level3-jax-ppo \
  --checkpoint saves/jax_ppo/level3_v2/best_ema --episodes 5000 --seed 30000

# First 50,069,504 steps of the unchanged 500M schedule (382 updates).
pixi run -e jax-rl-gpu train-level3-jax-ppo \
  --profile level3_v2_segment --max-updates 382 --skip-final-evaluation

# Run only after the pilot gates pass and the user explicitly authorizes continuation.
pixi run -e jax-rl-gpu train-level3-jax-ppo \
  --profile level3_v2_segment \
  --resume saves/jax_ppo/level3_v2_segment/latest

# Always evaluates original starts at full difficulty.
pixi run -e jax-rl-gpu eval-level3-jax-ppo \
  --profile level3_v2_segment \
  --checkpoint saves/jax_ppo/level3_v2_segment/best_ema \
  --episodes 5000 --seed 30000
```

Long jobs must use a named `systemd --user` transient service and require `Linger=yes`. Record the
unit, main PID, exact command, journal command, output/checkpoint path, and graceful stop command.

## Performance and current research status

The final 10M same-architecture A/B on the RTX 4070, 1,024 environments, 128 steps, eight
minibatches, and four epochs measured:

- eager PyTorch: 84,114.36 end-to-end SPS;
- fused JAX: 184,234.72 end-to-end SPS (2.19029x, gate passed);
- JAX rollout: 449,674.70 SPS;
- JAX optimizer: 281,442.44 samples/s (1,125,769.76 epoch-samples/s);
- median GPU utilization 100%, observed peak memory 9,510 MiB;
- XLA compile: 2.353 s rollout and 3.511 s optimizer, excluded from steady throughput;
- estimated raw 500M training time: 0.754 h plus validation/checkpoint overhead.

Reports are ignored artifacts under
`saves/jax_ppo/performance/level3_v2_final_{torch,jax}_10m.json`.

The mixed-segment 10M benchmark passed: 181,639.64 end-to-end SPS, 441,945.41 rollout SPS,
301,995.37 optimizer sample SPS, 2.15944x over the 84,114.36 SPS PyTorch reference, 100% median GPU
utilization, and 9,513 MiB peak memory. XLA rollout/optimizer compilation took 11.019/1.364 seconds
and is excluded from steady throughput. The ignored report is
`saves/jax_ppo/performance/level3_v2_segment_jax_10m.json`.

The first 50M reward pilot converged to hovering: deterministic EMA was 0/500 and 81.8% truncated.
Increasing directed progress produced a navigation policy whose raw parameters reached gate two
in 372/500 scenarios and gate three in 100/500, but collided in 81.2%. A strong safety ablation
reduced collision to 7-9% but truncated 91-93%, proving that it was over-conservative. The selected
balanced reward4 pilot completed 50,069,504 steps and maintained stable gate progress. At its final
fixed 500-scenario validation, raw/EMA collision was 59.4%/57.6%; 333/360 scenarios respectively
passed gate one, and 29/25 reached gate three. Neither completed all four gates, so the success
point estimate remains 0% and no 90% claim is warranted.

The previous 50M gate-completion/learning-direction gate was passed, but gate three remained the
bottleneck. The mixed-segment experiment supersedes the earlier policy-initialization proposal and
starts from random parameters. Its 382-update pilot is the first part of the same 500M schedule;
it is not a compressed pilot schedule. Implementation and tests do not authorize launching the
pilot, and continuation after the pilot still requires explicit user confirmation.

The mixed-segment 50M pilot completed at 50,069,504 steps but failed its learning gate. All segment
target-gate pass rates ended at zero, and original-start raw/EMA evaluation passed zero gates in
500 scenarios. A follow-up diagnostic EMA evaluation showed that 98.2% of scenarios observed the
current gate, but none passed it. The mean minimum plane distance was 0.014 m while near-plane
lateral error was 0.709 m. This rules out sensor visibility as the primary cause: V2 route progress
could be harvested near the gate plane outside the opening, and prediction-only gate risk vanished
when the policy stopped.

`level3_v3_reward` keeps the same observation, network, PPO, segment reset and formal evaluation.
It changes only reward geometry: route progress is reduced to 12, gate reward is increased to 30,
route phase transitions are gated by current opening alignment, and current-clearance near-plane/
stall penalties remain active even without a valid predicted crossing. It starts from random
parameters. The V3 10M performance check passed at 181,357.14 end-to-end SPS and 2.15608x the
PyTorch reference; rollout was 441,354.55 SPS and peak GPU memory was 9,513 MiB. Its first pilot is
77 updates (10,092,544 steps of the unchanged 500M schedule). Do not continue or add pre-gate
starts until its gate-pass diagnostics are reviewed.
