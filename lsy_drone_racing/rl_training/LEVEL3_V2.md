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
| `gate_tokens` | `[B, P, 23]` | one token per gate pass (`P = len(gate_order)`), sequence-ordered relative centre `/3`, directed normal/horizontal/vertical axes, measured-minus-nominal position `/0.20`, rotation correction `/0.25`, visited/passed/current/future flags, relative sequence offset |
| `obstacle_tokens` | `[B, 4, 15]` | relative position `/3`, measured correction `/0.20`, relative velocity `/4`, visited, route-corridor distance, time to closest approach, closest clearance, collision risk, valid mask |
| `direct_geometry` | `[B, 31]` | current directed gate-frame position/velocity, time and predicted point at the plane, opening clearances, prediction/correct-side flags, approach/exit vectors, next-turn sin/cos, current-gate visited, highest-risk obstacle geometry, nearest arena-boundary clearance |

Obstacle slots are padded to four with zero features and a false valid mask. A magic sentinel is
never used. Gate tokens follow `gate_sequence`; the directed normal and horizontal axes are
mirrored for reverse gates. All temporal and nominal-map state resets only for completed worlds.

## Critic contract

The critic has parameters independent from the Actor. It re-encodes the complete Actor belief and
adds privileged features that are available only during simulation training:

- actual per-pass gate tokens, four-obstacle tokens, and 31-value direct geometry;
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
initialization is rejected. Its historical format-3 checkpoint remains policy-only compatible;
format 4 added start-kind telemetry. Current exact resume uses format 5, which records the trained
`gate_sequence` and Crazyflow 0.4 runner state; formats 2-4 are policy-only and assume the 4-pass
gate order `(0, 1, 2, 3)`.

After the ordinary reset has generated the complete track, uncertainty, dynamics, and clean
controller state, each training world independently samples a start pass uniformly (empty
`start_probabilities`; an explicit tuple must have one entry per pass): the original start, or just
after any pass except the last. Per-start metrics and `before_gate_N` are indexed by pass, not by
physical gate. Segment states are
placed in the real previous-gate exit frame. The randomized exit distance is 0.25-0.35 m with
0.04 m lateral/vertical perturbations; unsafe candidates fall back to centred 0.20 m and 0.12 m
exit candidates. Initial velocity, attitude, angular velocity, passed/visited flags, potential,
reward history, previous action, hover rotor speed, and simulated elapsed time are initialized
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
- Checkpoints are V2/version-checked. Format 5 exact resume includes raw/EMA parameters,
  optimizer, PRNG, environment, curriculum progress, start progress/kind, belief anchors, and
  history. Existing format-2/3/4 V2 checkpoints remain policy-only compatible but are explicitly
  rejected for exact resume; V1 policies are rejected.
- SIGINT and SIGTERM request a save after the current PPO update.

Commands:

```bash
pixi run -e gpu train-level3-jax-ppo --profile level3_v2
pixi run -e gpu train-level3-jax-ppo \
  --profile level3_v2 --resume saves/jax_ppo/level3_v2/latest
pixi run -e gpu train-level3-jax-ppo \
  --profile level3_v2 \
  --initialize-policy saves/jax_ppo/level3_v2_reward4_pilot50m/latest \
  --initialize-policy-ema
pixi run -e gpu eval-level3-jax-ppo \
  --checkpoint saves/jax_ppo/level3_v2/best_ema --episodes 5000 --seed 30000

# First 50,069,504 steps of the unchanged 500M schedule (382 updates).
pixi run -e gpu train-level3-jax-ppo \
  --profile level3_v2_segment --max-updates 382 --skip-final-evaluation

# Run only after the pilot gates pass and the user explicitly authorizes continuation.
pixi run -e gpu train-level3-jax-ppo \
  --profile level3_v2_segment \
  --resume saves/jax_ppo/level3_v2_segment/latest

# Always evaluates original starts at full difficulty.
pixi run -e gpu eval-level3-jax-ppo \
  --profile level3_v2_segment \
  --checkpoint saves/jax_ppo/level3_v2_segment/best_ema \
  --episodes 5000 --seed 30000

# Mixed full/post/pre 50M pilot initialized from the learned crossing primitive.
pixi run -e gpu train-level3-jax-ppo \
  --profile level3_v3_mixed_curriculum \
  --initialize-policy saves/jax_ppo/level3_v3_crossing_seed/latest \
  --max-updates 382 --skip-final-evaluation

# Continuous approach-corridor 50M pilot initialized from the raw mixed policy.
pixi run -e gpu train-level3-jax-ppo \
  --profile level3_v3_route_corridor \
  --initialize-policy saves/jax_ppo/level3_v3_mixed_curriculum/latest \
  --max-updates 382 --skip-final-evaluation
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
PyTorch reference; rollout was 441,354.55 SPS and peak GPU memory was 9,513 MiB. Its 77-update
pilot completed 10,092,544 steps stably at about 185k SPS, but per-start target-gate pass rates
peaked at only roughly 0-1.5%; final full-start raw/EMA evaluations were both 0/500.

The next bounded A/B is `level3_v3_pregate`: training resets are uniformly assigned to target
gates 1-4 and placed 0.45-0.65 m on the directed entrance side, within +/-0.03 m lateral/vertical
error, with 0.20-0.45 m/s gate-directed velocity. Reverse gates use the mirrored directed frame.
This is a diagnostic crossing micro-curriculum only; formal evaluation continues to force original
starts. Run 77 updates from random parameters and do not extend the experiment unless local
target-gate pass rates materially exceed Reward V3. Its RTX 4070 benchmark passed at 182,103.90
end-to-end SPS and 2.16496x the 84,114.36 SPS same-environment PyTorch reference; peak GPU memory
was 9,511 MiB. The 10M pilot still failed: target-gate pass rates peaked at only 0.9-3.3% and ended
at 0-2.0%. An initialized-policy reachability probe produced only 3/1102 deterministic and 0/1813
stochastic completed-episode passes, so these starts remained too far away to bootstrap crossing.

`level3_v3_crossing_seed` corrects only that diagnostic distribution: 0.12-0.18 m entrance
distance, +/-0.015 m lateral/vertical offset, and 0.60-0.90 m/s forward speed. A no-learning probe
passed 89.8-91.6% of deterministic and 77.4-81.5% of stochastic completed episodes across all
four target gates. This proves reachability but is not itself learned performance. Keep its output
separate, run only 77 updates, and judge the pilot by whether pass rates remain high while later
segment completion emerges; formal full-start evaluation remains the headline protocol.

The 77-update pilot completed 10,092,544 steps stably at 184,089.88 final SPS. Final target-gate
pass rates were 100%, 98.39%, 100%, and 98.55% for starts 0-3. Start-2 remaining-track completion
briefly reached 3.33% and ended at zero, while the directly seeded gate-4 completion ended at
98.55%. Full-start raw and EMA evaluations remained 0/500 with zero gates passed. This validates
the crossing primitive but not approach or multi-segment racing. Do not extend the injected-only
profile to 500M; use its checkpoint only as a policy initialization candidate for a separately
versioned mixed and progressively widened reset curriculum.

`level3_v3_mixed_curriculum` policy-initializes from the raw crossing-seed checkpoint and resets
optimizer, EMA, PRNG, environment, and the 500M clock. Full/post/pre probabilities are
10/20/70% at progress 0, 20/25/55% at 10%, 45/30/25% at 40%, and 70/15/15% from 80% onward.
By 40% progress, pre-gate states expand from 0.12-0.18 m, +/-0.015 m, and 0.60-0.90 m/s to
0.60-1.00 m, +/-0.12 m, and 0.15-0.50 m/s. Post-gate states retain their validated geometry.
Start kind is tracked independently from gate progress, and evaluation always forces full-start.
Its RTX 4070 benchmark passed at 180,928.00 SPS, 2.15098x the 84,114.36 SPS PyTorch baseline,
with 9,511 MiB peak memory. The first gate is a 382-update 50M pilot, not a full-budget launch.

That pilot completed 50,069,504 steps stably at 188,939.33 final SPS. Pre-gate target passes ended
at 79.80%, while full-start and post-gate target passes both ended at zero. Formal raw/EMA
full-start validation was 0/500 with zero gates passed and 99.0%/99.4% truncation. The experiment
therefore preserves crossing but fails to bridge endpoint states into navigation. Do not exact-
resume it to 500M. The next curriculum should sample continuously along the approach route from
the original/previous-gate exit to the next-gate entrance and gradually add heading/lateral error,
so the policy learns the missing turn-and-acquire behavior rather than seeing only route endpoints.

`level3_v3_route_corridor` implements that next curriculum while leaving observation, Reward V3,
network, PPO, random Level 3 track generation, collision termination, and formal evaluation
unchanged. Target progress is uniform over the gate passes. A corridor source is the original race
start for the first pass and the previous pass's directed 0.25 m exit otherwise; candidates less
than 0.10 m in front of the target gate fall back to the approach point. The destination is the next
gate's directed entrance. During the first 10% of the unchanged 500M curriculum clock, the sampled
route fraction expands from `U(0.65,1)` to `U(0,1)`, while entrance depth expands from 0.12-0.18 m
to 0.30-0.45 m and position, velocity, yaw, roll/pitch, and angular-velocity errors widen. Unsafe
candidates use centered route/entrance fallbacks; invalid worlds revert to a true full start.

The full/corridor/crossing-seed mixture is 10/60/30% initially, 20/65/15% at 10% progress,
45/50/5% at 40%, and 70/30/0% from 80%. Corridor episodes use start kind 3 and retain the same
per-target and per-kind diagnostics. Policy-only initialization loads the raw mixed checkpoint;
optimizer, EMA, PRNG, environment and curriculum clock reset. Formal evaluation always uses the
original start. CPU regressions and the 1024-world full GPU rollout/update test pass. Its RTX 4070
10M benchmark measured 180,497.43 end-to-end SPS, 438,513.02 rollout SPS, 302,000.15 optimizer
sample SPS, 2.14888x the stored 83,996.06 SPS PyTorch reference, 100% median GPU use, and 9,511 MiB
peak memory. XLA rollout/optimizer compilation was 11.395/1.367 seconds and is excluded from
steady throughput. The report is ignored at
`saves/jax_ppo/performance/level3_v3_route_corridor_jax_10m.json`. Run exactly 382 updates for the
first 50M decision point; do not infer formal Level 3 success from corridor metrics.

## Yaw-locked fine-tuning

`level3_v22_yaw_lock` equals `level3_v20_memory` (25-frame history, V19 reward, same PPO) plus
`lock_yaw=True` and `yaw_anneal_timesteps=120_000_000`. The 5-pass v20 policy saturates its yaw
command at +/-90 deg and swings real yaw about 190 deg per lap, which is unsafe on hardware; hard
locking it immediately dropped rollout success to about 0%, so yaw authority is annealed instead.
The transition multiplies the normalized yaw (index 2) by
`s = clip(1 - steps / yaw_anneal_timesteps, 0, 1)` before `scale_action`, with
`steps = curriculum_progress * curriculum_timesteps` derived from total training steps (exact
resume reproduces it). The scaled action is the one stored as previous action and reward last
action. `yaw_anneal_timesteps=0` is a hard lock (`s=0`); it must not exceed
`curriculum_timesteps`. While `s>0` the yaw dimension is in the behavior and update
log-probabilities and the entropy bonus; once `s=0` it is excluded (Gaussian and tanh Jacobian
terms) through a per-sample weight stored in the rollout batch. Rollouts log `yaw_authority`.
Validation and evaluation force curriculum progress 1, so they are always fully locked and early
validation success is expected to be low. Network shapes are unchanged, so v20 checkpoints
initialize it. The controller reads `lock_yaw` from checkpoint metadata (missing means False) and
commands 0 rad yaw; exact resume treats missing fields as unlocked.

```bash
pixi run -e gpu train-level3-jax-ppo --profile level3_v22_yaw_lock \
  --initialize-policy saves/jax_ppo/level3_v20_memory_5pass_r2/latest --initialize-policy-ema \
  --total-timesteps 400000000 --output saves/jax_ppo/level3_v22_yaw_lock
```
