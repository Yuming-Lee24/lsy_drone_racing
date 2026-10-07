# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules for this clone

- **`CLAUDE.md`, `CODING_STYLE_AND_PREFERENCES.md` and `PLAN.md` are tracked only on
  `multi-drone-integration`** (user, 2026-10-07), which lives in the fork and is never merged. They
  stay out of every PR branch and of `upstream`: never copy or cherry-pick them into a branch cut
  from `upstream/main`. Commit changes to them only when the user asks. Stage files by explicit
  path, never with `git add -A`, `git add .` or `git commit -a`.
- **Follow `CODING_STYLE_AND_PREFERENCES.md`.** Read it in full before writing, changing or
  reviewing code, tests, docs or configuration. The summary below is only a reminder of its main
  points.
- Commit messages are one short, precise English line: imperative, sentence case, no body and no
  `Co-authored-by` trailers.
- `origin` is the user's fork (`Yuming-Lee24/lsy_drone_racing`) and `upstream` is
  `learnsyslab/lsy_drone_racing`. Never push to `upstream` and never commit on `main`. Create work
  branches from `upstream/main` with `--no-track`, so that VS Code's "Sync Changes" cannot push them
  to upstream's `main`.
- Push only when the user asks for that specific push, and never open a PR: the user opens them. A
  private `gh` login for pushing may exist on this machine; the memory
  `shared-lab-machine-git-identity` says how to use it. Without it `git push` fails from a Claude
  session: prepare the branch and ask the user to run `git push -u origin <branch>`.
- PR branches are cut in the worktree `~/yuming_ws/lsy_drone_racing_pr`, so this clone and its
  hand-edited `config/*.toml` stay untouched. Never stage or revert those config edits.

## Coding style in brief

The full guide is `CODING_STYLE_AND_PREFERENCES.md`. Its order of precedence is: intended behaviour
and physical safety, then tests and enforced configuration, then maintainer feedback, then existing
code patterns, then the guide itself. The points that most often decide a review:

- Fail fast. Access required config and state directly; no defaults, casts, clamps or `.get` that
  hide a broken contract.
- Use as few `try/except` blocks as possible and never a bare `except`. Release hardware, ROS and
  processes with `try/finally`, and send the safest physical command (emergency stop) first.
- Keep signatures narrow: pass the values a function needs, not a whole `ConfigDict`.
- Extend the existing core, state types, connectors and fixtures instead of adding parallel ones.
  No helper, wrapper, base class or option without a present use.
- Use full, precise names and the repository's vocabulary (`_pos`, `_quat`, `n_drones`, ...).
- No comments that narrate the code, no stale alternatives, no commented-out code.
- A bug fix needs a regression test that fails without the fix. Deployment-sensitive changes need a
  real-hardware check, and the result is recorded.
- Keep diffs focused: no unrelated cleanup, formatter churn, or `pixi.lock` change without a
  dependency change.

## Work in progress on this clone

The branch `multi-drone-integration` rebuilds upstream PR 102 (multi-drone real races with a
host/client split over ROS 2 and Zenoh) on top of `upstream/main`. It is a working baseline that is
never merged; small PRs are cut from it later. The steps are defined in `PLAN.md` in the repository
root. Read it before changing anything on this branch, and do one step at a time as the user asks.

The files ported from the PR were taken unchanged on purpose: `envs/real_race_host_env.py`,
`envs/real_race_client_env.py`, `utils/ros_race_comm.py`, `scripts/multi_deploy_host.py`,
`scripts/multi_deploy_client.py` and `ros_ws/src/drone_racing_msgs/`. The plan decides when they
change; the style guide decides how new or changed code is written. Do not restyle or fix them
outside a step of the plan. The message package has to stay identical to the PR's, because lab
machines already have it built. It is tracked first-party code on this branch, although the style
guide lists all of `ros_ws/` as generated; the rest of `ros_ws/` still is.

## Commands

Everything runs through pixi. The environments (`default`, `tests`, `gpu`, `deploy`, `docs`, ...)
are defined in `pyproject.toml`.

```bash
pixi run -e tests tests -v                    # full suite, as CI runs it (pytest -v tests)
pixi run -e tests pytest tests/unit/envs/test_race_core.py::<test_name> -v   # a single test
pixi run -e tests pytest -m unit tests        # by marker: unit or integration
pixi run ruff check .                         # lint, checked in CI
pixi run ruff format --check --diff .         # formatting, checked in CI
pixi run python scripts/sim.py --config level2.toml --controller <file.py> -n 10 -r
pixi run python scripts/multi_sim.py --config multi_level2.toml
pixi run evaluate                             # competition evaluation: 20 runs of level2
pixi run -e docs docs-build                   # or docs-serve
pixi lock --dry-run --check                   # is pixi.lock still valid? writes nothing
```

`--controller` takes a file name inside `lsy_drone_racing/control/`; `-r` renders and `-n` sets the
number of runs. `scripts/evaluate.py` is the course's evaluation script and must not be altered.

Running a pixi command has side effects:

- Every environment except `docs` runs `tools/setup_acados.sh` on activation, which clones and
  builds acados into `acados/` the first time.
- `deploy` also runs `tools/setup_mocap.sh` (clones `motion_capture_tracking` into `ros_ws/src/` and
  runs `colcon build` when the install is missing) and switches ROS 2 to the Zenoh RMW.
- pixi installs a missing environment and can rewrite `pixi.lock` after a dependency change. CI
  installs with `locked: true`, so check `git status` afterwards. For lint alone, the binary of an
  environment that is already installed works without activation: `.pixi/envs/<env>/bin/ruff`.
- The `deploy` tasks (`mocap`, `estimator`, `zenoh-router`) and the `scripts/*deploy*.py` scripts
  drive lab hardware and real drones. Run them only when the user asks.

## Architecture

**Configuration.** Each scenario is a TOML file in `config/`, loaded by `load_config` into an
`ml_collections.ConfigDict`. `[controller]` names a file in `lsy_drone_racing/control/`, `[sim]`
configures crazyflow, `[env]` holds the Gymnasium id, track, disturbances and randomizations, and
`[deploy]` is read only by the real-world scripts. Multi-drone configs turn the per-drone parts into
lists: `[[controller]]`, `[[deploy.drones]]` and `[[env.kwargs]]` (`freq`, `sensor_range` and
`control_mode` per drone). `level0` to `level3` add randomized inertia, then randomized gates and
obstacles, then random tracks; the online competition runs `level2`.

**Controllers.** A controller is one `Controller` subclass per file. `load_controller` imports the
file by path and asserts there is exactly one. The scripts own the loop: `compute_control`, then
`env.step`, then `step_callback`, with `episode_callback` and `episode_reset` between episodes.

**Simulation environments.** `envs/race_core.py` holds all race logic (gate passing, sensor range,
contacts, termination) for `n_envs × n_drones` on top of a crazyflow `Sim`. State is the `EnvData`
dataclass, a JAX pytree that also carries the simulator data. Reset and step are closures built by
`build_reset_fn` and `build_step_fn` and JIT-compiled, with the configured randomizations and
disturbances compiled in, so those have to be pure functions. `drone_race.py` and
`multi_drone_race.py` are thin Gymnasium adapters, single and vectorized. They return JAX arrays;
the scripts wrap them in `JaxToNumpy`.

**Real environments.** `envs/real_race_env.py` mirrors the simulation interface in NumPy. Drone
state comes from the `ROSConnector` of `drone_estimators`, gate and obstacle poses from Vicon TF
frames (`gate1`, `obstacle1`, ...), and commands go out through `utils/crazyflie.py`, a synchronous
wrapper around the async cflib2 API. `reset` can check the track and the start position against
the config's randomization ranges (`utils/checks.py`).

**Import boundary between sim and real.** `import lsy_drone_racing` registers every environment,
but as entry-point strings, so ROS, cflib2 and `drone_estimators` are only imported when a real
environment is created. The `tests` environment has none of them. Keep ROS-dependent helpers out of
modules that simulation code imports; `utils/ros.py` exists for this.

**Multi-drone real races (host/client).** One host process (`scripts/multi_deploy_host.py`,
`envs/real_race_host_env.py`) checks the track and spawns one `CrazyflieWorker` subprocess per drone,
which owns that drone's radio link. One client process per drone (`scripts/multi_deploy_client.py
--drone_rank <i>`, `envs/real_race_client_env.py`) runs the controller and publishes its actions.
They talk over ROS 2 with the Zenoh RMW: the host publishes `lsy_drone_racing/host_state`, each
client publishes `lsy_drone_racing/client/drone_<rank>/action`, and the
`lsy_drone_racing/calibrate_clock` service aligns client timestamps with the host clock. The message
types come from `ros_ws/src/drone_racing_msgs`, built by colcon when the `deploy` environment
activates. `scripts/multi_deploy.py` with `RealMultiDroneRaceEnv` is the older single-process
variant.

## Easy to get wrong

- State actions are 16-D: position, velocity, acceleration, quaternion, angular velocity. Attitude
  actions are `[roll, pitch, yaw, collective thrust]` with thrust last (see `build_action_space`);
  some docstrings still say thrust first.
- Quaternions are `(x, y, z, w)`.
- Gates and obstacles report their nominal pose until a drone has come within `sensor_range`.
  `gates_visited` and `obstacles_visited` flag which poses are measured.
- `env.track.gate_order` uses signed 1-based gate numbers, negative meaning the reverse direction.
  Observations expose it as 0-based `gate_sequence` plus `gate_sequence_direction`.
- The base `Controller.step_callback` returns `True`, and the run loops treat a true return as
  "controller finished". A controller that should keep flying has to override it.
