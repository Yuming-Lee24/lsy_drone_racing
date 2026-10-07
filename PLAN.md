# Split plan for PR 102 (rev 9, 2026-10-06)

Refs: PR head 7b581ed (upstream refs/pull/102/head), branch point f452f92, main 54e64cf.
Integration branch `multi-drone-integration`: 4731416 (port), fd904a0 (adapt to main), 292589a (controller fix).
Basis: rev 8 came from code reading only. Since then the integration branch was built, tested and flown
(2026-10-06). Line counts are still estimates.

## Review status (2026-10-07, 14:25 UTC) - read this first in a new session
The threads move: read the PRs on GitHub again before acting. Nothing below marked "suggested" has
been decided by the user.

- #143 (PR 0, SciPy fix): merged by amacati as 735437f. He asked to delete the explanatory paragraph
  of the test's docstring ("verbose test comment"); the user did.
  To do: merge upstream/main (now 7cbbaee) into the integration branch.
- #144 ([Multi-drone 1], Zenoh): MERGED by amacati at 14:23 as 7cbbaee "Configure deploy env for
  Zenoh with multicast discovery (#144)": the three variables, no `zenoh-router` task. The thread
  is kept below for the day a router is needed.
  - amacati first believed the PR always starts a router; the user cleared that up.
  - amacati, 11:43: "Let's skip it for now, if we need it, it should be easy to put it back in."
  - amacati, 12:03: the router inherits ZENOH_CONFIG_OVERRIDE from the env, so it would
    listen on `tcp/0.0.0.0:0`, a random port, and clients could not find it; should it not be a
    fixed endpoint?
    He is right according to the rmw_zenoh README ("These overrides apply to Zenoh sessions and the
    Zenoh router"). The task was never run under this env. In September the router was started with
    its own override, `listen/endpoints=["tcp/0.0.0.0:7447"]` (PR 102 comment of 2026-09-10).
    Checked in the installed rmw_zenoh 0.6.4 (12:20 UTC, nothing was run): `rmw_zenohd` reads
    ZENOH_CONFIG_OVERRIDE; the default router config listens on `tcp/[::]:7447`; the default session
    config connects to `tcp/localhost:7447`. Under this env the router therefore leaves 7447 and is
    only found through multicast, which is the case where it is not needed.
  - User, 12:23: "yes you are right. I don't think I should hardcode an address, I would rather
    delete this task, we don't use it now anyways."
  - amacati, 12:31, then closed the PR: "Okay. Then let's not merge this, and we have the PR to go
    back to once we need it."
  - It was a misunderstanding. amacati, 13:12: "No, Zenoh is fine. But we don't merge the router
    config for now". The user, 13:17: the variables are for multicast discovery and have nothing
    to do with the router; delete the task and merge the rest. amacati, 13:21: "Okay, then let's
    do that", and he reopened the PR.
  - What main does without this PR: the deploy env has rmw_zenoh_cpp 0.6.4 installed (through
    ros-kilted-desktop) but runs the default RMW, and tools/launch_mocap.sh sets
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST. Under the default DDS that should hide mocap from a
    second machine (not tested). The multi-drone code was never flown on the default RMW.
  - Done by the user: 03e1e2a "Remove zenoh-router task" on `multi-drone-1-zenoh`, pushed.
  - The integration branch keeps its own task and its one-line `env = {...}` form in
    pyproject.toml. Merging upstream/main will conflict there; take upstream's form.
- #146 ([Multi-drone 2], messages + comm node): CHANGES_REQUESTED by amacati, eight inline comments.
  1. ros_race_comm.py module docstring: no implementation details in the module description.
  2. EpisodeReset.msg: have one observation message and compose reset and step from it.
  3. RaceEnd.msg: how does it differ from EpisodeEnd, why not one message.
  4. StepResult.msg: unify the messages for the real and the simulated race.
  5. CMakeLists.txt: can we use cmake > 4, or does ROS not allow that?
  6. tools/setup_mocap.sh: if cmake 4 works, drop -DCMAKE_POLICY_VERSION_MINIMUM=3.5.
  7. ros_race_comm.py `_suppress_shutdown_thread_errors`: "fishy"; turning a KeyboardInterrupt into
     a debug log "seems dangerous".
  8. calibrate_clock: replace it with ROS's built-in /clock topic. The host publishes it at 500 Hz,
     the clients use simulated time and so all share the host's clock.
  Notes for the answers (facts checked on 2026-10-07, 13:41 UTC; the proposals were put to the
  user and none is decided yet):
  - 1: shorten the module docstring. No decision needed.
  - 2 to 4: only RealClientAction, RealHostState and RealCalibrateClock are imported anywhere
    (integration branch and PR 102 head). The other six came with ratheron's fb33705 (2026-06-09)
    and were never used by code on PR 102. Suggested: remove the six from this PR (was PR 9),
    which settles the three threads. Contradicts "carried over whole and unchanged" under Settled.
  - 5 and 6: cmake is pinned to 3.26.0 in [tool.pixi.dependencies] by #94 (ratheron, 2026-06-10:
    lock updates broke the mocap build). ament_cmake in Kilted asks for 3.20, so ROS does not
    forbid cmake 4. -DCMAKE_POLICY_VERSION_MINIMUM only exists in cmake >= 4, where the mocap
    package's vendored vrpn (minimum 2.6) needs it. Under 3.26 it is ignored: the colcon log of
    this clone warns "Manually-specified variables were not used" for both packages. So the
    reviewer's relation is the other way round. Suggested: drop the flag from the line and say
    so; cmake 4 would be its own PR (unpin, new lock, mocap build).
  - 7: the hook replaces the process-wide threading.excepthook. RaceCommNode's own thread already
    catches the same exceptions in `_spin`, and ROSConnector uses processes, not threads, so the
    hook should have nothing to catch (to confirm with a node and Ctrl-C, no hardware). Plan:
    remove the hook and drop KeyboardInterrupt from `_spin`. Only shutdown output changes.
  - 8: the client timestamp is used for the worker watchdog (older than 10 control periods) and
    for debug latency logs, nothing else. The host creates the service and sleeps 1 s without
    knowing whether the clients finished. In this PR the change is only: delete calibrate_clock
    and RealCalibrateClock.srv. The replacement (host publishes /clock, client node uses
    use_sim_time) lands in the client (PR 5), the host (PR 6) and on the integration branch, and
    has to be flown again. It contradicts "clock calibration is kept as flown" under Settled, so
    it is the user's decision. Simpler fallback: the worker stamps actions on receipt.
  - The PR description says "the message package from #102, unchanged" and names
    `calibrate_clock`; both sentences change with the decisions above.
- #147 ([Multi-drone 3], client): DRAFT, opened by the user at 14:21, head b312b80, no reviewers
  requested. It is stacked on #146: two commits (b08c35e, b312b80), +799/-6 in 20 files, of which
  6 files and +540/-3 are the client's. Its description is the series list plus "Needs #146"; the
  long version is under "Drafts" at the end of this file.
  - Of the #146 review only comment 8 reaches the client: it imports calibrate_clock and
    RealCalibrateClock, calibrates in lock_until_race_start and adds `_clock_offset` to the
    timestamp (about 10 lines). Comments 1 to 7 do not change client code.
  - Every new commit on `multi-drone-2-messages` has to be brought into `multi-drone-3-client`.
    Once #146 is squash-merged: `git rebase --onto upstream/main <last #146 commit>
    multi-drone-3-client`, which needs a force push to the fork (ask the user then).

## Settled (user, 2026-10-05/06)
- Zenoh is taken as PR 102 has it: global for the deploy env, multicast discovery, no router.
  Router-free operation confirmed by the user across two machines in the lab.
  Reconsidered and kept on 2026-10-06. Background: the code has no Zenoh-specific part (plain rclpy);
  on one subnet the default DDS would work as well, but that was never flown. Zenoh's real advantage
  is the router + client mode for machines outside the subnet. It was needed in September, when st-14
  sat behind an ASUS router (NAT), see the PR 102 comments of 2026-09-10 to 09-18. st-14 is now on the
  same subnet as st-04, so multicast is enough. Students use these two lab PCs by default. Router or
  client mode is not documented or set up; revisit only if machines outside the subnet are needed.
  Decided on 2026-10-07 after the review of #144: PR 1 keeps the three variables and has no
  `zenoh-router` task.
- Ownership: Yuming-Lee24 takes all PRs.
- Message package: keep the typed `drone_racing_msgs` package vendored in-tree under ros_ws/src, carried
  over whole and unchanged (all 8 messages and the srv), including the six types that no code uses today.
  Types that still have no consumer are deleted later, once the design is settled.
- No comment on PR 102. The context goes into each PR's description instead.
- Principle: migrate faithfully first, fix afterwards. Clock calibration is kept as flown, and with it
  the watchdog that compares against the calibrated client timestamp (the two belong together).

Changed on 2026-10-06, after the baseline flights (these replace rev 8):
- Client design: the client is cut as flown, a standalone env. The subclass design (pluggable drone link
  in the real env) is deferred to a follow-up PR after the migration is on main.
- Start-position check: no fix. The lab sets `check_drone_start_pos = false` by hand for every flight.
  Known consequence: with the shipped `true` the host raises, because the drones have no `nominal_pos`.
- Worker cleanup order and aborted start: both judged minor and parked until the host PR is cut.
- Two-drone return policy: dropped. Two drones flew and returned without problems, so there is no
  separate PR and no extra validation.
- Old PR 1 (gate index of finished drones): dropped. It only affects scripts/multi_deploy.py, which is
  removed anyway. It comes back only with the deferred client refactor.
- PR 3 (Zenoh) can be opened at any time; #137 and #142 can merge alongside it.
- No Co-authored-by trailers anywhere, neither in commits nor in PR descriptions.
- Commit messages are one short English line. Lab configs are edited by hand and never committed.
- Documentation: no docs changes in the individual PRs. All docs are updated together in one PR at
  the end of the series (PR 10 below).

## Still open (maintainers / lab)
- amacati's unanswered objection on PR 74 to a ros_ws inside the repo and to custom messages. PR 4's
  description must address it directly, and must say that six types have no consumer yet and are kept
  on purpose.
- Isolation between lab PCs under multicast. All machines on the same network with the same
  ROS_DOMAIN_ID share one graph; rmw_zenoh disables multicast by default for this reason. Different
  ROS_DOMAIN_IDs isolate: the id is the first element of every Zenoh key (rmw_zenoh design doc), and a
  local check on 2026-10-06 under the PR's settings confirmed it (same id: messages arrive; different
  ids: none). Whether the lab assigns ids per PC is the maintainers' call.
- Real-track tolerances for multi configs (2 cm in multi_level2.toml; multi_level0.toml has no
  env.randomizations block and both scripts crash on it).

## Consequences of not posting on PR 102
- No freeze of the PR head. Before cutting each PR, check that it is still 7b581ed:
  `git ls-remote upstream refs/pull/102/head` (unchanged on 2026-10-06).
- Each PR description says which files it ports from #102. No Co-authored-by trailers.
- amacati has never reviewed PR 102, so PR 4 is the first place he sees the design. Its description needs
  a short account of the process model (host, one worker per drone, one client per drone), the two
  topics, and why custom messages.

## Step 0: integration branch
Purpose: a baseline that behaves like the flown PR 102 but runs on main. Not for merge.

A. Done. `multi-drone-integration` created from upstream/main (54e64cf), without upstream tracking.

B. Done, 4731416 "Port multi-drone host/client from PR 102 as-is".
   Taken unchanged from the PR head:
   - lsy_drone_racing/envs/real_race_host_env.py
   - lsy_drone_racing/envs/real_race_client_env.py
   - lsy_drone_racing/utils/ros_race_comm.py
   - scripts/multi_deploy_host.py, scripts/multi_deploy_client.py
   - ros_ws/src/drone_racing_msgs/ (whole package, unchanged)
   - docs/multi_agent/index.md, docs/img/multi_adr_scheme.svg
   Not taken: config/multi_test*.toml, lsy_drone_racing/control/testing_race_*.py, pixi.lock,
   the deletion of scripts/multi_deploy.py.
   Hand-applied on main's versions:
   - .gitignore: ros_ws build dirs ignored one by one, so ros_ws/src/drone_racing_msgs is tracked.
     Main's `acados/` line kept (the PR's `acados/**` was not taken).
   - tools/setup_mocap.sh: the PR's rebuild condition (mocap node missing or drone_racing_msgs missing)
   - lsy_drone_racing/envs/__init__.py: client registration
   - lsy_drone_racing/utils/utils.py and utils/__init__.py: extract_config_for_rank
   - properdocs.yml: nav entry
   - pyproject.toml: only the three Zenoh variables in the kilted activation env and the zenoh-router
     task. No packaging/lark pins, no formatting changes. `pixi lock --dry-run --check`: no lock change.
   - config/multi_level0.toml, multi_level2.toml: only `radio` per drone and return_height_min/max under
     [deploy]. Main's `drone` key and main's values kept.

C. Done, fd904a0 "Adapt multi-drone host/client to current main" (ported files only):
   1. drone key: `drone_model` -> `drone` in host and client.
   2. parameter loading: crazyflow load_dynamics_params + mellinger pwm_min/pwm_max, as in main's real
      env; force2pwm from crazyflow.control.transform.
   3. 16-D state action: worker forwards a[9:13] and a[13:16]; client dummy and stop actions are 16-D
      with quaternion [0, 0, 0, 1]. The as-flown worker passed only two body-rate components (a[10:12]).
   4. dynamics argument: host, worker and client; both scripts pass config.sim.dynamics.

D. Done on 2026-10-06, without drones:
   - Tests: 86 passed, 22 skipped (20 GPU variants, 2 render tests because the run was headless),
     0 failed. ruff check and ruff format --check green.
   - First build against main's lock, clean deploy environment: drone_racing_msgs builds (4.8 s),
     motion_capture_tracking builds (16.7 s), no lark warning. Main needs no lark pin.
   - Same build after updating a PR 102 environment in place: fails. The PR's lock installs lark 1.3.1
     and lark-parser 0.12 over the same files; removing lark deletes lark/__init__.py, rosidl then fails
     with "cannot import name 'Lark'". pixi runs the activation script inside its own update, so the
     environment is left half-installed and every later activation fails the same way.
     Every machine that flew PR 102 needs this once before using the branch:
       pixi clean -e deploy
       rm -rf ros_ws/build ros_ws/install ros_ws/log
   - `ros2 interface show drone_racing_msgs/msg/RealClientAction` works; both env modules import; a
     second activation does not rebuild.

E. Done on 2026-10-06, flown by the user on fd904a0 plus the controller fix, without a router:
   single-drone scripts/deploy.py under Zenoh, host + one client, host + two clients, and the host on a
   different machine than the clients. All worked. Configs were edited by hand.
   Found while flying:
   - Main's attitude_controller_multi.py read the private `CubicSpline._c`, which SciPy 1.18 (main's
     lock) no longer has. It crashed the client and also main's multi_sim.py. Fixed in 292589a, see PR 0.
   - docs/multi_agent/index.md gives `estimator --drone_name cfXX`; the pixi task takes the name
     positionally (`pixi run -e deploy estimator cf11`). Fix it with the host PR's docs.
   - The host never resets a client's "stopped" flag: restart the host before every new attempt.
   - With check_drone_start_pos = false the host does not read start poses from mocap either, so the
     return target is the `env.track.drones` position in the config.
   - A single drone always returns at return_height_max.

F. Remaining work on the branch.
   Parked until the host PR is cut (PR 6):
   1. worker cleanup: close the drone (emergency stop) first inside `try`, close ROS in `finally`, as
      RealRaceCoreEnv.close does. Today the two ROS closes run first and can skip the stop.
   2. aborted start: track whether the race started and publish that in close(); raise instead of
      returning when the init barrier is broken; no catch-all `except Exception` in the host script.
      Today close() always publishes race_started = true and the script exits 0. A client only starts by
      mistake if it has already finished clock calibration; otherwise it times out after 120 s.
   Deferred until after the migration is on main (each its own later PR if wanted):
   - pluggable drone link in the real env; client as a subclass; the gate-index fix of the old PR 1
   - watchdog on host receive time, which would make calibration optional
   - calibration: host waits for each client to finish instead of sleeping 1 s
   - arm only after all clients are ready
   - drop the client-side command publish (the host worker already publishes it)
   - reset the "client stopped" flags so the host does not need a restart between attempts

## How a PR is cut
- Separate working directory, so the flying clone and its hand-edited configs stay untouched:
  ~/yuming_ws/lsy_drone_racing_pr (git worktree of this repo).
- New branch from upstream/main with --no-track. Content comes from the integration branch by file,
  by lines, or by cherry-pick of a single-purpose commit.
- The branch is verified on its own: tests, ruff, docs build if docs changed. It must not depend on
  anything that is not on main yet.
- Claude prepares the branch and the English PR title and description. The user pushes and opens the PR.
- After a merge: merge upstream/main into the integration branch (no rebase, so lab machines can pull).

## PR sequence
Titles (user, 2026-10-06): the six PRs of the migration carry a series tag with an index and no
total, `[Multi-drone N] <imperative title>`. The total and the full list go into a series block at the
top of each PR description, with the current PR marked. PR 0 and the later cleanups are not part of the
series and use plain titles. The numbers on the left are this plan's own and are not shown upstream.

0. Fix multi attitude controller with SciPy 1.18 (new)
   The fix of 292589a plus `test_multi_drone_controllers` in tests/integration/test_controllers.py, which
   runs both controllers of multi_level0.toml as multi_sim.py does. Fails without the fix, passes with
   it. Branch `fix-multi-attitude-controller`, commit 795e8d3, ready on 2026-10-06 (full suite 87 passed,
   22 skipped; ruff green). No hardware. Open upstream as #143.
1. Dropped (gate index of finished drones in real race env). See "Settled".
2. Deferred until after the migration (pluggable drone link in the real race env).
3. [Multi-drone 1] Switch deploy env to Zenoh with multicast discovery
   The three variables in the kilted activation env and nothing else: the `zenoh-router` task of
   bf8ac62 is removed again (decided 2026-10-07, see "Review status"). pyproject.toml only, 6
   lines. No docs (see PR 10).
   No lock change. Evidence: single-drone flight under Zenoh on 2026-10-06. Can be opened at any time.
   Branch `multi-drone-1-zenoh`, commits bf8ac62 and 03e1e2a; merged upstream on 2026-10-07 as
   7cbbaee (#144). The env is written as an `activation.env` table with comments (same three values as
   flown); the task is `[tool.pixi.feature.kilted.tasks] zenoh-router`. Checked: lock unchanged, task
   listed, ROS_DOMAIN_ID isolation under these settings.
   Raised in the PR description: tools/launch_mocap.sh sets ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
   (added in #128), which rmw_zenoh 0.6.4 does not implement, so mocap is visible across machines.
   The line was left in place for the maintainers to decide.
4. [Multi-drone 2] Add ROS2 messages and communication node
   ros_ws/src/drone_racing_msgs unchanged from PR 102 (CMakeLists.txt, package.xml, 8 messages, 1 srv),
   .gitignore, tools/setup_mocap.sh rebuild trigger, lsy_drone_racing/utils/ros_race_comm.py. ~230 lines.
   The package is byte-identical to what lab machines built from the PR branch, so existing installs stay
   compatible.
   Not inert: tools/setup_mocap.sh is the deploy-env activation script, so every checkout without the
   package runs a colcon build at its next activation. The trigger must rebuild only this package, must
   not loop if the build fails, and must survive three terminals activating at once. Step D showed what
   a failing build does: it blocks the whole environment. This hardening is new work, not on the branch.
   Lab check: fresh clone builds; existing checkout rebuilds exactly once; `pixi run -e deploy mocap` still
   starts; one single-drone lap.
   Branch `multi-drone-2-messages`, commit b08c35e, pushed to the fork on 2026-10-06; open upstream as
   #146. 14 files, +259/-3.
   - Package byte-identical to PR 102. ros_race_comm.py: code identical, only stale names in
     docstrings and one comment were corrected. .gitignore as on the integration branch.
   - Trigger in tools/setup_mocap.sh (differs from the as-flown line): main's `setup.sh` check plus
     `|| [ ! -d ros_ws/install/drone_racing_msgs ]`, and
     `flock . colcon build --packages-skip-build-finished ...`. No failure handling was added: a
     failed build stops activation, as it already does for the mocap package on main.
   - Tested in the PR worktree with its own deploy env: fresh clone builds both packages; second
     activation builds nothing; existing checkout builds only drone_racing_msgs (4 s), mocap node
     untouched; three and six activations at once all succeed with the lock (without it 11 of 12
     failed with build errors). Tests 86 passed, ruff and docs build green.
   - Still to do in the lab on this branch: `pixi run -e deploy mocap` starts; one single-drone lap.
   - PR description answers amacati's comment on #74 (2026-04-05): ros_ws is at the repo root and is
     the workspace setup_mocap.sh already creates; the package was moved in-tree on #102 by
     ratheron's commits fb33705 and 9ebb375 after an external-repository variant; one typed message
     per direction keeps action, flags and timestamp together.
5. [Multi-drone 3] Add client env and deploy script
   The client as flown: standalone env (~390 lines), registration, extract_config_for_rank with a unit
   test, client script. Review risk: the reviewers asked for reuse of the core env on PR 74; say in the
   description that the refactor follows as its own PR.
   Branch `multi-drone-3-client`, commit b312b80, pushed by the user on 2026-10-07 and open
   upstream as draft #147. 6 files, +540/-3. Recommitted on
   2026-10-07 after the user reviewed the diff in VS Code; it replaces 4cecd87 and differs from it
   only in the unit test.
   Stacked on `multi-drone-2-messages` (b08c35e), because the client imports the messages and the comm
   node. Once PR 4 is merged upstream: `git rebase --onto upstream/main b08c35e multi-drone-3-client`.
   - Client env: code identical to the integration branch (AST compared without docstrings). Text
     only: six stale docstrings corrected and one TODO comment removed. Script, registration and
     extract_config_for_rank are byte-identical to the integration branch.
   - New: test_extract_config_for_rank in tests/unit/utils/test_utils.py (fails for three broken
     variants of the helper). It sets `control_mode` per rank itself and does not look at `freq`
     (user, 2026-10-07): the user doubts that 50/100 Hz in the multi configs is intended and is
     asking the maintainers. Both values are amacati's from April 2025.
   - Checked: tests 87 passed, 22 skipped; ruff and docs build green; in the deploy env the module
     imports, the env is registered and constructs for both ranks of multi_level2.toml.
   - Not in this PR: `radio` and return_height_* in the multi configs (host only, PR 6).
   - Known and left as flown: unused `randomizations` argument; `drone_rank: int | None = None` with
     an assert; `.get(name, nan)` for other drones although all drones are tracked by estimators;
     the client script fails on multi_level0.toml (no env.randomizations).
6. [Multi-drone 4] Add host and deploy script
   Worker, orchestration, pre-flight checks, host script. No docs page (see PR 10).
   Includes the two parked fixes (F.1, F.2). Any number of drones, as flown with two. ~550 lines.
   One flight with failure injection.
7. Dropped (two-drone PR). Two drones come with PR 6 as flown.
8. [Multi-drone 5] Remove old multi-drone deploy script
   Delete scripts/multi_deploy.py, fix dangling multi_level3.toml references. After PR 6. Decide then
   whether RealMultiDroneRaceEnv and its registration go too or stay for the deferred refactor.
10. [Multi-drone 6] Document multi-drone real races
   All documentation of the series in one PR, after PR 8 (user, 2026-10-06):
   - docs/multi_agent/index.md, docs/img/multi_adr_scheme.svg and the properdocs.yml nav entry from
     PR 102, corrected: the Zenoh router is optional, not the first required step; the estimator
     task takes the name positionally (`pixi run -e deploy estimator cf11`); the host has to be
     restarted before every new attempt.
   - A note in docs/getting_started/setup.md for everyone using the deploy env. Drafted on
     2026-10-06 and removed again from PR 3:
       The deploy environment uses [Zenoh](https://github.com/ros2/rmw_zenoh) as ROS 2 middleware.
       Nodes discover each other through multicast, also across machines in the same network, so you
       do not need to start a Zenoh router. All machines in the network that use the same
       `ROS_DOMAIN_ID` (`0` by default) share their topics. Setups that should not see each other
       have to use different ids, e.g. `export ROS_DOMAIN_ID=7` in every terminal of one setup.
       Every node logs the warning `Scouting delay elapsed before start conditions are met` at
       startup, which is expected without a router.
   Review risk: the reviewers asked for usage docs together with the code on PR 74. Each PR
   description should say that the docs follow in the last PR of the series.
9. (Later, once the design is settled) Remove message types without a consumer
   Delete whichever of Action, EpisodeEnd, EpisodeReset, Observations, RaceEnd and StepResult still have
   no user, and their CMakeLists entries. (RealCalibrateClock is in use while calibration is kept.)
   Removing unused types does not change the remaining ones, so machines with an older install keep
   working without a rebuild.

Order: 0 now. 3 and 4 are independent of each other. 5 needs 3 and 4. 6 needs 5. 8 after 6. 10 (docs)
after 8. 9 last.
PRs 3 and 4 are the ones that change what single-drone users run (transport, activation script).
Everything after adds new files.
Size caveat: 5 and 6 are larger than anything a non-maintainer has landed here (largest ~+218). Seam if
reviewers want 6 smaller: worker vs orchestration.

## Must be in the host PR
- Emergency stop first, with the ROS closes in a finally block.
- Aborted start never publishes race_started; script exits non-zero.
Everything else in the host is carried over as flown, including clock calibration and the watchdog.

## Not carried into any PR
- lsy_drone_racing/control/testing_race_fast.py, testing_race_slow.py, config/multi_test*.toml.
- Global packaging / lark pins and formatting changes in pyproject.toml; the PR's pixi.lock.
- Lab-specific drone ids and channels.
- The multi_level2.toml randomization retune that reverts #100.

## Credit
No Co-authored-by trailers (user, 2026-10-06). PR descriptions name PR 102 as the source of ported files.

## Drafts
### PR description for [Multi-drone 3] (branch `multi-drone-3-client` at b312b80)
Written on 2026-10-06, before the review of #146. Revise it if comment 8 on #146 (ROS /clock instead
of calibrate_clock) is taken up: the "What the client does" section and the testing evidence change.

Title: `[Multi-drone 3] Add client env and deploy script`

```markdown
Part 3 of 6 of the multi-drone real race migration, which splits #102 into reviewable pieces:

1. Switch deploy env to Zenoh with multicast discovery (#144)
2. Add ROS2 messages and communication node (#146)
3. **Add client env and deploy script** (this PR)
4. Add host and deploy script
5. Remove old multi-drone deploy script
6. Document multi-drone real races

Needs #146, which adds the messages and the communication node the client imports. The
documentation follows in the last PR of the series.

## What this adds

- `lsy_drone_racing/envs/real_race_client_env.py`: `RealMultiDroneRaceEnvClient`, registered as
  `RealMultiDroneRaceEnvClient-v0`.
- `scripts/multi_deploy_client.py`: runs one controller against that env, one process per drone:
  `python scripts/multi_deploy_client.py --config multi_level2.toml --drone_rank 0`.
- `extract_config_for_rank` in `lsy_drone_racing/utils`: copies a multi-drone config and sets
  `env.freq`, `env.sensor_range` and `env.control_mode` from `env.kwargs[rank]`, so controllers read
  the same fields as in a single-drone config. Comes with a unit test.

The client cannot fly on its own. It needs the host of part 4, which owns the radio links.

## What the client does

- `reset` reads the track poses from mocap, opens the `ROSConnector` for the estimators of all drones
  and creates the communication node.
- `lock_until_race_start` publishes a hold action at the control frequency, waits until the host
  reports ready, calibrates its clock offset against the host and waits for the race start.
- `step` updates gate progress and sensor-range visibility from the estimator states, checks the
  safety limits and publishes the action as `RealClientAction`.
- `close` publishes `controller_stopped` and closes the ROS connections.

The client never talks to the drone. The host's worker for this drone forwards the latest action to
the Crazyflie, returns the drone to its start once the client reports `controller_stopped`, and takes
over when the newest message is older than ten control periods. That comparison is why the client
stamps its messages in the host's clock.

## Ported from #102

The env and the script are the ones from #102, adapted to current main:

- `drone` instead of `drone_model` in `deploy.drones`.
- Drone parameters are loaded through crazyflow, with the Mellinger PWM limits, as in
  `RealRaceCoreEnv`.
- Hold and stop actions in state mode are 16-D with an identity quaternion.
- New `dynamics` argument, passed from `config.sim.dynamics`.

Several docstrings that still described earlier versions were corrected.

## Not in this PR

On #74 you asked to reuse the core environment instead of adding a parallel one. The client here is
still the standalone env that was flown. It shares `EnvData`, `load_track`, `load_gate_order`,
`gate_passed` and `track_poses` with the real env, but has its own `reset`, `step` and `obs`. Making
it a subclass of `RealRaceCoreEnv` with a pluggable drone link is planned as its own PR after the
migration, so that this series stays a port of what was flown.

## Testing

- `test_extract_config_for_rank` checks the per-rank values and that the input config is unchanged.
- Test suite (87 passed, 22 skipped), `ruff check`, `ruff format --check` and the docs build pass.
- In the deploy environment the module imports with the built messages, and the env constructs for
  both drones of `multi_level2.toml`.
- Flown on 2026-10-06 with this client code and the host of part 4: host with one client, host with
  two clients, and host and clients on two machines.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```
