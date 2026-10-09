# Split plan for PR 102 (rev 10, 2026-10-09)

Refs: PR head 7b581ed (upstream refs/pull/102/head), branch point f452f92, main 54e64cf (the base of
the integration branch and of the open PR branches; upstream/main is 7cbbaee since #144).
Integration branch `multi-drone-integration`: 4731416 (port), fd904a0 (adapt to main), 292589a (controller fix).
Basis: rev 8 came from code reading only. Since then the integration branch was built, tested and flown
(2026-10-06). Line counts are still estimates.
Rev 10 records the decisions taken after the review of #146: the six unused message types are removed,
the thread excepthook is removed, and clock calibration is replaced by ROS simulated time (`/clock`).
The integration branch itself still has the as-flown code (all messages, calibration).

## Review status (2026-10-09) - read this first in a new session
The threads move: read the PRs on GitHub again before acting. Nothing below marked "suggested" has
been decided by the user.

- #143 (PR 0, SciPy fix): merged by amacati as 735437f. He asked to delete the explanatory paragraph
  of the test's docstring ("verbose test comment"); the user did.
  To do: merge upstream/main (now 7cbbaee) into the integration branch; mind pyproject.toml, see the
  last point under #144.
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
    pyproject.toml. `git merge upstream/main` reports no conflict there, but the result declares
    `activation.env` twice (the inline form and upstream's table), which is invalid TOML, and keeps
    the `zenoh-router` task (simulated in a scratch clone on 2026-10-09). Take upstream's file:
    `git merge --no-commit upstream/main && git checkout upstream/main -- pyproject.toml`, then
    commit.
- #146 ([Multi-drone 2], messages + comm node): CHANGES_REQUESTED by amacati on 2026-10-07, eight
  inline comments. ratheron and rducrist are requested reviewers and have not reviewed.
  amacati, PR comment of 2026-10-09 14:35 UTC: "Any progress on this? We will need the multi-agent
  part for the new semester". Not answered yet.
  Replies so far, all by the user: thread 4 (2026-10-08: asks rducrist whether the unused messages
  can be deleted; no answer), thread 5 (2026-10-09: ROS allows cmake > 4, it is pinned to 3.26 in
  pyproject, "Should we pinned to >4 in pyproject?"; no answer), thread 6 (2026-10-09: the flag is
  for cmake > 4, without it the mocap package does not compile there). Threads 1, 2, 3, 7 and 8
  have no reply.
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
  State on 2026-10-09. The decisions are the user's, taken on 2026-10-08/09 after each comment was
  checked (builds, local ROS runs on one PC, no hardware); see "Settled" and "Clock".
  - 2 to 4: done and pushed, d022f81 "Delete unused ROS messages". Action, EpisodeEnd, EpisodeReset,
    Observations, RaceEnd and StepResult and their CMake entries are gone. No ref imports them. They
    are byte-identical to rducrist/drone_racing_msgs, written for a simulation stepped over ROS
    (branch exp/multi-sim on rducrist's fork, never a PR here), and the three with observations
    still carry `target_gate`, which #121 removed from the observation. The type hashes of
    RealClientAction and RealHostState are unchanged (compared after a build), and an old and a
    new install exchanged messages in both directions under Zenoh on one PC.
  - 7: done and pushed, e2cdddd "Remove excepthook and KeyboardInterrupt handling from
    RaceCommNode". `_suppress_shutdown_thread_errors` and its call are gone, and `_spin` no longer
    catches KeyboardInterrupt (Python raises it only in the main thread). 30 local runs (Ctrl-C and
    normal close, default RMW and Zenoh) printed no traceback. Left as it was: `_spin` still
    swallows every exception whose class is named RCLError, also while the context is healthy.
  - 8 and 1: done and pushed on 2026-10-09 (17:45 UTC), d654357 "Replace clock calibration with ROS
    simulated time". calibrate_clock, RealCalibrateClock.srv and its CMake entry are removed,
    RaceCommNode takes the keyword-only `use_sim_time: bool = False`, and the module docstring is
    one line. The package now has two messages and no service. Type hashes unchanged again.
  - 5 and 6: answered by the user on 2026-10-09 (see above), no code change so far. Open with
    amacati: whether to pin cmake above 4. That would be its own PR (unpin, new lock, mocap and
    acados builds in the lab). Facts: cmake is pinned to 3.26.0 in [tool.pixi.dependencies] by #94
    (ratheron, 2026-06-10). ROS does not forbid cmake 4: Kilted's ament_cmake and rosidl ask for
    3.20, and the package builds with cmake 4.2.3, 4.3.1 and 4.4.2 (binaries from the pixi cache
    and a miniconda env). `cmake_minimum_required(VERSION 4.0)` fails under the pin.
    -DCMAKE_POLICY_VERSION_MINIMUM only exists in cmake >= 4, where the mocap package needs it
    (vendored vrpn asks for 2.6, pybind11 for 3.4: without the flag cmake 4 stops, with it the
    build passes). Under 3.26 it is ignored with the warning "Manually-specified variables were
    not used". So moving to cmake 4 is the case that needs the flag, not the case that allows
    dropping it. The flag text is the same as on main; the line is in the diff only because of
    `flock` and `--packages-skip-build-finished`. Also open: whether to change the inherited 3.11
    to 3.20 (what Kilted's template uses).
  - Still to do for #146: answer amacati's PR comment and threads 1, 2, 3, 7 and 8, follow up on
    4, 5 and 6, and update the PR description: "the message package from #102,
    unchanged", "and `calibrate_clock`" and "The code is as in #102" are no longer true; it should
    name `use_sim_time` and say that the client of #147 is its caller, link #144 in the series list,
    and answer the PR 74 question on custom messages (see "Still open"). Lab check still owed on
    this branch: `pixi run -e deploy mocap` starts; one single-drone lap.
  - Found on the way, not part of the review and not changed: RaceCommNode.close() does not join
    the spin thread. A process that exits right after rclpy.shutdown() aborted with "terminate
    called without an active exception" in 20 of 60 local runs (default RMW) and 26 of 32
    (Zenoh); with `self._thread.join(timeout=1.0)` before `destroy_node()` in 0 of 92. The host
    script, the worker processes and the client script all end with close() and rclpy.shutdown().
    The one-line fix belongs to RaceCommNode.close(), which #146 introduces, so decide before #146
    is merged. Put to the user on 2026-10-09, not decided.
  - Also found, inherited from PR 102 and not changed: CMakeLists.txt and package.xml declare
    std_msgs and builtin_interfaces, which no message uses (a build without them gives the same
    type hashes).
- #147 ([Multi-drone 3], client): DRAFT, opened by the user on 2026-10-07, no reviewers requested.
  It is stacked on #146. Its description is the series list plus "Needs #146"; the long version is
  under "Drafts" at the end of this file.
  - Adapted and force-pushed on 2026-10-09 (17:45 UTC; the old head b312b80 is replaced):
    `multi-drone-3-client` is rebased onto `multi-drone-2-messages` (no conflict). 48f4d02 is the
    client commit (b312b80 rebased), c52b544 "Use the host clock for client timestamps" is new.
    Any other clone that still has b312b80 checked out has to reset to the remote branch instead
    of pulling.
  - Every further commit on `multi-drone-2-messages` has to be brought into `multi-drone-3-client`
    again. Once #146 is squash-merged: `git rebase --onto upstream/main <last #146 commit>
    multi-drone-3-client` (the last #146 commit is d654357 at present).
  - After the lab test: replace the one-line body of #147 with the text under "Drafts".
  - Found on the way, inherited from PR 102 and not changed: `lock_until_race_start` raises the
    race-start TimeoutError without `stop_sending.set()`, so the hold-action thread keeps
    publishing while close() sends its stop messages, and dies with a traceback when the node is
    destroyed. The host-ready timeout a few lines above does set it. The fix is one line
    (`stop_sending.set()` before the raise in `real_race_client_env.py`). Decided by the user on
    2026-10-09: not fixed for now, #147 stays as it is. Mention it under known limitations in the
    description of #147, or fix it later in its own change.
  - Found on the way, not changed: after Ctrl-C the client's close() publishes five stop actions on
    a context that rclpy's signal handler has already shut down. In a mock with the same statement
    order the publish raised RCLError every time, so the host got no `controller_stopped` and the
    later closes were skipped. Not run with the real client; check it in the lab.

- [Multi-drone 4] host: no PR opened yet. Prepared on 2026-10-09 and pushed to the fork: branch
  `multi-drone-4-host`, one commit 2d416d0 "[Multi-drone 4] Add host and deploy script", stacked on
  `multi-drone-3-client` (c52b544). Files: lsy_drone_racing/envs/real_race_host_env.py
  and scripts/multi_deploy_host.py (taken from the integration branch), config/multi_level0.toml and
  multi_level2.toml (`radio` per drone, return_height_min/max).
  Differences from the flown host; nothing else differs (checked by diff):
  1. `/clock` at CLOCK_FREQ = 500 Hz from `init_comm`; `_calibrate_client_clocks`, its call and the
     srv import are removed (see "Clock").
  2. Worker `_cleanup`: emergency stop first in `try`, the ROS closes and rclpy.shutdown() in
     `finally` (F.1).
  3. Aborted start (F.2): `_race_started` flag, close() publishes it; `host_main_loop` raises
     RuntimeError when the init barrier is broken; no catch-all `except Exception` in the script.
  4. Worker init order `tasks = [_init_ros_comm, _init_cf, _init_ros_connector]`, because the
     calibration's 1 s pause is gone. Claude's choice, not asked for by the user; one line to revert.
  5. Docstrings: the non-existent `init_pose` parameter removed; `deploy_args` typed and described
     as the deploy config.
  Checked on 2026-10-09, after `pixi reinstall -e deploy --locked` on that PC (crazyflow 0.3.2; the
  real modules import there now): the main() of both scripts, the host, the spawned workers and the
  clients ran end to end, 18 runs, default RMW and Zenoh, with only `Crazyflie` and `ROSConnector`
  replaced by fakes. No hardware.
  - `/clock` holds 500 Hz, also while connect_drones waits and during the race loop. Host and worker
    nodes are on wall time. No action was stamped 0.0. Watchdog age: median 2 to 16 ms, at most
    23 ms. A normal race ends with return heights 2.00 and 1.75 and close() publishes
    race_started=True.
  - Cleanup: close(emergency_stop=True) comes before the ROS closes (the flown worker did it after
    them). With ROSConnector.close raising, the stop had already been sent; with Crazyflie.close
    raising, the ROS closes and rclpy.shutdown() still ran.
  - Aborted start (one worker cannot connect): RuntimeError, close() publishes race_started=False,
    exit code 1 with the traceback, the other worker sends its emergency stop, the clients keep
    waiting.
  - Client killed with `kill -9`: its worker stops after 104 ms (100 Hz) and 206 ms (50 Hz). The
    other drone finishes. The host loop does not end (as flown).
  - Init order, clients started first: with the flown order the worker got the client's queued
    actions about 90 ms after the race start, the oldest 93 ms old against the 100 ms threshold (no
    trip, 7 ms of margin). With the new order they arrive before the loop starts and are cleared.
  - Host main process stopped for 0.5 s: both workers stop their drones 105 to 218 ms later (the
    consequence documented under "Clock").
  Found, NOT changed, to decide before the PR is opened:
  a. Ctrl-C on the host (as flown). rclpy's SIGINT handler invalidates the context before the
     KeyboardInterrupt is handled, so the first statement of close(), a publish, raises RCLError:
     the barrier is not aborted, stop_event is not set, no worker is joined, exit code 1. During a
     race the drones still stop, because the same signal shuts the workers' contexts down and their
     watchdogs trip. Before the race start the workers stay at the barrier with connected, armed
     drones and the host hangs. Tested fix: `rclpy.init(signal_handler_options=
     SignalHandlerOptions.NO)` in scripts/multi_deploy_host.py (clean shutdown, both workers send
     the emergency stop, exit 0). Untested alternative: `if self._host_state_pub and rclpy.ok():`
     in close().
  b. A stop message that reaches the host before connect_drones() raises AttributeError in the
     callback (`_assigned_return_heights` is still None) and ends the host's spin thread (as flown).
     New: that thread also publishes `/clock`, so the clock stops too. Fix: create the return
     events and the height array in `__init__` before `init_comm()`. Not tested.
  c. The docstring of `crazyflie_process_worker` says SIGINT is ignored. rclpy.init() in run()
     installs rclpy's own handler afterwards, and that is what stops the drones on Ctrl-C (as
     flown).
  d. The worker logs "No command received ... handover control to host" also when the host clock
     stalls, and it is an emergency stop, not a handover.
  e. One unexplained emergency stop of a healthy drone: in 1 of 11 runs under the default RMW a
     client's host clock froze for 200 ms while its actions kept arriving and the host published
     `/clock` without a gap. It coincided with the late discovery of an extra ROS participant (the
     test's observer node) on a loaded PC. Not seen in 3 Zenoh runs. A lab test is added under
     "Clock".
     Follow-up on 2026-10-09, same PC, fake hardware, real scripts: seen once more under the
     default RMW, in 1 of 6 races with the CPU saturated (16 busy processes on 16 cores) and one
     long-lived extra node. About 1 s after the start both clients' host clock froze at the same
     time, for 329 ms and 222 ms, while the extra node received `/clock` without a gap (largest
     7.5 ms). Both workers had received nothing from their clients until then; the 100 Hz worker
     then got 10 queued actions with one stamp, 164 ms old, and stopped its drone. Under Zenoh, 14
     races without any freeze: 8 with a node joining and leaving every 0.8 s (105 joins, 2 of the
     races with the CPU saturated) and 6 in exactly the setup that froze under the default RMW;
     the oldest action a worker saw was 34 ms (50 Hz) and 23 ms (100 Hz). Under the default RMW
     with joining nodes only: 5 races, 75 joins, no freeze. So far the freeze needs the default RMW
     and a saturated PC; the lab runs Zenoh since #144. One PC only, small numbers: the lab test
     stays. Scripts: scratchpad/clock_join_test and scratchpad/host-functional on that PC.
  f. A second `/clock` publisher whose clock is ahead delays the watchdog by its lead, because the
     stamps lie in the future. Optional hardening: `abs(...)` around the age in the watchdog.
  g. In #147, as flown: at the race start the client's first step compiles `gate_passed`, so no
     action is sent for 58 to 186 ms. On a cold PC the 100 Hz worker (threshold 100 ms) stopped its
     drone 0.1 s after the start, once. RealRaceCoreEnv compiles the function up front in `_jit`;
     the client does not.
  h. rclpy.shutdown() after RaceCommNode.close() was also seen to hang, not only to abort: one
     more reason for the join in #146.
  For the description, known and as flown: with the shipped `check_drone_start_pos = true`
  check_track raises (the drones have no `nominal_pos`); multi_level0.toml has no
  env.randomizations; the host does not watch its workers and does not end its loop after a client
  crash. No CI test is possible for the host (the tests env has no ROS, cflib2 or
  drone_estimators), so the evidence above goes into the description.

- [Multi-drone 5] cleanup and docs: no PR opened. The user decided on 2026-10-10 to merge the last
  two PRs of the series (PR 8 and PR 10 below) into one. Committed on the user's PC on 2026-10-10,
  NOT pushed: branch `multi-drone-5-docs`, one commit 42e3f2b "[Multi-drone 5] Remove old deploy
  script and document multi-drone real races", stacked on `multi-drone-4-host` (2d416d0), worktree
  ../lsy_drone_racing_wt_docs.
  - scripts/multi_deploy.py is deleted. Nothing else referred to it.
  - docs/multi_agent/index.md and the nav entry in properdocs.yml, from PR 102, corrected: no Zenoh
    router; `pixi run -e deploy estimator cfXX`; no clock calibration but the host clock on `/clock`
    and the ten-period watchdog; a note to restart the host before every attempt; a warning not to
    run a second `/clock` publisher.
  - docs/img/multi_adr_scheme.svg from PR 102 with two labels changed: the `calibrate_clock`
    service box is now `/clock`, its tag `service` is now `500 Hz`; the bitmap fallbacks of these
    two labels are removed. Other labels of the figure are still as in PR 102 and do not match the
    code: topic names (`lsy_drone_racing/client/cf52/action`, `lsy_drone_racing_client/cf20/action`
    instead of `lsy_drone_racing/client/drone_<rank>/action`), node names (`lsy_race_worker_1/2`
    instead of `_0/_1`, `lsy_race_client_1` twice), `RaceHost`, `race_start` and `race_end`.
  - docs/getting_started/setup.md: unchanged. Claude had added the Zenoh and ROS_DOMAIN_ID note
    drafted for PR 3; the user removed it on 2026-10-10: not needed here, and another open PR
    with docs changes is waiting to be merged, so it can be added later. In the docs page the
    commands use `<config_name>.toml` and `--controller <controller_name>.py` (user,
    2026-10-10).
  - Checked: `properdocs build` without warnings (docs env installed in the main clone with
    --frozen), ruff, 48 unit tests.
  - Not done, to decide: RealMultiDroneRaceEnv and its registration stay, although no script uses
    them any more. The commented-out row in docs/challenge/overview.md still links
    config/multi_level3.toml. benchmarks/sim.py still loads config/multi_level3.toml, and with the
    name corrected it fails later for another reason (`_reset()` is called without `data`), so the
    benchmark is broken on main independently of this series; left out of this PR.
  - The series is now five PRs. The descriptions of #146 and #147 and the draft for the host still
    say "of 6" and list six items.

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
- Message package: the typed `drone_racing_msgs` package stays vendored in-tree under ros_ws/src.
  Until the review of #146 the plan was to carry it over whole and unchanged (all 8 messages and the
  srv), including the six types that no code uses, and to delete those later.
  Changed on 2026-10-09: the six unused types are removed in #146 (d022f81; the user's question to
  rducrist of 2026-10-08 on that thread is still unanswered), and RealCalibrateClock.srv is removed
  as well (see "Clock"). The package is now RealClientAction.msg and RealHostState.msg, both
  unchanged, so their type hashes are the same.
- No comment on PR 102. The context goes into each PR's description instead.
- Principle: migrate faithfully first, fix afterwards. Until 2026-10-09 that included the clock
  calibration as flown, together with the watchdog that compares against the calibrated client
  timestamp.
  Changed on 2026-10-09: the user follows amacati's comment on #146 and replaces the calibration
  with ROS simulated time (see "Clock"). The watchdog stays. The principle holds for everything
  the review does not touch.

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

## Clock: /clock replaces the calibration (user, 2026-10-09)
amacati on #146: "replace the whole calibrate_clock stuff with ROS's built-in /clock topic. The server
publishes it at 500Hz ... Then all the clients use simulated time, grab their times from ROS, and
automatically use the master clock." The user decided to do exactly that.

Design:
- The host main process publishes its wall clock (`time.time_ns()`) as rosgraph_msgs/msg/Clock on
  `/clock` at 500 Hz, from a timer on its comm node. The publisher is created in `init_comm`, which
  the constructor calls, so the clock is there before the host waits for clients.
- The host's and the workers' nodes stay on wall time. A node on simulated time cannot run the
  timer that publishes the clock (its timers follow `/clock`).
- The client's comm node is created with `use_sim_time=True`. The client stamps every
  RealClientAction with that node's clock and waits for a non-zero clock before it sends its first
  hold action, so every action the worker checks has a valid stamp. One exception: when the clock
  never arrives, the script's `finally` calls close(), whose five stop messages carry
  `client_ready` and a stamp of 0.0. The worker does not check the stamp of a stop message.
- The worker is unchanged: it compares `time.time()` with the stamp and stops its drone when the
  newest action is older than 10 control periods. Worker and host main process run on one machine.
- ROSConnector and the estimators are not touched: they run in their own processes on wall time.

Where:
- #146: done and pushed (see "Review status").
- #147, lsy_drone_racing/envs/real_race_client_env.py: done and pushed. `_init_comm` passes
  `use_sim_time=True`; new `_host_time()` reads the node clock; `_send_action_update` stamps with it;
  `lock_until_race_start` first waits for the host clock (TimeoutError "Timeout waiting for host
  clock"); the calibration call, `_clock_offset`, `_clock_calib_client` and the srv import are gone.
- Host PR (PR 6), lsy_drone_racing/envs/real_race_host_env.py: done on `multi-drone-4-host`
  (2d416d0, pushed, no PR yet; see "Review status"), including the subscription-first order mentioned
  below. The integration branch is still to do (F.3). What the change is:
  Remove the RealCalibrateClock import, `_calibrate_client_clocks` and its call in
  `host_main_loop`, and the calibration sentences in the docstrings. Add (this code ran in a
  stand-in host on 2026-10-09, default RMW and Zenoh, and held 500 Hz):
  ```python
  from rclpy.qos import QoSProfile, ReliabilityPolicy
  from rosgraph_msgs.msg import Clock

  CLOCK_FREQ = 500.0

  # in init_comm
  self._clock_pub = node.create_publisher(
      Clock, "/clock", QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
  )
  self._clock_timer = node.create_timer(1 / CLOCK_FREQ, self._publish_clock)

  def _publish_clock(self):
      """Publish the system time of the host on ``/clock``."""
      msg = Clock()
      msg.clock.sec, msg.clock.nanosec = divmod(time.time_ns(), 1_000_000_000)
      self._clock_pub.publish(msg)
  ```
  The QoS is the one rclpy subscribes with. `rosgraph_msgs` comes with rclpy, no new dependency.
  Mind: `_calibrate_client_clocks` also held the only pause (`time.sleep(1.0)`) between "all clients
  ready" and the release of the start barrier, and each worker creates its action subscription as
  its last init task. Without the pause the control loop can start before that subscription has
  matched. In a stand-in run under the default RMW the worker then got the client's 10 queued
  messages at once, the oldest 9 periods old against a threshold of 10; no trip, and not seen under
  Zenoh in one run. When the host PR is cut, create the subscription first
  (`tasks = [self._init_ros_comm, self._init_cf, self._init_ros_connector]`) or show in the lab
  that it does not matter.
- Also on the integration branch then: CLAUDE.md (the sentences on the calibrate_clock service and
  on the message package being identical to the PR's). Docs, for PR 10: docs/multi_agent/index.md
  line 67 ("calibrates clocks") and line 93 ("calibrates its clock against the host"), and the
  `calibrate_clock` service box in docs/img/multi_adr_scheme.svg.

Measured on one PC on 2026-10-08, replica processes, default RMW and Zenoh, no hardware:
- A 500 Hz timer in rclpy holds 496 to 499 Hz. CPU about 10 % of a core on the host and on each
  client (100 Hz would be about 3 to 4 %).
- The client's stamp is behind the host clock by about 1 ms when idle and by 6 ms in the median,
  16 ms at most, when a Python controller keeps the interpreter busy. The watchdog thresholds are
  100 ms (100 Hz drone) and 200 ms (50 Hz drone); no false trip in any run.
- The error only makes actions look older, never newer, as long as there is one publisher.
- 2026-10-09: the changed client methods, run from their unmodified source in a harness against a
  stand-in host and worker (the module itself does not import on that PC, whose deploy env is
  outdated): no hold or control action was stamped 0.0, no backward step, no false trip. Age seen
  by the watchdog: median 9 to 18 ms, at most 23 ms. A client started before the host waits in
  "Waiting for host clock" and then runs normally; without a host it raises the TimeoutError.

Known behaviour, to be tested in the lab and accepted or not:
- If `/clock` stops while actions still arrive (host main process stalls or dies), the client stamps
  freeze and every worker stops its drone after 10 control periods: an emergency stop of all drones
  at once. Measured: stopping the publishing process for 0.5 s tripped the check. With the
  calibration the host main process was not part of that loop.
- `/clock` is one global topic. Any other publisher on the same ROS graph (a second host with the
  same ROS_DOMAIN_ID, `ros2 bag play --clock`, a simulator) is applied by the clients without a
  warning. Run `ros2 topic info /clock -v` in the lab before flying; this is the same question as
  "Isolation between lab PCs" under "Still open".
- A client without a host raises TimeoutError after its timeout (120 s in the script). Its close()
  then still sends the five stop messages with `client_ready` and a stamp of 0.0 (see "Design").
- Old and new code do not work together: a new client waits for a `/clock` that the as-flown host
  never publishes (and after 120 s reports ready and stopped through close()), and an as-flown
  client waits for a calibration service that a new host no longer offers. Update host and clients
  on every machine in one step.
- The client's log line "Clock offset = ... ms" is gone; it was the only direct display of the
  clock difference between the lab PCs. The client's debug log of the host latency
  (`time.time() - msg.timestamp`) still compares two wall clocks and was not changed.

Lab test before the host PR is opened (the watchdog input changed, so this replaces part of the
flights of 2026-10-06). It needs F.3 on the integration branch first.
- The watchdog only runs after the race has started. So start the race with the drones staying on
  the ground (a controller that does not take off), then:
  - `kill -9` a client (not Ctrl-C, which sends the stop message instead): its worker has to log
    "No command received" and stop within about 11 control periods (0.22 s at 50 Hz, 0.11 s at
    100 Hz);
  - stop only the host main process (`kill -STOP <pid>`; Ctrl-Z would stop the workers as well):
    all workers have to stop. `kill -CONT` or kill it afterwards.
- Start a client without a host: it has to time out. Start the clients first and the host last:
  the race has to start normally.
- `ros2 topic info /clock -v` shows exactly one publisher.
- During a race on the ground, start and stop other ROS participants (`ros2 topic hz /clock`,
  `ros2 node list`, RViz): no worker may log "No command received". This is finding e under
  "Review status"; if it happens under Zenoh, the watchdog input has to be discussed with amacati.
- Watch the first 0.2 s after "Race started" on the 100 Hz drone (finding g): a stop there comes
  from the client's first step, not from `/clock`.
- Flights: host + one client, host + two clients, host and clients on two machines. No worker may
  log "No command received" during a normal flight.

## Still open (maintainers / lab)
- amacati's objection on PR 74 to a ros_ws inside the repo and to custom messages ("Do we need these
  messages? Can't we build them out of existing ones?"). N0OBSTUDENT answered the ros_ws part on
  2026-04-26 (external repository), ratheron moved the package back in-tree on 2026-06-09; whether
  custom messages are needed at all was never answered. The posted description of #146 does not
  address it. With two messages and no service left, the updated description has to.
- Whether ratheron and rducrist, the requested reviewers of #146, want the six removed simulation
  types back. They would return with the PR that adds their consumer, with the current observation
  keys.
- Isolation between lab PCs under multicast. All machines on the same network with the same
  ROS_DOMAIN_ID share one graph; rmw_zenoh disables multicast by default for this reason. Different
  ROS_DOMAIN_IDs isolate: the id is the first element of every Zenoh key (rmw_zenoh design doc), and a
  local check on 2026-10-06 under the PR's settings confirmed it (same id: messages arrive; different
  ids: none). Whether the lab assigns ids per PC is the maintainers' call.
- Real-track tolerances for multi configs (2 cm in multi_level2.toml; multi_level0.toml has no
  env.randomizations block and both scripts crash on it).

## Consequences of not posting on PR 102
- No freeze of the PR head. Before cutting each PR, check that it is still 7b581ed:
  `git ls-remote upstream refs/pull/102/head` (unchanged on 2026-10-06 and on 2026-10-09).
- Each PR description says which files it ports from #102. No Co-authored-by trailers.
- amacati has never reviewed PR 102, so PR 4 (#146) was the first place he saw the design. Its description still needs
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
      With `/clock` instead of the calibration (see "Clock") nothing holds a client back any more: a
      client that has seen the host ready starts its controller on that message.
   Not parked, needed for the lab test under "Clock" (suggested steps, 2026-10-09):
   3. bring the reviewed comm node, client and message package onto the branch from
      `multi-drone-3-client` (c52b544). A merge or a cherry-pick conflicts, because the branch added
      the same files on its own; take them by file instead:
        git checkout origin/multi-drone-3-client -- lsy_drone_racing/utils/ros_race_comm.py \
            lsy_drone_racing/envs/real_race_client_env.py \
            ros_ws/src/drone_racing_msgs/CMakeLists.txt tools/setup_mocap.sh
        git rm ros_ws/src/drone_racing_msgs/srv/RealCalibrateClock.srv \
            ros_ws/src/drone_racing_msgs/msg/{Action,EpisodeEnd,EpisodeReset,Observations,RaceEnd,StepResult}.msg
      Then change the host as described under "Clock". Until the host is changed it does not import
      (it still imports RealCalibrateClock) and cannot run with the new client. Update every lab
      machine in one step. Existing builds of the message package keep working (same type hashes).
   Later:
   4. after #146 is merged: merge upstream/main (pyproject.toml: see #144 under "Review status").
      The files that F.3 took from the PR branch should merge without a conflict if #146 was merged
      with the same content; otherwise take upstream's side. Without F.3 the merge conflicts in
      ros_race_comm.py and CMakeLists.txt (add/add) and in tools/setup_mocap.sh, the six removed
      .msg files and the .srv survive it and have to be deleted by hand, and host and client no
      longer import.
   Deferred until after the migration is on main (each its own later PR if wanted):
   - pluggable drone link in the real env; client as a subclass; the gate-index fix of the old PR 1
   - watchdog on host receive time, which needs no clock shared between the machines (looked at on
     2026-10-08 as the alternative to `/clock`; the user chose `/clock`)
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
- Review fixes are made on the PR branch first; the integration branch takes them over afterwards
  (F.3, F.4). For the host PR the content still comes from the integration branch.

## PR sequence
Titles (user, 2026-10-06): the six PRs of the migration carry a series tag with an index and no
total, `[Multi-drone N] <imperative title>`. The total and the full list go into a series block at the
top of each PR description, with the current PR marked. PR 0 and the later cleanups are not part of the
series and use plain titles. The numbers on the left are this plan's own and are not shown upstream.

0. Fix multi attitude controller with SciPy 1.18 (new)
   The fix of 292589a plus `test_multi_drone_controllers` in tests/integration/test_controllers.py, which
   runs both controllers of multi_level0.toml as multi_sim.py does. Fails without the fix, passes with
   it. Branch `fix-multi-attitude-controller`, commit 795e8d3, ready on 2026-10-06 (full suite 87 passed,
   22 skipped; ruff green). No hardware. Merged upstream as 735437f (#143).
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
   State after the review (2026-10-09, see "Review status"): d022f81, e2cdddd and d654357 pushed.
   With all three the PR is 7 files, +122/-3: .gitignore, tools/setup_mocap.sh,
   CMakeLists.txt, package.xml, RealClientAction.msg, RealHostState.msg and ros_race_comm.py
   (RaceCommNode only). The two messages have the type hashes of PR 102, so existing installs stay
   compatible. The rest of this entry describes the plan and the PR as it was opened (b08c35e).
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
   - The PR description was meant to answer amacati's comment on #74 (2026-04-05); the posted one
     does not (see "Still open"). The intended answer: ros_ws is at the repo root and is
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
   node. Once PR 4 is merged upstream: `git rebase --onto upstream/main <last commit of #146>
   multi-drone-3-client`.
   Adapted and force-pushed on 2026-10-09 (see "Review status" and "Clock"): rebased onto the reviewed
   `multi-drone-2-messages`, plus the commit "Use the host clock for client timestamps". The client
   no longer equals the integration branch: it reads the host clock from `/clock` instead of
   calibrating. The sentences below describe b312b80.
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
   Since 2026-10-09 also: the host publishes `/clock` and the calibration is removed (F.3, "Clock").
   That changes the watchdog's input, so the lab test listed under "Clock" comes before this PR.
7. Dropped (two-drone PR). Two drones come with PR 6 as flown.
8 and 10 are merged into one PR since 2026-10-10 (user): `[Multi-drone 5] Remove old deploy script
   and document multi-drone real races`, see "Review status". The two entries below are the
   original plan.
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
9. Dropped on 2026-10-09: the six message types without a consumer are removed in #146 (PR 4), and
   RealCalibrateClock goes with the calibration.

Order: 0 (#143) and 3 (#144) are merged. 4 (#146) is in review, 5 (#147) is a draft stacked on it.
6 needs 5 and the lab test under "Clock". 8 after 6. 10 (docs) after 8.
PRs 3 and 4 are the ones that change what single-drone users run (transport, activation script).
Everything after adds new files.
Size caveat: 5 and 6 are larger than anything a non-maintainer has landed here (largest ~+218). Seam if
reviewers want 6 smaller: worker vs orchestration.

## Must be in the host PR
- Emergency stop first, with the ROS closes in a finally block.
- Aborted start never publishes race_started; script exits non-zero.
- The `/clock` publisher instead of the clock calibration (see "Clock"), tested in the lab first.
Everything else in the host is carried over as flown, including the watchdog.

## Not carried into any PR
- lsy_drone_racing/control/testing_race_fast.py, testing_race_slow.py, config/multi_test*.toml.
- Global packaging / lark pins and formatting changes in pyproject.toml; the PR's pixi.lock.
- Lab-specific drone ids and channels.
- The multi_level2.toml randomization retune that reverts #100.

## Credit
No Co-authored-by trailers (user, 2026-10-06). PR descriptions name PR 102 as the source of ported files.

## Drafts
### PR description for [Multi-drone 3] (branch `multi-drone-3-client`)
Written on 2026-10-06 for b312b80. Revised on 2026-10-09 for the `/clock` change: "What the client
does", "Ported from #102" and the last bullet of "Testing". The flights of 2026-10-06 used the
calibration, so the testing section may only claim them for the rest of the client until the lab
test under "Clock" is done. The test-suite and import bullets were measured at b312b80: run them
again on the tip that is posted (the count also changes once the branch sits on a main with #143's
test).

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
- `lock_until_race_start` waits for the host clock, then publishes a hold action at the control
  frequency until the host reports ready and starts the race.
- `step` updates gate progress and sensor-range visibility from the estimator states, checks the
  safety limits and publishes the action as `RealClientAction`.
- `close` publishes `controller_stopped` and closes the ROS connections.

The client never talks to the drone. The host's worker for this drone forwards the latest action to
the Crazyflie, returns the drone to its start once the client reports `controller_stopped`, and stops
the drone when the newest message is older than ten control periods. That comparison is why the
client stamps its messages with the host's clock: its communication node uses ROS simulated time and
reads the clock that the host publishes on `/clock` (as suggested in the review of #146).

## Ported from #102

The env and the script are the ones from #102, adapted to current main:

- `drone` instead of `drone_model` in `deploy.drones`.
- Drone parameters are loaded through crazyflow, with the Mellinger PWM limits, as in
  `RealRaceCoreEnv`.
- Hold and stop actions in state mode are 16-D with an identity quaternion.
- New `dynamics` argument, passed from `config.sim.dynamics`.
- The clock calibration service of #102 is replaced by the host clock on `/clock`.

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
- Flown on 2026-10-06 with the host of part 4, still with the clock calibration of #102: host with
  one client, host with two clients, and host and clients on two machines. The `/clock` path was
  only exercised without hardware so far: the changed methods against a stand-in host and worker
  (TODO before leaving draft: replace this with the lab result).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```
