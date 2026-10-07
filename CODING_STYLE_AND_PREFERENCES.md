# Repository Coding Style, Habits, and Review Preferences

This guide describes how to write changes that fit `lsy_drone_racing` and how to prepare them for
review by `amacati` and `ratheron`. It combines enforced project tooling, maintained code,
tests, repository history, and feedback from closed pull requests in the upstream repository.

The local-code snapshot is branch `deploy-viewer` at `d37621f`. The review corpus was checked on
2026-07-31. Revisit this guide when project tooling, core architecture, or maintainer guidance
changes.

## Review corpus and confidence

The upstream review was performed against the exact GitHub accounts, not profile display names:

| Reviewer | Closed-PR evidence inspected |
| --- | --- |
| `amacati` | 126 inline comments in 119 threads across 13 PRs; 21 conversation comments across 17 PRs; 52 submitted reviews across 26 PRs |
| `ratheron` | 54 inline comments across 9 PRs; 11 conversation comments; 29 submitted reviews; 25 closed PRs matched the reviewer/commenter search overall |

Together, the closed/merged corpus contains 293 records: 180 inline comments, 32 conversation
comments, and 81 submitted reviews. The inline comments occupy 159 threads. PR #102 was still open
and was deliberately excluded from these totals and from the derived guidance. This is complete for
records currently exposed by GitHub; deleted comments cannot be recovered.

For every inline comment, the audit could see the file, current and original line, diff hunk,
replies, direct URL, resolution state, and outdated state. Guidance in this document is therefore
based on the commented code and full thread context, not comment text in isolation.

Interpret the evidence carefully:

- Repeated, direct requests across PRs are treated as strong preferences.
- The final reply, resulting code, and later approval outweigh an initial suggestion.
- A resolved flag is useful bookkeeping, but old threads can remain unresolved after their code was
  replaced.
- PR #74 was closed without merging. It is strong evidence of what the reviewers requested, but not
  merged-code precedent.
- `ratheron` authored PRs #103 and #121. His replies there are supporting context, not independent
  supervisor review requests.
- `mschuck-rai` and `amacati` are distinct accounts with the same public profile name. Feedback by
  `mschuck-rai` was not attributed to `amacati`.

## How to use this guide

When instructions appear to conflict, use this order:

1. The task's intended behavior, public contracts, and physical safety.
2. Tests and enforced configuration in `pyproject.toml` and CI.
3. Explicit maintainer feedback, interpreted in its full thread and current context.
4. Repeated patterns in maintained first-party code.
5. This guide.

Do not turn every review question into a universal rule. A request tied to one ROS lifecycle,
specific shell expression, or one line wrap remains contextual unless it expresses a repeated
principle.

## Supervisor review standard

The two reviewers have compatible emphases. `amacati` most often pushes for fail-fast contracts,
explicit APIs, precise names, minimal exception handling, and regression proof. `ratheron` most
often pushes for reuse of existing architecture, direct code, narrow scope, behavioral tests,
workflow documentation, and hardware-safe lifecycle order.

| They favor | They push back on |
| --- | --- |
| Useful functionality with a clear student, user, or operational need | Machinery whose necessity cannot be explained from the PR |
| Extending the existing core, state types, connectors, and fixtures | Parallel classes, state containers, resources, examples, or environments |
| Required values accessed directly with accurate types and defaults | Defensive defaults, casts, clamps, or `None` paths that conceal broken contracts |
| Narrow parameters and visible dependencies | Passing a whole `ConfigDict` or other god object into leaf code |
| Precise domain terms used consistently across layers | Abbreviations, vague names, misleading flag/path names, and internal ranks in user APIs |
| Short, direct control flow that remains easy to read | Redundant wrappers, deep nesting, narrational comments, or clever compression |
| Regression, negative, integration, and hardware evidence proportional to risk | Happy-path-only tests or deployment changes not checked on real hardware |
| Focused diffs plus contract and workflow documentation | Unrelated refactors, extra options, stale alternatives, and accidental lockfile churn |

### What receives positive feedback

Positive comments consistently combine real usefulness with a small design and evidence:

- `amacati` liked [saving measured track layouts](https://github.com/learnsyslab/lsy_drone_racing/pull/56#discussion_r2537352998)
  once the student-facing use was explained, called [live track
  visualization](https://github.com/learnsyslab/lsy_drone_racing/pull/105#pullrequestreview-4517004479)
  useful while still asking to reduce its code, and praised both a [demonstrated regression
  test](https://github.com/learnsyslab/lsy_drone_racing/pull/66#issuecomment-3777056917) and a
  [thorough real-hardware test
  sheet](https://github.com/learnsyslab/lsy_drone_racing/pull/56#issuecomment-3629554787).
- `ratheron` called a [configuration-extraction
  helper](https://github.com/learnsyslab/lsy_drone_racing/pull/74#discussion_r3032379621) elegant,
  praised a deployment implementation's [safety
  guards](https://github.com/learnsyslab/lsy_drone_racing/pull/74#pullrequestreview-4055182615),
  and liked [new user-facing
  tasks](https://github.com/learnsyslab/lsy_drone_racing/pull/74#discussion_r3032402382) while still
  requesting semantic tests and usage docs.
- Recent positive signals on the repository author's own work include [“Overall nice
  PR”](https://github.com/learnsyslab/lsy_drone_racing/pull/76#pullrequestreview-4076160605),
  [“I like the
  change”](https://github.com/learnsyslab/lsy_drone_racing/pull/122#pullrequestreview-4742330888),
  and [“Nice!”](https://github.com/learnsyslab/lsy_drone_racing/pull/124#pullrequestreview-4766598457).

Approval is not evidence that every line is a preferred pattern. The stronger lesson is that a
useful addition reviews well when its purpose is obvious, its implementation is economical, and
its behavior is demonstrated.

### What most often triggers changes

- A function silently compensates for a state that is supposed to be impossible.
- A broad or bare exception catch hides the first actionable failure.
- New code duplicates an existing connector, environment, state field, fixture, or controller
  example.
- A helper, property, wrapper, base class, or configuration option exists without a present use.
- Names force the reader to infer type, scope, units, or domain meaning.
- A bug fix lacks a test that actually fails on the unfixed behavior.
- Comments repeat nearby syntax while an important public or operational contract remains
  undocumented.
- The PR changes unrelated files or includes generated dependency churn without a dependency
  change.

## Quick checklist

Before requesting review:

- Explain why the behavior is needed and keep every changed file connected to that purpose.
- Search for an existing core abstraction, state type, connector, fixture, or example to extend.
- Pass only what a function needs; make types and defaults describe valid calls.
- Access required fields directly. Use fallbacks only for data that is genuinely optional.
- Give each function one coherent responsibility, but do not extract trivial indirection.
- Choose full, precise names and reuse the repository's domain vocabulary.
- Let invalid contracts fail visibly; catch only exceptions that can be handled meaningfully.
- Preserve the original exception during partial initialization and make shutdown physically safe.
- Add a focused regression or behavior test, including negative and boundary cases when relevant.
- Validate deployment-sensitive changes on real hardware when feasible and record the result.
- Remove stale alternatives, unused options, commented-out code, and narrational comments.
- Run Ruff formatting, Ruff linting, and the relevant tests.
- Resolve addressed review threads and keep the PR targeted at the correct fork/base branch.

## Enforced Python style

`pyproject.toml:155-209` and `.github/workflows/ruff.yml:7-19` are the source of truth:

- Target Ruff's Python 3.11 syntax rules.
- Indent with four spaces and keep lines at or below 100 characters.
- Use double quotes.
- Let Ruff sort imports and format code.
- Follow lint families `E`, `F`, `I`, `D`, `TCH`, and `ANN`.
- Use Google-style docstrings.
- Add type annotations where Ruff requires them. Use `Any` only when a more precise contract is not
  practical.
- Do not rely on a trailing comma to force multiline formatting; magic trailing commas are
  disabled.
- Tests and benchmarks may omit some module or function docstrings. Production modules and public
  APIs should be documented.

Typical local checks:

```bash
pixi run ruff format --check --diff .
pixi run ruff check .
pixi run -e tests tests -v
```

Apply mechanical formatting intentionally:

```bash
pixi run ruff format .
pixi run ruff check --fix .
```

CI tests use the locked Pixi environment. The `tests` task expands to `pytest -v tests`.

## Imports and typing

- Group imports as future, standard library, third party, then local package imports.
- Keep normal imports at module scope and finish imports before definitions. Inline imports have
  received repeated review pushback.
- Put imports used only for annotations behind `if TYPE_CHECKING:`, especially for heavy or optional
  dependencies. If a runtime import truly must be deferred to avoid an optional-dependency or cycle
  problem, make that constraint explicit.
- Add `from __future__ import annotations` in typed production modules following the existing
  pattern.
- Prefer built-in generics such as `list[str]` and PEP 604 unions such as `Path | None`. Do not use
  `typing.Union` in new code.
- Use `numpy.typing.NDArray` for NumPy APIs and `jax.Array` or the established `Array` alias for JAX
  APIs where practical.
- Use established aliases: `numpy as np`, `jax.numpy as jp`, functional simulation helpers as `F`,
  and SciPy `Rotation as R`.
- Use `pathlib.Path` for filesystem paths.
- Test for `None` explicitly. Do not write `value or default` when an empty or false value is valid.

See `lsy_drone_racing/control/controller.py:18-27` and
`lsy_drone_racing/envs/race_core.py:21-59`.

## Naming, contracts, and API shape

### Names

- Use `snake_case` for modules, functions, methods, and variables; use `PascalCase` for classes.
- Prefix private implementation state with `_`.
- Prefer complete words over ambiguous abbreviations: `drone_channel` is clearer than `drone_ch`.
- Make the name reveal the value's role. A filesystem destination should use a `_path` suffix; a
  name such as `save_track` misleadingly sounds like a boolean.
- Use domain suffixes consistently: `_pos`, `_quat`, `_vel`, `_fn`, `_ids`, and `_path`.
- Prefix counts with `n_`, such as `n_drones` and `n_gates_passed`.
- Use one term for one concept across controllers, messages, environments, scripts, tests, and logs.
  Distinguish controller-, drone-, and race-level state.
- Avoid `get_...` for attribute-like queries. Prefer a noun or behavior that says what is returned.
- Expose user-facing identifiers such as a drone name instead of an internal rank when practical.
- Conventional mathematical names such as `N`, `Q`, `R`, `Tf`, `J`, and `J_inv` remain appropriate in
  math-heavy code.

### Contracts and data flow

- Keep signatures explicit and narrow. Pass the arrays, paths, flags, or collaborators actually
  needed instead of an entire nested `ConfigDict`.
- A type annotation and default must describe a usable call. Do not accept or default to `None` when
  the function cannot operate with `None`.
- Access required configuration and state directly so a missing field or wrong type fails at its
  source. Do not use `.get`, coercion, clipping, minimums, or reshaping merely to conceal a violated
  invariant.
- Use `.get` with an intentional fallback for genuinely optional data. This is not a blanket ban on
  dictionary defaults.
- Prefer keyword-only parameters for optional tuning values when positional use would be unclear.
- Keep conditions about whether an operation should run at the caller when practical. The called
  function should perform the behavior promised by its name rather than silently no-op behind an
  unrelated flag.
- Avoid hard-coded operational values and magic numbers. Put volatile or scenario-dependent values
  in named constants or configuration.
- Preserve Gymnasium's `reset`, `step`, observation, action, termination, and truncation contracts.

Controller files have an additional stable contract:

- Put implementations in `lsy_drone_racing/control/`.
- Inherit from `Controller` and preserve its callback signatures.
- Define exactly one local `Controller` subclass per file. Dynamic loading enforces this.

See `lsy_drone_racing/control/controller.py:13-15` and
`lsy_drone_racing/utils/utils.py:23-50`.

## Structure, reuse, and state

- Before creating a class, data container, connector, environment, controller example, property, or
  wrapper, identify why the existing one cannot be extended.
- Keep shared simulation behavior in `RaceCoreEnv`. Keep single- and multi-drone Gymnasium adapters
  thin, and make the real environment mirror the simulation interface where hardware permits.
- Reuse established environment data and canonical state instead of adding parallel bookkeeping.
  Do not duplicate values such as drone counts or PRNG state.
- Give each function one cohesive responsibility. Separate state mutation, readiness checks, and
  validation when they are independently meaningful.
- Extract an operation when its name exposes a domain concept or reusable responsibility. Do not
  create an 11-line wrapper around a readable one-liner merely to shorten its caller.
- Do not introduce speculative inheritance for one implementation. If consumers need a structural
  interface without shared implementation, consider a `Protocol`.
- Remove a superseded implementation instead of retaining multiple indistinguishable options.
- Keep each heavyweight resource under one clear owner. Reuse ROS connectors rather than opening
  parallel connections for adjacent responsibilities.
- Keep ROS workspaces and packages in their established system/workspace boundary, not embedded
  inside the Python package.
- Prefer deterministic synchronization, acknowledgements, or blocking APIs to fixed sleeps when
  the underlying system provides them.

For state-heavy numerical code:

- Represent performance-critical state with typed structs.
- Update immutable state with `.replace()` or indexed `.at[...]` operations.
- Keep small controller lifecycle state, such as ticks and completion flags, on the controller and
  update it through callbacks.
- Use `# region` headings only when they make a genuinely large module easier to navigate.

## Readability, docstrings, and comments

Optimize for visual readability, not the fewest characters:

- Reduce avoidable indentation, brackets, and sprawling multiline assignments.
- Use an early guard or `continue` when it makes the main path more direct.
- Split dense conditions into named pieces.
- Introduce a short intermediate variable when it makes a long expression easier to read.
- Keep a concise assertion or expression on one line when it remains clear.
- Never shorten a meaningful name or compress logic just to satisfy line length.

Write module and public API docstrings that state purpose and contract. Use Google sections such as
`Args`, `Returns`, `Raises`, `Note`, `Warning`, and `Todo` only when they add information.

Document:

- units, shapes, coordinate frames, and quaternion order when they are part of the contract;
- lifecycle, concurrency, and physical safety constraints;
- public specialization such as “multi-drone only” or “wraps the default controller”;
- setup steps and new operational workflows;
- non-obvious domain decisions and JAX transformation constraints;
- why an unavoidable fallback or special case exists.

Do not:

- repeat types already visible in a typed signature;
- narrate syntax visible immediately below the comment;
- leave stale or commented-out implementation;
- add generic background prose or generated-sounding filler.

The history explicitly removes overly generated-sounding text (`35edd69` and `dd3b794`). Use
MkDocs/Google conventions for new documentation rather than copying isolated reStructuredText from
older files.

## Errors, cleanup, concurrency, and safety

### Fail visibly

- Program fail-fast around declared contracts. A required field, guaranteed boolean, valid index,
  or internal state should not be quietly repaired so execution can continue.
- Use assertions for programmer sanity invariants, such as internal shapes that should be impossible
  to violate through a valid public call.
- Raise a specific `ValueError`, `KeyError`, `TypeError`, or `RuntimeError` for expected invalid user
  input, files, configuration, or runtime conditions. Assertions can disappear under `python -O`.
- Preserve an already actionable lower-level error. Translate only when the new exception adds
  domain context, and chain it with `raise ... from error`.

### Catch sparingly

- Use as few `try/except` blocks as possible. Early failure is usually preferable.
- Do not catch, log, and immediately re-raise without meaningful recovery or added context.
- Bare `except:` blocks are effectively prohibited. An exceptional requirement would need a
  detailed explanation of why no narrower handling is possible.
- Catch a specific exception only when the code can recover, provide a clearer boundary error, or
  intentionally keep a best-effort secondary feature from breaking the primary operation.
- Remove defensive exception handling when its underlying failure mode no longer exists.

### Always clean up without hiding the cause

- Use `try/finally` or a context manager when hardware, GUI, ROS, threads, processes, or environments
  must be released.
- Handle partial initialization so cleanup does not replace the constructor's original failure with
  a secondary `None` or `close()` error.
- Make resource ownership and shutdown order explicit.
- During hardware shutdown, issue the safest physical command, such as emergency stop, before
  stopping the updater or tearing down communication. Continue safe cleanup where practical.
- Put CLI execution in an `if __name__ == "__main__":` block.

For real-time and concurrent paths:

- Use `time.perf_counter()` for loop timing, sleep only for the remaining period, and warn when
  control computation overruns.
- Keep the control loop free of slow estimator work.
- Make cross-thread coroutine submission explicit.
- Use the multiprocessing `spawn` context and a barrier where the established multi-drone design
  requires them.

See `scripts/deploy.py:52-102`, `lsy_drone_racing/utils/crazyflie.py:216-267`, and
`lsy_drone_racing/envs/real_race_env.py:330-350`.

## JAX and performance-sensitive code

- Build simulation operations as pure functions or closures.
- Reuse the simulation's PRNG state. When several subkeys are needed, split them in one operation and
  carry the new key forward.
- Put repeated numerical validation and sanitization in the shared compiled core rather than
  duplicating slow outer-environment code.
- JIT reusable simulation reset, step, and action functions as coherent units, and vectorize worlds
  with `jax.vmap`. Do not assume user controllers are JIT-compatible.
- Prefer array operations and `jp.where` over Python control flow inside transformed code.
- Keep shapes and dtypes stable. Avoid unnecessary copies, bool-to-int casts, or other changes that
  trigger recompilation.
- Do not clamp or cast merely to hide a broken invariant.
- Preserve reshaping or canonicalization that a downstream transform actually requires. A reviewer
  explicitly retracted one removal request after that requirement was explained.

The clearest local references are:

- `lsy_drone_racing/envs/race_core.py:67-177` for typed environment state;
- `lsy_drone_racing/envs/race_core.py:496-568` for compiled reset and step closures;
- `lsy_drone_racing/envs/randomize.py:1-5,24-50` for pure randomization functions;
- `lsy_drone_racing/envs/randomize.py:260-294` for vectorization and PRNG splitting.

## Testing habits

Validation should be proportional to the behavior and risk:

- A bug fix needs a regression test that fails against the unfixed behavior and passes with the fix.
  Confirm the pre-fix failure locally when practical.
- Test semantics, not merely successful execution. Assert representative values when values define
  the behavior.
- Test shapes, dtypes, and masks when those are the contract; do not hard-code unrelated values that
  make the test resist valid refactoring.
- Include invalid input, reverse direction, outside-boundary, unset-environment, completion, reset,
  and cleanup cases as relevant.
- Exercise supported interfaces together when a shared architecture promises consistent behavior.
- Put suite-wide fixtures and environment setup in `tests/conftest.py`.
- Parametrize backend, physics, device, configuration, and control-mode variants rather than
  duplicating tests.
- Use deterministic seeds when randomness matters.
- Prefer `numpy.testing` or `allclose`/`array_equal` as appropriate. Check dtype strictly when it is
  part of the contract.
- Construct immutable state directly when that isolates core logic from expensive physics.
- Deployment-sensitive changes should be exercised on real hardware when feasible. Record the
  tested matrix, result, and any unavailable case in the PR.

Test organization:

- `tests/unit/`: isolated logic and fast state transitions;
- `tests/integration/`: lifecycle and backend combinations;
- `tests/deploy/`: manual hardware-oriented validation outside the normal Pytest suite.

Use the registered `unit` and `integration` markers for new tests. Reuse the shared headless marker
for graphical skips, and skip an unavailable accelerator at runtime rather than requiring it for
the whole suite.

Useful commands:

```bash
pixi run -e tests pytest tests/unit -v
pixi run -e tests pytest tests/integration -v
pixi run -e tests pytest path/to/test_file.py -v
```

## Configuration, dependencies, and documentation

- Keep TOML loading at the configuration boundary. Leaf helpers should receive explicit typed
  values, not the whole nested `ConfigDict`.
- Treat level configurations as behavioral fixtures. Keep equivalent initial conditions consistent
  across related levels unless the difference is intentional and documented.
- Put operational comments near non-obvious values, especially units, valid choices, and safety
  consequences.
- Do not introduce a volatile default, such as a particular drone, when the right value can vary
  between runs.
- During normal controller or course work, preserve supplied scenarios unless broader configuration
  change is explicitly in scope.
- Use Pixi as the primary environment manager.
- When a dependency change is intentional, update `pyproject.toml` with the generated `pixi.lock`.
  Do not hand-edit the lockfile.
- Exclude lockfile churn from PRs that do not change dependencies.
- Keep optional groups (`sim`, `gpu`, `deploy`, `rl`, and `gamepad`) and Pixi features (`kilted`,
  `tests`, `profile`, and `docs`) scoped to the feature that requires them.
- Document new tasks, setup requirements, flags, and user workflows. Build public API docs with
  `pixi run -e docs docs-build`.

## Git and review habits

Keep the diff reviewable:

- Every changed file should support the PR's stated behavior.
- Avoid unrelated cleanup, formatter churn, generated files, redundant examples, and speculative
  options.
- Remove obsolete alternatives consistently rather than leaving multiple half-supported paths.
- Run targeted checks before requesting review and report what was actually run.
- Resolve a thread after implementing its request or answering it with evidence.
- If a suggestion appears wrong, explain the invariant, user need, or test result. Both reviewers
  have revised initial suggestions when given concrete context.
- Student work should stay in the student fork and must not target the official course repository's
  main branch unless explicitly requested.

Recent commit subjects use short, imperative sentence case without Conventional Commit prefixes:

```text
Add live MuJoCo viewer for deployment
Fix deploy runtime errors
Decouple onboard estimator updates onto a background thread
```

Before merging:

- run Ruff check and format validation;
- run targeted tests, then the full suite when dependencies and time permit;
- build docs when public APIs or documentation change;
- evaluate simulation changes locally;
- validate hardware-affecting behavior on real drones when feasible.

## Repository boundaries

Infer conventions from maintained, tracked source. Do not edit or review these as first-party code
unless the task explicitly concerns their setup:

- `acados/`: externally cloned and built by `tools/setup_acados.sh`;
- `ros_ws/`: generated ROS workspace and external packages;
- `c_generated_code/`: generated Acados output;
- `.pixi/`, `build/`, caches, bytecode, and documentation build output.

Preserve binary checkpoints unless a task explicitly replaces the model. Avoid committing logs,
plots, saves, generated JSON or CSV, and secrets covered by `.gitignore`.

## Known inconsistencies: do not copy blindly

- Python support signals differ: package metadata says `>=3.10`, Ruff targets 3.11, and Pixi permits
  3.11 through 3.13. Do not introduce a 3.11-only runtime dependency without deciding the supported
  range explicitly.
- Existing boundary checks mix `assert` and explicit exceptions. New code should use assertions for
  programmer invariants and explicit exceptions for expected user, configuration, file, or runtime
  errors.
- Some maintained files contain inline imports. Reviewer feedback prefers module-scope imports;
  defer only when an actual optional-dependency or import-cycle constraint requires it.
- A few tests omit their directory's marker. New tests should use the intended marker.
- Older docs contain reStructuredText despite the current MkDocs/Google setup.
- Interactive code has some `print` calls, but maintained runtime code generally uses logging.
- Public versus private mutable controller attributes vary. Prefer a leading underscore for new
  internal state.
- Shell files are not consistently formatted or strict-mode protected; do not infer Python style
  from them.
- The repository has no configured mypy, coverage threshold, pre-commit, tox, or nox gate. Do not
  claim these checks are required without adding and documenting them.

## Representative upstream review evidence

These links show the strongest recurring preferences in their code context:

- Fail-fast required data: [`amacati` on direct required configuration access in
  #122](https://github.com/learnsyslab/lsy_drone_racing/pull/122#discussion_r3620286207) and
  [removing invariant-hiding clipping in
  #121](https://github.com/learnsyslab/lsy_drone_racing/pull/121#discussion_r3608544731).
- Exceptions and cleanup: [`amacati` on early failure and
  `try/finally`](https://github.com/learnsyslab/lsy_drone_racing/pull/56#discussion_r2600438266),
  [the bare-catch standard](https://github.com/learnsyslab/lsy_drone_racing/pull/74#discussion_r3044996135),
  and [`ratheron` on preserving the useful root
  error](https://github.com/learnsyslab/lsy_drone_racing/pull/65#discussion_r2639357821).
- Assertions versus boundary errors: [`amacati` on `python
  -O`](https://github.com/learnsyslab/lsy_drone_racing/pull/56#discussion_r2600547950) and
  [an accepted internal shape
  assertion](https://github.com/learnsyslab/lsy_drone_racing/pull/69#discussion_r2776422236).
- Explicit APIs: [`amacati` on passing arrays rather than nested
  configs](https://github.com/learnsyslab/lsy_drone_racing/pull/56#discussion_r2600513801) and
  [visible constructor arguments](https://github.com/learnsyslab/lsy_drone_racing/pull/74#discussion_r3045119765).
- Reuse: [`ratheron` on extending existing environment
  data](https://github.com/learnsyslab/lsy_drone_racing/pull/74#discussion_r3032463051),
  [reusing the core environment](https://github.com/learnsyslab/lsy_drone_racing/pull/74#discussion_r3035591433),
  and [`amacati` on keeping shared validation in
  `RaceCore`](https://github.com/learnsyslab/lsy_drone_racing/pull/69#discussion_r2776422236).
- Simplicity with clarity: [`ratheron` asking to keep the timing change
  simpler](https://github.com/learnsyslab/lsy_drone_racing/pull/119#pullrequestreview-4693360136),
  [`amacati` rejecting a trivial
  helper](https://github.com/learnsyslab/lsy_drone_racing/pull/74#discussion_r3045287884), and
  [the explicit clarity-over-brevity
  qualification](https://github.com/learnsyslab/lsy_drone_racing/pull/66#discussion_r2711952638).
- Naming: [`amacati` on avoiding an ambiguous
  abbreviation](https://github.com/learnsyslab/lsy_drone_racing/pull/56#discussion_r2600358353),
  [precise gate-progress
  vocabulary](https://github.com/learnsyslab/lsy_drone_racing/pull/121#discussion_r3608498503), and
  [`ratheron` on consistent status names across
  layers](https://github.com/learnsyslab/lsy_drone_racing/pull/74#discussion_r3032386809).
- Tests: [`amacati` requiring an old-fails/new-passes regression
  test](https://github.com/learnsyslab/lsy_drone_racing/pull/66#pullrequestreview-3630730638),
  [`ratheron` requiring reverse and outside-gate
  cases](https://github.com/learnsyslab/lsy_drone_racing/pull/76#discussion_r3052499343), and
  [real-hardware evidence before
  merge](https://github.com/learnsyslab/lsy_drone_racing/pull/118#issuecomment-5003044981).
- Documentation versus narration: [`ratheron` requiring a wrapper's real
  scope](https://github.com/learnsyslab/lsy_drone_racing/pull/74#discussion_r3032342390) and
  [removing an implementation-narrating
  comment](https://github.com/learnsyslab/lsy_drone_racing/pull/118#discussion_r3545209750).
- Scope and safety: [`ratheron` questioning an unrelated
  change](https://github.com/learnsyslab/lsy_drone_racing/pull/74#discussion_r3032348302) and
  [requiring emergency stop before updater
  shutdown](https://github.com/learnsyslab/lsy_drone_racing/pull/118#discussion_r3578217019).

Two examples show why full-thread context matters:

- [The request to remove a reshape was
  retracted](https://github.com/learnsyslab/lsy_drone_racing/pull/70#discussion_r2800388551) after
  its canonicalization role was explained.
- [Launch-position update logic was
  accepted](https://github.com/learnsyslab/lsy_drone_racing/pull/122#discussion_r3630231818) after
  its student use case was explained.

## Local evidence map

- `pyproject.toml:57-58,63-153,155-209`
- `.github/workflows/ruff.yml`
- `.github/workflows/testing.yml`
- `.github/workflows/docs.yml`
- `.gitattributes` and `.gitignore`
- `lsy_drone_racing/control/controller.py`
- `lsy_drone_racing/envs/race_core.py`
- `lsy_drone_racing/envs/randomize.py`
- `lsy_drone_racing/envs/real_race_env.py`
- `lsy_drone_racing/utils/crazyflie.py`
- `scripts/sim.py`, `scripts/deploy.py`, and `scripts/multi_deploy.py`
- `tests/unit/`, `tests/integration/`, and `tests/conftest.py`
- `properdocs.yml` and `docs/getting_started/`
- recent first-parent history and focused author history through `d37621f`
