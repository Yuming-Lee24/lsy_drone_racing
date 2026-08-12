# End-to-end JAX PPO

This package trains the 88-dimensional racing policy directly against Crazyflow's functional JAX
state. The production path does not import Torch, NumPy, SciPy, Gymnasium wrappers, or
`JaxToTorch` inside the rollout.

## Commands

```bash
pixi run -e jax-rl-gpu train-jax-ppo --profile fast
pixi run -e jax-rl-gpu train-jax-ppo --profile fast --disable-early-stopping
pixi run -e jax-rl-gpu train-jax-ppo --profile safe_fast
pixi run -e jax-rl-gpu eval-jax-ppo --checkpoint saves/jax_ppo/fast/best --profile fast
```

Resume an exact training state with:

```bash
pixi run -e jax-rl-gpu train-jax-ppo \
  --profile fast --resume saves/jax_ppo/fast/latest
```

`safe_fast` automatically initializes from `saves/jax_ppo/fast/best` when that policy exists. It
loads only policy parameters and creates a new optimizer, as required for the second stage.

## Performance gate

The local PyTorch reference is isolated from the production environment:

```bash
pixi run -e rl-benchmark-gpu benchmark-torch-ppo \
  --timesteps 10000000 --output saves/jax_ppo/performance/torch_10m.json
pixi run -e jax-rl-gpu benchmark-jax-ppo \
  --timesteps 10000000 \
  --pytorch-report saves/jax_ppo/performance/torch_10m.json \
  --output saves/jax_ppo/performance/jax_10m.json
```

The JAX command exits unsuccessfully when steady end-to-end throughput is below 2x the supplied
PyTorch report. Compilation is measured separately. Rollout and PPO optimization remain separate
XLA executables: a second outer `jit` was measured to inflate memory use and reduce RTX 4070
throughput substantially.

## Checkpoints and controller

Each checkpoint directory contains `checkpoint.msgpack` for exact continuation, `policy.msgpack`
for evaluation/deployment, and `metadata.json`. To run the deterministic controller, set
`controller.file = "attitude_jax_ppo.py"` and either:

- set `controller.checkpoint` in the race config; or
- set `LSY_JAX_PPO_CHECKPOINT` to a checkpoint directory.

All checkpoints, metrics, evaluation results, and performance reports live below the ignored
`saves/jax_ppo/` directory.
