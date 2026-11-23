from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest
from crazyflow.dynamics import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.brake_rl import (
    N_OBS,
    BrakeRL,
    brake_obs,
    brake_obs_jax,
    is_stopped,
    quat2mat,
)


def _random_states(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "pos": rng.uniform(-2, 2, (n, 3)),
        "quat": R.random(n, random_state=seed).as_quat(),
        "vel": rng.uniform(-6, 6, (n, 3)),
        "ang_vel": rng.uniform(-3, 3, (n, 3)),
    }


@pytest.mark.unit
def test_quat2mat_matches_scipy():
    q = R.random(16, random_state=1).as_quat()
    np.testing.assert_allclose(quat2mat(q), R.from_quat(q).as_matrix(), atol=1e-12)


@pytest.mark.unit
def test_obs_numpy_jax_parity():
    s = _random_states(32)
    a1, a2 = np.full((32, 3), 0.1), np.full((32, 3), -0.2)
    p0, d_max = np.ones((32, 3)), np.full(32, 0.7)
    o_np = brake_obs(s["pos"], s["quat"], s["vel"], s["ang_vel"], p0, d_max, a1, a2)
    states = (jp.asarray(s[k]) for k in ("pos", "quat", "vel", "ang_vel"))
    o_jax = brake_obs_jax(*states, p0, d_max, a1, a2)
    assert o_np.shape == (32, N_OBS)
    np.testing.assert_allclose(o_np, np.asarray(o_jax), atol=1e-5)


@pytest.mark.unit
def test_obs_yaw_invariant():
    s = _random_states(8)
    a, p0, d_max = np.zeros((8, 3)), np.zeros((8, 3)), np.zeros(8)
    rot = R.from_euler("z", 1.234)
    o1 = brake_obs(s["pos"], s["quat"], s["vel"], s["ang_vel"], p0, d_max, a, a)
    q2 = (rot * R.from_quat(s["quat"])).as_quat()
    pos2 = rot.apply(s["pos"])
    o2 = brake_obs(pos2, q2, rot.apply(s["vel"]), s["ang_vel"], p0, d_max, a, a)
    np.testing.assert_allclose(o1, o2, atol=1e-9)


@pytest.mark.unit
def test_is_stopped():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    assert is_stopped(q, np.zeros(3), np.zeros(3))
    assert not is_stopped(q, np.array([0.2, 0.0, 0.0]), np.zeros(3))
    assert not is_stopped(R.from_euler("x", 10, degrees=True).as_quat(), np.zeros(3), np.zeros(3))


@pytest.mark.unit
def test_checkpoint_matches_training_actor(tmp_path: Path):
    from lsy_drone_racing.control.train_brake_rl import actor_mean, init_params, save_params

    params = init_params(jax.random.key(0), 128)
    params["actor"][-1] = (params["actor"][-1][0] * 100, params["actor"][-1][1])  # Non-trivial out
    save_params(params, tmp_path / "a.npz")
    brake = BrakeRL(load_params("first_principles", "cf21B_500"), tmp_path / "a.npz")
    s = {k: v[0] for k, v in _random_states(1).items()}
    brake.reset(s)
    x = brake_obs(
        s["pos"], s["quat"], s["vel"], s["ang_vel"], s["pos"], 0.0, brake.act1, brake.act2
    )
    rpy, thrust = brake.compute_control(s)
    expected = np.asarray(actor_mean(params, jp.asarray(x, jp.float32)[None])[0])
    np.testing.assert_allclose(np.asarray(brake.act1), expected, atol=1e-5)
    assert brake.thrust_min <= thrust <= brake.thrust_max
