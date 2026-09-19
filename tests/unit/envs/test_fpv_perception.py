"""Geometric FPV sensing, independent of the GUI and controller performance."""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.envs.race_core import (
    EnvData,
    RaceCoreEnv,
    _fpv_camera_mounts,
    _fpv_wireframe,
    _reset_env_data,
    _rotate_vectors,
    _update_visited_objects,
    _visible_objects,
    obs,
)
from tests.unit.envs.test_race_core import make_env

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def env():
    instance = make_env("level2.toml")
    instance.reset(seed=12)
    yield instance.unwrapped
    instance.close()


def identity_data(env: RaceCoreEnv) -> EnvData:
    data = env.data
    states = data.sim_data.states.replace(
        pos=jp.zeros((1, 1, 3)), quat=jp.array([[[0.0, 0.0, 0.0, 1.0]]])
    )
    return data.replace(
        sim_data=data.sim_data.replace(states=states),
        camera_pos=jp.zeros((1, 3)),
        camera_forward=jp.array([[1.0, 0.0, 0.0]]),
        camera_right=jp.array([[0.0, 1.0, 0.0]]),
        camera_up=jp.array([[0.0, 0.0, 1.0]]),
    )


@pytest.mark.parametrize(
    "point,expected",
    [
        ([0.5, 0, 0], True),
        ([0, 0.5, 0], False),
        ([-0.5, 0, 0], False),
        ([0.5, 0, 0.4], False),
        ([0.999, 0, 0], True),
        ([1.0, 0, 0], False),
        ([1.001, 0, 0], False),
        ([0, 0, 0], False),
    ],
)
def test_range_and_direction(env: RaceCoreEnv, point: list[float], expected: bool):
    result = jax.jit(_visible_objects)(identity_data(env), jp.array([[point]]))
    assert bool(result[0, 0, 0]) == expected


@pytest.mark.parametrize("offset,expected", [(-0.01, True), (0.0, False), (0.01, False)])
@pytest.mark.parametrize("vertical", [False, True])
def test_frustum_boundary(env: RaceCoreEnv, offset: float, expected: bool, vertical: bool):
    tangent = env.data.camera_tan_half_fov[0, int(vertical)]
    angle = jp.arctan(tangent) + jp.deg2rad(offset)
    side = 0.5 * jp.sin(angle)
    point = jp.array([0.5 * jp.cos(angle), 0 if vertical else side, side if vertical else 0])
    assert bool(_visible_objects(identity_data(env), point[None, None])[0, 0, 0]) == expected


@pytest.mark.parametrize("angles", [[0, 0, 90], [0, -60, 0], [30, -40, 70]])
def test_mount_and_rotation_match_mujoco(env: RaceCoreEnv, angles: list[float]):
    model = env.sim.mj_model
    mj_data = mujoco.MjData(model)
    camera = model.camera("fpv_cam:0").id
    body = model.cam_bodyid[camera]
    mocap = model.body_mocapid[body]
    quat = R.from_euler("xyz", angles, degrees=True).as_quat()
    position = np.array([0.4, -0.2, 0.8])
    mj_data.mocap_pos[mocap] = position
    mj_data.mocap_quat[mocap] = quat[[3, 0, 1, 2]]
    mujoco.mj_kinematics(model, mj_data)
    mujoco.mj_camlight(model, mj_data)
    origin = position + np.asarray(_rotate_vectors(jp.array(quat), env.data.camera_pos))[0]
    forward = np.asarray(_rotate_vectors(jp.array(quat), env.data.camera_forward))[0]
    np.testing.assert_allclose(origin, mj_data.cam_xpos[camera], atol=1e-6)
    np.testing.assert_allclose(forward, -mj_data.cam_xmat[camera].reshape(3, 3)[:, 2], atol=1e-6)
    for axis, column in [(env.data.camera_right, 0), (env.data.camera_up, 1)]:
        actual = np.asarray(_rotate_vectors(jp.array(quat), axis))[0]
        np.testing.assert_allclose(
            actual, mj_data.cam_xmat[camera].reshape(3, 3)[:, column], atol=1e-6
        )
    states = env.data.sim_data.states.replace(pos=jp.array([[position]]), quat=jp.array([[quat]]))
    data = env.data.replace(sim_data=env.data.sim_data.replace(states=states))
    points = jp.array([[origin + forward * 0.5, origin - forward * 0.5]])
    np.testing.assert_array_equal(_visible_objects(data, points), [[[True, False]]])


def test_memory_observations_and_reset(env: RaceCoreEnv):
    data = identity_data(env)
    gates = jp.tile(jp.array([[[0.5, 0, 0]]]), (1, data.gates_pos.shape[1], 1))
    obstacles = jp.tile(jp.array([[[-0.5, 0, 0]]]), (1, data.obstacles_pos.shape[1], 1))
    data = _reset_env_data(data.replace(gates_pos=gates, obstacles_pos=obstacles))
    assert bool(data.gates_visited.all())
    assert not bool(data.obstacles_visited.any())
    np.testing.assert_allclose(obs(data)["gates_pos"], gates[:, None])
    np.testing.assert_allclose(obs(data)["gates_quat"], data.gates_quat[:, None])
    np.testing.assert_allclose(obs(data)["obstacles_pos"], data.nominal_obstacles_pos[:, None])
    states = data.sim_data.states.replace(quat=jp.array([[[0.0, 0.0, 1.0, 0.0]]]))
    data = _update_visited_objects(data.replace(sim_data=data.sim_data.replace(states=states)))
    assert bool(data.gates_visited.all()) and bool(data.obstacles_visited.all())
    data = _reset_env_data(data)
    assert not bool(data.gates_visited.any())
    assert bool(data.obstacles_visited.all())
    np.testing.assert_allclose(obs(data)["gates_pos"], data.nominal_gates_pos[:, None])


def test_batch_worlds_and_drones(env: RaceCoreEnv):
    data = identity_data(env)
    states = data.sim_data.states.replace(
        pos=jp.array([[[0, 0, 0], [0, 0, 0]], [[2, 0, 0], [2, 0, 0]]], dtype=float),
        quat=jp.array([[[0, 0, 0, 1], [0, 0, 1, 0]]] * 2, dtype=float),
    )
    data = data.replace(sim_data=data.sim_data.replace(states=states))
    result = jax.jit(_visible_objects)(data, jp.array([[[0.5, 0, 0]], [[1.5, 0, 0]]]))
    np.testing.assert_array_equal(result, [[[True], [False]], [[False], [True]]])


def test_legacy_distance_ignores_direction_and_height(env: RaceCoreEnv):
    data = identity_data(env).replace(sensor_use_camera_fov=False, sensor_range=jp.array([0.7]))
    points = jp.array([[[-0.5, 0, 10], [0.7, 0, 0]]])
    np.testing.assert_array_equal(_visible_objects(data, points), [[[True, False]]])


def test_masked_reset_preserves_other_world_memory(env: RaceCoreEnv):
    data = identity_data(env)
    states = data.sim_data.states.replace(
        pos=jp.zeros((2, 2, 3)), quat=jp.array([[[0, 0, 0, 1], [0, 0, 1, 0]]] * 2, dtype=float)
    )
    gates = jp.tile(jp.array([[[0.5, 0, 0]]]), (2, data.gates_pos.shape[1], 1))
    obstacles = jp.tile(jp.array([[[0.5, 0, 0]]]), (2, data.obstacles_pos.shape[1], 1))
    data = data.replace(
        sim_data=data.sim_data.replace(states=states),
        steps=jp.zeros((2,), dtype=int),
        n_gates_passed=jp.zeros((2, 2), dtype=int),
        gates_pos=gates,
        obstacles_pos=obstacles,
        gates_visited=jp.ones((2, 2, gates.shape[1]), dtype=bool),
        obstacles_visited=jp.ones((2, 2, obstacles.shape[1]), dtype=bool),
    )
    data = _reset_env_data(data, jp.array([True, False]))
    np.testing.assert_array_equal(data.gates_visited[:, :, 0], [[True, False], [True, True]])
    np.testing.assert_array_equal(data.obstacles_visited[:, :, 0], [[True, False], [True, True]])


def test_missing_camera_is_explicit():
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(ValueError, match="fpv_cam:0"):
        _fpv_camera_mounts(model, 1)


def test_camera_fov_matches_model(env: RaceCoreEnv):
    camera = env.sim.mj_model.camera("fpv_cam:0").id
    width, height = env.sim.mj_model.cam_resolution[camera]
    vertical = np.tan(np.deg2rad(env.sim.mj_model.cam_fovy[camera] / 2))
    np.testing.assert_allclose(
        env.data.camera_tan_half_fov[0], [vertical * width / height, vertical]
    )


def test_rectangular_corner_and_roll(env: RaceCoreEnv):
    data = identity_data(env)
    horizontal, vertical = np.asarray(data.camera_tan_half_fov[0])
    # Near an image corner: visible in a rectangle, outside an inscribed circular cone.
    corner = np.array([1, 0.95 * horizontal, 0.95 * vertical])
    corner *= 0.8 / np.linalg.norm(corner)
    assert bool(_visible_objects(data, jp.array([[corner]]))[0, 0, 0])
    point = jp.array([[[0.5, 0.5 * (horizontal + vertical) / 2, 0.0]]])
    assert bool(_visible_objects(data, point)[0, 0, 0])
    quat = jp.array([[R.from_euler("x", 90, degrees=True).as_quat()]])
    states = data.sim_data.states.replace(quat=quat)
    rotated = data.replace(sim_data=data.sim_data.replace(states=states))
    assert not bool(_visible_objects(rotated, point)[0, 0, 0])


def test_level2_headless_step(env: RaceCoreEnv):
    assert env.data.sensor_use_camera_fov
    assert float(env.data.sensor_range[0]) == 1
    env.reset(seed=12)
    position = np.asarray(env.data.sim_data.states.pos[0, 0])
    action = np.zeros(13, dtype=np.float32)
    action[:3] = position
    action[2] = 0.5
    for _ in range(3):
        observation, _, _, _, _ = env.step(jp.array(action))
        assert all(np.isfinite(value).all() for value in observation.values())


@pytest.mark.parametrize("radius", [0.7, 1.0])
def test_wireframe_matches_camera_frustum(env: RaceCoreEnv, radius: float):
    data = env.data.replace(sensor_range=jp.array([radius]))
    states = data.sim_data.states.replace(
        pos=jp.array([[[0.2, -0.4, 0.8]]]),
        quat=jp.array([[R.from_euler("xyz", [30, -40, 70], degrees=True).as_quat()]]),
    )
    data = data.replace(sim_data=data.sim_data.replace(states=states))
    origin = np.asarray(states.pos + _rotate_vectors(states.quat, data.camera_pos))[0, 0]
    forward = np.asarray(_rotate_vectors(states.quat, data.camera_forward))[0, 0]
    right = np.asarray(_rotate_vectors(states.quat, data.camera_right))[0, 0]
    up = np.asarray(_rotate_vectors(states.quat, data.camera_up))[0, 0]
    limits = np.asarray(data.camera_tan_half_fov[0])
    lines = _fpv_wireframe(data, 0)
    for i, edge in enumerate(lines[:4]):
        delta = edge - origin
        np.testing.assert_allclose(np.linalg.norm(delta, axis=-1), radius, atol=1e-6)
        axis = up if i % 2 == 0 else right
        limit = limits[1] if i % 2 == 0 else limits[0]
        np.testing.assert_allclose(np.abs(delta @ axis) / (delta @ forward), limit, atol=1e-6)
    for ray in lines[4:8]:
        np.testing.assert_allclose(ray[0], origin, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(ray[1] - origin), radius, atol=1e-6)
    for cap in lines[8:]:
        np.testing.assert_allclose(np.linalg.norm(cap - origin, axis=-1), radius, atol=1e-6)
        np.testing.assert_allclose(cap[len(cap) // 2], origin + forward * radius, atol=1e-6)


def test_wireframe_disabled_without_fov(env: RaceCoreEnv):
    assert _fpv_wireframe(env.data.replace(sensor_use_camera_fov=False), 0) == []
    assert _fpv_wireframe(env.data.replace(sensor_range=jp.array([0.0])), 0) == []
