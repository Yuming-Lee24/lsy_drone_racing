"""环境调试脚本

用于验证环境创建、观测空间、奖励函数是否正确工作。

运行方式:
    python test_env.py
    python test_env.py --render True --num_steps 200
"""

from __future__ import annotations

import numpy as np
import torch
import fire
from pathlib import Path

# 环境相关
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.utils import load_config
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch

# 自定义 Wrapper
from lsy_drone_racing.rl_training.wrappers.observation import RacingObservationWrapper
from lsy_drone_racing.rl_training.wrappers.reward import RacingRewardWrapper


def test_base_env(config_path: str, num_envs: int = 4):
    """测试基础环境 (无 Wrapper)"""
    print("=" * 60)
    print("1. 测试基础环境 VecDroneRaceEnv")
    print("=" * 60)
    
    config = load_config(config_path)
    n_gates = len(config.env.track.gates)
    n_obstacles = len(config.env.track.get("obstacles", []))
    print(f"   配置文件: {config_path}")
    print(f"   门数量: {n_gates}, 障碍物数量: {n_obstacles}")
    
    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        disturbances=config.env.get("disturbances", None),
        randomizations=config.env.get("randomizations", None),
        seed=42,
        max_episode_steps=1500,
        device="cpu",
    )
    
    print(f"   ✓ 环境创建成功")
    print(f"   num_envs: {env.num_envs}")
    print(f"   action_space: {env.single_action_space}")
    print(f"   observation_space keys: {list(env.single_observation_space.spaces.keys())}")
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"\n   Reset 后观测:")
    for k, v in obs.items():
        print(f"     {k}: shape={v.shape}, dtype={v.dtype}")
    
    # Step with random action
    action = np.zeros((num_envs, 4))  # 悬停动作
    action[:, 0] = 0.3  # 一点推力
    obs, reward, term, trunc, info = env.step(action)
    print(f"\n   Step 后:")
    print(f"     reward: {reward}")
    print(f"     terminated: {term}")
    print(f"     target_gate: {obs['target_gate']}")
    
    env.close()
    print(f"   ✓ 基础环境测试通过\n")
    return True


def test_reward_wrapper(config_path: str, num_envs: int = 4):
    """测试奖励 Wrapper"""
    print("=" * 60)
    print("2. 测试 RacingRewardWrapper")
    print("=" * 60)
    
    config = load_config(config_path)
    n_gates = len(config.env.track.gates)
    
    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        disturbances=config.env.get("disturbances", None),
        randomizations=config.env.get("randomizations", None),
        seed=42,
        max_episode_steps=1500,
        device="cpu",
    )
    
    env = NormalizeActions(env)
    env = RacingRewardWrapper(
        env,
        n_gates=n_gates,
        stage=1,
        coef_progress=1.0,
        coef_gate=10.0,
        coef_align=0.5,
        coef_collision=5.0,
        coef_smooth=0.1,
        coef_spin=0.02,
    )
    
    print(f"   ✓ RacingRewardWrapper 创建成功")
    
    obs, info = env.reset(seed=42)
    print(f"   观测类型: {type(obs)}")
    print(f"   观测 keys: {list(obs.keys())}")
    
    # 测试多步，观察奖励变化
    print(f"\n   测试 10 步奖励:")
    rewards = []
    for i in range(10):
        action = np.zeros((num_envs, 4))
        action[:, 0] = 0.3  # 推力
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward[0])
        
    print(f"     奖励序列 (env 0): {[f'{r:.3f}' for r in rewards]}")
    print(f"     奖励范围: [{min(rewards):.3f}, {max(rewards):.3f}]")
    
    env.close()
    print(f"   ✓ 奖励 Wrapper 测试通过\n")
    return True


def test_observation_wrapper(config_path: str, num_envs: int = 4):
    """测试观测 Wrapper"""
    print("=" * 60)
    print("3. 测试 RacingObservationWrapper")
    print("=" * 60)
    
    config = load_config(config_path)
    n_gates = len(config.env.track.gates)
    n_obstacles = len(config.env.track.get("obstacles", []))
    
    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        disturbances=config.env.get("disturbances", None),
        randomizations=config.env.get("randomizations", None),
        seed=42,
        max_episode_steps=1500,
        device="cpu",
    )
    
    env = NormalizeActions(env)
    env = RacingRewardWrapper(env, n_gates=n_gates, stage=1)
    env = RacingObservationWrapper(env, n_gates=n_gates, n_obstacles=n_obstacles, stage=1)
    
    print(f"   ✓ RacingObservationWrapper 创建成功")
    print(f"   observation_space: {env.single_observation_space}")
    
    obs, info = env.reset(seed=42)
    print(f"\n   观测向量:")
    print(f"     类型: {type(obs)}")
    print(f"     形状: {obs.shape}")
    print(f"     dtype: {obs.dtype}")
    
    # 打印各部分
    print(f"\n   观测分解 (env 0):")
    print(f"     [0:3]   pos:        {obs[0, 0:3]}")
    print(f"     [3:6]   vel_body:   {obs[0, 3:6]}")
    print(f"     [6:9]   ang_vel:    {obs[0, 6:9]}")
    print(f"     [9:18]  rot_mat:    {obs[0, 9:18]}")
    print(f"     [18:30] gate1:      {obs[0, 18:30]}")
    print(f"     [30:42] gate2:      {obs[0, 30:42]}")
    print(f"     [42:46] prev_act:   {obs[0, 42:46]}")
    print(f"     [46:58] obstacles:  {obs[0, 46:58]}")
    
    # 验证 Stage 1 masking (障碍物应该是 10.0)
    if n_obstacles == 0:
        print(f"\n   Stage 1 Masking 验证 (无障碍物配置):")
        expected = np.allclose(obs[0, 46:58], 10.0)
        print(f"     障碍物位置全为 10.0: {expected}")
    
    env.close()
    print(f"   ✓ 观测 Wrapper 测试通过\n")
    return True


def test_full_pipeline(config_path: str, num_envs: int = 4, render: bool = False, num_steps: int = 50):
    """测试完整的 Wrapper 链 (包括 JaxToTorch)"""
    print("=" * 60)
    print("4. 测试完整 Pipeline (含 JaxToTorch)")
    print("=" * 60)
    
    config = load_config(config_path)
    n_gates = len(config.env.track.gates)
    n_obstacles = len(config.env.track.get("obstacles", []))
    
    device = torch.device("cpu")
    
    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        disturbances=config.env.get("disturbances", None),
        randomizations=config.env.get("randomizations", None),
        seed=42,
        max_episode_steps=1500,
        device="cpu",
    )
    
    env = NormalizeActions(env)
    env = RacingRewardWrapper(env, n_gates=n_gates, stage=1)
    env = RacingObservationWrapper(env, n_gates=n_gates, n_obstacles=n_obstacles, stage=1)
    env = JaxToTorch(env, device)
    
    print(f"   ✓ 完整 Pipeline 创建成功")
    
    obs, info = env.reset(seed=42)
    print(f"\n   PyTorch 输出验证:")
    print(f"     obs 类型: {type(obs)}")
    print(f"     obs 形状: {obs.shape}")
    print(f"     obs device: {obs.device}")
    
    # 运行多步
    print(f"\n   运行 {num_steps} 步模拟...")
    total_rewards = torch.zeros(num_envs)
    episode_lengths = torch.zeros(num_envs)
    gates_passed = torch.zeros(num_envs)
    
    for step in range(num_steps):
        # 随机动作 (网络输出范围 [-1, 1])
        action = torch.randn(num_envs, 4) * 0.3
        action[:, 0] = torch.abs(action[:, 0])  # 推力为正
        
        obs, reward, term, trunc, info = env.step(action)
        total_rewards += reward
        episode_lengths += 1
        
        if render:
            env.unwrapped.render()
        
        # 检查是否有 episode 结束
        done = term | trunc
        if done.any():
            for i in range(num_envs):
                if done[i]:
                    print(f"     Env {i} 结束: 奖励={total_rewards[i]:.2f}, 步数={int(episode_lengths[i])}")
                    total_rewards[i] = 0
                    episode_lengths[i] = 0
    
    print(f"\n   最终统计 (未结束的 episodes):")
    print(f"     累计奖励: {total_rewards.numpy()}")
    print(f"     当前步数: {episode_lengths.numpy()}")
    
    env.close()
    print(f"   ✓ 完整 Pipeline 测试通过\n")
    return True


def test_action_space(config_path: str):
    """测试动作空间和 NormalizeActions"""
    print("=" * 60)
    print("5. 测试动作空间")
    print("=" * 60)
    
    config = load_config(config_path)
    
    env = VecDroneRaceEnv(
        num_envs=2,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        seed=42,
        max_episode_steps=100,
        device="cpu",
    )
    
    print(f"   原始动作空间:")
    print(f"     low:  {env.single_action_space.low}")
    print(f"     high: {env.single_action_space.high}")
    
    env = NormalizeActions(env)
    print(f"\n   归一化后动作空间:")
    print(f"     low:  {env.single_action_space.low}")
    print(f"     high: {env.single_action_space.high}")
    
    env.close()
    print(f"   ✓ 动作空间测试通过\n")
    return True


def main(
    config_file: str = "level3_stage1.toml",
    render: bool = False,
    num_steps: int = 50,
    num_envs: int = 4,
):
    """运行所有测试。
    
    Args:
        config_file: 配置文件名 (在 config/ 目录下)
        render: 是否渲染
        num_steps: 完整测试运行步数
        num_envs: 并行环境数
    """
    config_path = Path(__file__).parents[2] / "config" / config_file
    
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        print(f"请先将 level3_stage1.toml 复制到 config/ 目录")
        return
    
    print(f"\n{'='*60}")
    print(f"环境调试测试")
    print(f"配置文件: {config_file}")
    print(f"{'='*60}\n")
    
    try:
        test_base_env(config_path, num_envs)
        test_reward_wrapper(config_path, num_envs)
        test_observation_wrapper(config_path, num_envs)
        test_action_space(config_path)
        test_full_pipeline(config_path, num_envs, render, num_steps)
        
        print("=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    fire.Fire(main)