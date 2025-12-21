# test_wrapped_env_collision.py
"""测试包装后环境的碰撞检测功能"""

import torch
import numpy as np
import jax.numpy as jp
import time
from pathlib import Path
from dataclasses import dataclass

# 你的环境创建函数需要的依赖
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.utils import load_config
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from lsy_drone_racing.rl_training.wrappers.observation import RacingObservationWrapper
from lsy_drone_racing.rl_training.wrappers.reward_racing_lv0 import RacingRewardWrapper as RacingRewardWrapperLv0


@dataclass
class Args:
    """模拟训练配置"""
    config_file: str = "level0.toml"
    num_envs: int = 1  # 单环境便于观察
    seed: int = 42
    n_history: int = 2
    
    # 奖励系数
    coef_progress: float = 1.0
    coef_gate: float = 10.0
    coef_finish: float = 100.0
    coef_time: float = -0.01
    coef_align: float = 0.1
    coef_collision: float = -10.0
    coef_smooth: float = 0.1
    coef_spin: float = 0.1


def make_env(
    args: Args,
    jax_device: str = "cpu",
    torch_device: torch.device = torch.device("cpu"),
):
    """你的环境创建函数（复制过来）"""
    config_path = Path(__file__).parents[2] / "config" / args.config_file
    config = load_config(config_path)
    
    # 🔥 强制启用渲染
    config.sim.render = True
    
    n_gates = len(config.env.track.gates)
    n_obstacles = len(config.env.track.get("obstacles", []))
    print(f"[make_env] 配置: {args.config_file}, 门数: {n_gates}, 障碍物数: {n_obstacles}")
    
    env = VecDroneRaceEnv(
        num_envs=args.num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        disturbances=config.env.get("disturbances", None),
        randomizations=config.env.get("randomizations", None),
        seed=args.seed,
        max_episode_steps=1500,
        device=jax_device,
    )
    
    env = NormalizeActions(env)
    env = RacingRewardWrapperLv0(
        env,
        n_gates=n_gates,
        coef_progress=args.coef_progress,
        coef_gate=args.coef_gate,
        coef_finish=args.coef_finish,
        coef_time=args.coef_time,
        coef_align=args.coef_align,
        coef_collision=args.coef_collision,
        coef_smooth=args.coef_smooth,
        coef_spin=args.coef_spin,
    )
    env = RacingObservationWrapper(
        env,
        n_gates=n_gates,
        n_obstacles=n_obstacles,
        stage=1,
        n_history=args.n_history,
    )
    env = JaxToTorch(env, torch_device)
    
    return env


def get_base_env(wrapped_env):
    """递归获取底层环境"""
    env = wrapped_env
    while hasattr(env, 'env'):
        env = env.env
    return env


def test_collision_detection():
    """测试碰撞检测"""
    print("="*60)
    print("测试包装后环境的碰撞检测")
    print("="*60)
    
    # 创建环境
    args = Args()
    env = make_env(args)
    
    # 获取底层环境（用于访问原始状态）
    base_env = get_base_env(env)
    
    # 重置环境
    obs, info = env.reset()
    print(f"\n包装后观测维度: {obs.shape}")
    print(f"包装后观测类型: {type(obs)}")
    
    # 获取门的位置（需要从底层环境获取）
    raw_obs = base_env.obs()
    gate_pos = raw_obs['gates_pos'][0, 0]  # (num_envs, n_gates, 3)
    drone_pos = raw_obs['pos'][0]
    
    print(f"\n初始状态:")
    print(f"  无人机位置: {drone_pos}")
    print(f"  第一个门位置: {gate_pos}")
    print(f"  控制模式: {base_env.sim.control}")
    
    # 方案1: 让无人机飞向门框（自然碰撞）
    print("\n开始飞行，目标：撞击门框右侧...")
    
    target_pos = gate_pos.copy()
    target_pos[1] += 0.25  # 偏移到门框外侧
    
    collision_detected = False
    collision_reward_sum = 0.0
    
    for step in range(300):
        # 构造动作（注意：动作已经被归一化到[-1,1]）
        if base_env.sim.control == "attitude":
            # [roll, pitch, yaw, thrust] 已归一化
            action = torch.tensor([[0.0, 0.3, 0.0, 0.0]], dtype=torch.float32)
        else:
            # state control 的动作空间
            action = torch.zeros(1, 13, dtype=torch.float32)
            # 目标位置（需要归一化？取决于你的NormalizeActions实现）
            # 这里假设已经处理好了，直接设置
            action[0, :3] = torch.tensor(target_pos)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 渲染
        if step % 2 == 0:  # 降低渲染频率
            base_env.render()
            time.sleep(0.02)
        
        # 获取底层状态
        raw_obs = base_env.obs()
        drone_pos = raw_obs['pos'][0]
        disabled = base_env.data.disabled_drones[0, 0]
        
        # 获取接触信息
        contacts = base_env.sim.contacts()
        contact_count = int(jp.sum(contacts))
        
        # 每20步打印一次状态
        if step % 20 == 0:
            dist_to_gate = np.linalg.norm(np.array(drone_pos) - np.array(gate_pos))
            print(f"\n步骤 {step}:")
            print(f"  位置: {drone_pos}")
            print(f"  到门距离: {dist_to_gate:.3f}m")
            print(f"  接触数: {contact_count}")
            print(f"  奖励: {reward[0].item():.3f}")
            print(f"  disabled: {disabled}")
        
        # 检测碰撞（接触数 > 4 说明除了地面还有其他接触）
        if contact_count > 4 and not collision_detected:
            collision_detected = True
            print(f"\n🚨 步骤 {step}: 检测到碰撞！")
            print(f"  接触数: {contact_count}")
            print(f"  奖励: {reward[0].item():.3f}")
            print(f"  disabled: {disabled}")
            print(f"  terminated: {terminated[0].item()}")
            
            # 显示具体接触信息
            contact_impl = base_env.sim.mjx_data._impl.contact
            active_contacts = jp.where(contacts[0])[0]
            
            print(f"\n  碰撞详情:")
            for idx in active_contacts[:10]:
                idx = int(idx)
                geom1 = int(contact_impl.geom1[0, idx])
                geom2 = int(contact_impl.geom2[0, idx])
                dist = float(contact_impl.dist[0, idx])
                
                try:
                    geom1_name = base_env.sim.mj_model.geom(geom1).name
                    geom2_name = base_env.sim.mj_model.geom(geom2).name
                    if 'ground' not in geom1_name and 'ground' not in geom2_name:
                        print(f"    💥 {geom1_name} <-> {geom2_name}, dist={dist:.4f}")
                except:
                    pass
            
            # 碰撞后继续渲染观察
            for _ in range(30):
                base_env.render()
                time.sleep(0.033)
        
        if contact_count > 4:
            collision_reward_sum += reward[0].item()
        
        if terminated[0] or truncated[0]:
            print(f"\n✅ 回合结束于步骤 {step}")
            print(f"  terminated: {terminated[0].item()}")
            print(f"  truncated: {truncated[0].item()}")
            print(f"  最终奖励: {reward[0].item():.3f}")
            
            if collision_detected:
                print(f"  碰撞期间累计奖励: {collision_reward_sum:.3f}")
            
            # 保持渲染
            for _ in range(60):
                base_env.render()
                time.sleep(0.033)
            break
    
    # 测试结果总结
    print("\n" + "="*60)
    print("测试总结:")
    print("="*60)
    print(f"碰撞检测: {'✅ 正常' if collision_detected else '❌ 未检测到碰撞'}")
    print(f"环境终止: {'✅ 正常' if (terminated[0] or truncated[0]) else '⚠️  未终止'}")
    if collision_detected:
        print(f"碰撞惩罚: {collision_reward_sum:.3f} (应该 < 0)")
    
    env.close()


if __name__ == "__main__":
    test_collision_detection()