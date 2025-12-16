"""PPO 训练脚本 - 无人机竞速

基于 CleanRL PPO 实现，使用自定义的观测和奖励 Wrapper。

使用方法:
    # 训练
    python ppo_racing.py --train True --wandb_enabled True
    
    # 评估
    python ppo_racing.py --train False --eval 5
    
    # WandB Sweep
    wandb sweep sweep.yaml
    wandb agent <sweep_id>
    
    # 从 WandB 下载的 config.yaml 加载最佳参数
    python ppo_racing.py --load_config_from ./config.yaml --wandb_enabled True
    
    # 加载 config 并覆盖部分参数
    python ppo_racing.py --load_config_from ./config.yaml --total_timesteps 5000000
"""

from __future__ import annotations

import random
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch import Tensor
from torch.distributions.normal import Normal

# 环境相关
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.utils import load_config
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch

# 自定义 Wrapper
from lsy_drone_racing.rl_training.wrappers.observation import RacingObservationWrapper
from lsy_drone_racing.rl_training.wrappers.reward import RacingRewardWrapper as BaseRewardWrapper
from lsy_drone_racing.rl_training.wrappers.reward_racing_lv0 import RacingRewardWrapper as RacingRewardWrapperLv0


# ============================================================================
# 配置参数
# ============================================================================

@dataclass
class Args:
    """训练配置参数。
    
    支持通过命令行或 WandB Sweep 覆盖。
    """
    
    # ---------- 基础设置 ----------
    seed: int = 42
    """随机种子"""
    torch_deterministic: bool = True
    """是否使用确定性 CUDA 操作"""
    cuda: bool = True
    """是否使用 CUDA"""
    jax_device: str = "gpu"
    """JAX 环境设备 (cpu/gpu)"""
    
    # ---------- WandB 设置 ----------
    wandb_project_name: str = "DroneRacing-PPO"
    """WandB 项目名"""
    wandb_entity: str = None
    """WandB 团队/用户名"""
    
# ---------- 环境配置 ----------
    config_file: str = "level0_no_obst.toml" # 确保是用无障碍的配置
    
    # [关键修改 1] 并行环境数
    # 稍微降低环境数，把内存留给更长的 num_steps
    num_envs: int = 64  
    
    # ---------- PPO 超参数 (竞速调优版) ----------
    
    # [关键修改 2] 训练总量
    # 竞速需要精细打磨轨迹，通常需要较多步数
    total_timesteps: int = 5_000_000  
    
    # [关键修改 3] 学习率
    # 3e-4 是黄金标准。开启退火(anneal_lr)非常重要，
    # 可以在训练后期让无人机动作更细腻，不再抖动。
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    
    # [关键修改 4] 视野长度 (Rollout Length)
    # 128 steps @ 50Hz = 2.56秒。
    # 这让 GAE (优势函数) 能更准确地评估当前动作对未来的影响。
    num_steps: int = 128  
    
    # 批次计算
    # Batch Size = 64 * 128 = 8192
    # Minibatch Size = 8192 / 4 = 2048
    num_minibatches: int = 4
    update_epochs: int = 10
    
    # [关键修改 5] 熵系数 (Entropy Coef)
    # 竞速任务（尤其是过拟合）需要确定性策略。
    # 0.01 适合初期探索。如果你发现后期收敛不够快，可以改小到 0.001 或 0.0
    ent_coef: float = 0.01  
    
    # 其他标准参数 (保持默认即可)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    norm_adv: bool = True
    
    # 网络结构
    hidden_dim: int = 256
    """隐藏层维度"""
    
# ---------- 奖励系数 (适配 RacingRewardWrapperLv0) ----------
    coef_progress: float = 20.0   # [修改] 差分奖励需要较大的系数
    coef_gate: float = 10.0       # [保持]
    coef_finish: float = 50.0     # [新增] 完赛大奖
    coef_time: float = 0.05       # [新增] 时间惩罚
    coef_align: float = 0.5       # [保持]
    coef_collision: float = 10.0  # [修改] 稍微加大碰撞惩罚
    coef_smooth: float = 0.1      # [保持]
    coef_spin: float = 0.1        # [修改] 稍微加大防震荡
    
    n_history: int = 2
    """状态堆叠数量"""
    
    # ---------- 运行时计算 ----------
    batch_size: int = 0
    """批大小 (运行时计算)"""
    minibatch_size: int = 0
    """minibatch 大小 (运行时计算)"""
    num_iterations: int = 0
    """迭代次数 (运行时计算)"""
    
    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """创建并初始化 Args 实例。"""
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        return args


# ============================================================================
# 工具函数
# ============================================================================

def set_seeds(seed: int):
    """设置所有随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """正交初始化网络层。"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ============================================================================
# 环境创建
# ============================================================================

def make_env(
    args: Args,
    jax_device: str = "cpu",
    torch_device: torch.device = torch.device("cpu"),
) -> gym.vector.VectorEnv:
    """创建训练环境。
    
    Wrapper 链:
        VecDroneRaceEnv
        → NormalizeActions (动作归一化 [-1,1] → 实际范围)
        → RacingRewardWrapper (计算 dense reward)
        → RacingObservationWrapper (观测变换: 字典 → 58D 向量)
        → JaxToTorch (JAX Array → PyTorch Tensor)
    
    Args:
        args: 训练配置
        jax_device: JAX 设备
        torch_device: PyTorch 设备
        
    Returns:
        包装后的向量化环境
    """
    # 加载配置文件
    config_path = Path(__file__).parents[2] / "config" / args.config_file
    config = load_config(config_path)
    
    # 从配置文件自动读取门和障碍物数量
    n_gates = len(config.env.track.gates)
    n_obstacles = len(config.env.track.get("obstacles", []))
    print(f"[make_env] 配置: {args.config_file}, 门数: {n_gates}, 障碍物数: {n_obstacles}")
    
    # 创建基础环境
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
    
    # 1. 动作归一化: 将网络输出 [-1, 1] 映射到实际动作范围
    env = NormalizeActions(env)
    
    # 2. 包装奖励 (需要原始 obs 字典)
    env = RacingRewardWrapperLv0(  # 确保类名和你 import 的一致
        env,
        n_gates=n_gates,           # 显式传入 n_gates
        # stage=1,                 # 竞速模式下 stage 通常不再通过参数控制逻辑，可以注释掉或留着占位
        
        # 传递所有系数
        coef_progress=args.coef_progress,
        coef_gate=args.coef_gate,
        coef_finish=args.coef_finish,     # <--- 之前缺失的
        coef_time=args.coef_time,         # <--- 之前缺失的 (虽然你snippet里有，但确保 args 里有定义)
        coef_align=args.coef_align,
        coef_collision=args.coef_collision,
        coef_smooth=args.coef_smooth,
        coef_spin=args.coef_spin,
    )
    

    # 3. 包装观测 (将字典转为向量)
    env = RacingObservationWrapper(
        env,
        n_gates=n_gates,
        n_obstacles=n_obstacles,
        stage=1,  # Stage 1: 屏蔽障碍物
        n_history=args.n_history,  # 新增 状态堆叠数量
    )
    
    # 4. 数据类型转换: JAX Array → PyTorch Tensor
    env = JaxToTorch(env, torch_device)
    
    return env


# ============================================================================
# 神经网络
# ============================================================================

class Agent(nn.Module):
    """PPO Agent 网络。
    
    Actor-Critic 结构:
    - Actor: 输出动作均值和标准差
    - Critic: 输出状态价值
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        """初始化网络。
        
        Args:
            obs_dim: 观测维度 (58)
            action_dim: 动作维度 (4)
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # Critic 网络
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        
        # Actor 网络 (输出动作均值)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )
        
        # ============================================================
        # [关键修改] 推力偏置 (Thrust Bias) 初始化
        # ============================================================
        # 我们要修改 actor_mean 的最后一层 Linear 层的 bias
        # 结构是: [Linear, Tanh, Linear, Tanh, Linear(索引-2), Tanh(索引-1)]
        with torch.no_grad():
            last_layer = self.actor_mean[-2] # 获取最后一个 Linear 层
            
            # 1. 确保姿态通道 (0,1,2) 偏置为 0 (水平)
            # 这里的 0 对应 NormalizeActions 映射后的中间值
            last_layer.bias[0] = 0.0  # Roll
            last_layer.bias[1] = 0.0  # Pitch
            last_layer.bias[2] = 0.0  # Yaw
            
            # 2. 给推力通道 (3) 加偏置
            last_layer.bias[3] = 1.0

        # 动作标准差 (可学习参数)
        # 根据动作维度自适应初始化
        if action_dim == 4:
            # attitude 模式: [roll, pitch, yaw,thrust]
            init_logstd = torch.tensor([[-1.0, -1.0, -1.0, -0.5]])
        self.actor_logstd = nn.Parameter(init_logstd)
    
    def get_value(self, x: Tensor) -> Tensor:
        """获取状态价值。"""
        return self.critic(x)
    
    def get_action_and_value(
        self, 
        x: Tensor, 
        action: Tensor | None = None, 
        deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """获取动作和价值。
        
        Args:
            x: 观测
            action: 已有动作 (用于计算 log_prob)
            deterministic: 是否使用确定性动作
            
        Returns:
            action: 动作
            log_prob: 动作的 log 概率
            entropy: 策略熵
            value: 状态价值
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = action_mean if deterministic else probs.sample()
        
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


# ============================================================================
# 训练函数
# ============================================================================

def train_ppo(
    args: Args,
    model_path: Path,
    device: torch.device,
    jax_device: str,
    wandb_enabled: bool = False,
) -> list[float]:
    """PPO 训练主循环。
    
    基于 CleanRL 实现: https://docs.cleanrl.dev/
    
    Args:
        args: 训练配置
        model_path: 模型保存路径
        device: PyTorch 设备
        jax_device: JAX 设备
        wandb_enabled: 是否启用 WandB
        
    Returns:
        训练过程中的 episode reward 历史
    """
    # ========== 初始化 ==========
    if wandb_enabled and wandb.run is None:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
    
    train_start_time = time.time()
    set_seeds(args.seed)
    print(f"Training on device: {device} | Environment device: {jax_device}")
    print(f"Config: {args.config_file}")

        # ========== 保存配置 ==========
    config_save_path = model_path.parent.parent / "train_args" / "train_args.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Config saved to {config_save_path}")

    # ========== 创建环境 ==========
    envs = make_env(args, jax_device=jax_device, torch_device=device)
    
    obs_dim = envs.single_observation_space.shape[0]  # 58
    action_dim = envs.single_action_space.shape[0]    # 4
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # ========== 创建 Agent ==========
    agent = Agent(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # ========== 存储缓冲区 ==========
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # =================初始化保存变量 =================
    best_reward = -float('inf')
    best_model_path = model_path.parent / "best_model.ckpt"
    print(f"训练开始。按 Ctrl+C 可安全停止并保存到: {model_path}")
    print(f"best model saved under: {best_model_path}")
    # ===========================================================
    
    # ========== 开始训练 ==========
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = next_obs.to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # 统计
    sum_rewards = torch.zeros(args.num_envs).to(device)
    sum_rewards_hist = []
    len_hist = []
    episode_count = 0
    
    try:
        for iteration in range(1, args.num_iterations + 1):
            iter_start_time = time.time()
            
            # 学习率退火
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            
            # ========== 收集数据 ==========
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                
                # 采样动作
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                
                # 执行动作 (JaxToTorch wrapper 会处理 tensor 转换)
                next_obs, reward, terminations, truncations, infos = envs.step(action)
                next_obs = next_obs.to(device)
                reward = reward.to(device)
                
                rewards[step] = reward
                sum_rewards += reward
                
                # 处理 episode 结束
                next_done = (terminations | truncations).float().to(device)
                
                if next_done.any():
                    finished_rewards = sum_rewards[next_done.bool()]
                    for r in finished_rewards:
                        sum_rewards_hist.append(r.item())
                        episode_count += 1
                        if wandb_enabled:
                            wandb.log({"train/episode_reward": r.item()}, step=global_step)
                    sum_rewards[next_done.bool()] = 0
            
            # ========== 计算 GAE ==========
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                
                returns = advantages + values
            
            # ========== 展平数据 ==========
            b_obs = obs.reshape((-1, obs_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, action_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            
            # ========== PPO 更新 ==========
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                # 早停
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            
            # ========== 日志记录 ==========
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            iter_time = time.time() - iter_start_time
            
            if wandb_enabled:
                wandb.log({
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/SPS": int(args.num_envs * args.num_steps / iter_time),
                    "charts/episode_count": episode_count,
                    "losses/policy_loss": pg_loss.item(),
                    "losses/value_loss": v_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                }, step=global_step)
            

            # 打印进度
            if iteration % 10 == 0 or iteration == 1:
                avg_reward = np.mean(sum_rewards_hist[-100:]) if sum_rewards_hist else 0
                # 计算平均长度
            
                print(f"Iter {iteration}/{args.num_iterations} | Steps: {global_step/1e6:.2f}M/{args.total_timesteps/1e6:.2f}M")
                
                # [修改 C] 打印更详细的调试信息
                # 1. 任务表现: 奖励和存活时长
                print(f"  [Perf] Avg Rew: {avg_reward:.2f}")
                
                # 2. 网络健康度: 
                #    Val (Critic误差): 越低越好
                #    Ent (策略熵): 代表探索欲望。如果迅速掉到 0 说明过早收敛(学傻了)
                #    KL  (更新幅度): 应该在 0.01 左右。太大说明学习率太高，太小说明学不动
                print(f"  [Loss] Val: {v_loss.item():.4f} | Pol: {pg_loss.item():.4f} | "
                    f"Ent: {entropy_loss.item():.4f} | KL: {approx_kl.item():.4f}")
                
                print(f"  [Time] {iter_time:.2f}s | SPS: {int(args.num_envs * args.num_steps / iter_time)}")
                print("-" * 50)
                # ================= [插入点 3] 自动保存逻辑 =================
                # 1. 每次打印日志都保存一下最新模型 (覆盖)
                if model_path is not None:
                    torch.save(agent.state_dict(), model_path)
                
                # 2. 如果奖励创新高，额外存一份 "best_model"
                if avg_reward > best_reward and iteration > 10:  # 前10次不稳定，不存
                    best_reward = avg_reward
                    torch.save(agent.state_dict(), best_model_path)
                    print(f"  [★] 新纪录！最佳模型已保存 (Rew: {best_reward:.2f})")
                # =========================================================
    # ========== [新增] 捕获中断 ==========
    except KeyboardInterrupt:
        print("\n\n[警报] 用户手动停止训练 (Ctrl+C)！")
        if model_path is not None:
            torch.save(agent.state_dict(), model_path)
            print(f"  [安全] 模型已紧急保存到: {model_path}")
        print("正在关闭环境...")
        envs.close()
        return sum_rewards_hist
    # ===================================
    # ========== 保存模型 ==========
    train_time = time.time() - train_start_time
    # 计算分和秒
    m, s = divmod(int(train_time), 60)
    
    # 打印格式: "Training completed in 28m 52s ..."
    print(f"\nTraining completed in {m}m {s}s ({global_step:,} steps)")
    
    if model_path is not None:
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    envs.close()
    return sum_rewards_hist


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_ppo(
    args: Args,
    n_eval: int,
    model_path: Path,
    render: bool = True,
) -> tuple[list[float], list[int]]:
    """调试版评估函数：打印动作和状态详情。"""
    set_seeds(args.seed)
    # 强制使用 CPU 进行评估，避免 tensor device 错误
    device = torch.device("cpu")
    
    print(f"\n[Eval] 加载模型: {model_path}")
    print(f"[Eval] 渲染模式: {render}")

    # 1. 创建单环境 (num_envs=1)
    # 注意：必须确保这里使用的 config 和训练时一致
    args_eval = Args.create(**{**vars(args), "num_envs": 1})
    eval_env = make_env(args_eval, jax_device="cpu", torch_device=device)
    
    obs_dim = eval_env.single_observation_space.shape[0]
    action_dim = eval_env.single_action_space.shape[0]
    
    # 2. 加载模型
    agent = Agent(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    try:
        # 尝试加载权重
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # 兼容只保存了 state_dict 或保存了整个 dict 的情况
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            agent.load_state_dict(checkpoint["model_state_dict"])
        else:
            agent.load_state_dict(checkpoint)
        print("[Eval] 模型加载成功！")
    except Exception as e:
        print(f"[Eval] 模型加载失败: {e}")
        return [], []

    agent.eval()
    
    episode_rewards = []
    episode_lengths = []
    
    with torch.no_grad():
        for episode in range(n_eval):
            print(f"\n=== Episode {episode + 1} Start ===")
            obs, _ = eval_env.reset(seed=args.seed + episode)
            obs = obs.to(device)
            
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                action, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
                # ====================================================
                # [DEBUG] 打印 eval 环境下的网络原始输出
                # ====================================================
                # if steps == 0: # 只打印第一步
                #     raw_act_eval = action[0].cpu().numpy()
                #     print(f"\n[VS] Eval Raw Output: {raw_act_eval}")
                    # print(f"[VS] Eval Obs Sum  : {obs.sum().item():.5f}") # 双重确认输入校验和
                
                # 确保截断
                action = torch.clamp(action, -1.0, 1.0)

                # 执行动作
                obs, reward, terminated, truncated, info = eval_env.step(action)
                obs = obs.to(device)
                
                if render:
                    eval_env.unwrapped.render()
                    # 稍微睡一下，不然画面太快看不清
                    time.sleep(0.02) 
                
                episode_reward += reward[0].item()
                steps += 1
                
                # 检查是否撞毁
                term = terminated[0].item()
                trunc = truncated[0].item()
                done = term or trunc
                
                if done:
                    print(f"!!! Episode End at Step {steps} !!!")
                    print(f"Reason: Terminated={term}, Truncated={trunc}")
                    # 检查 info 里有没有碰撞信息
                    if "final_info" in info:
                        final_info = info["final_info"][0]
                        if final_info:
                            print(f"Final Info: {final_info}")
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")
    
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f}")
    eval_env.close()
    return episode_rewards, episode_lengths

# ============================================================================
# 主函数
# ============================================================================

def load_wandb_config(config_path: str | Path) -> dict:
    """从 WandB 下载的 config.yaml 加载参数。
    
    WandB config.yaml 格式可能是:
        learning_rate:
          value: 0.001
    或者:
        learning_rate: 0.001
    
    Args:
        config_path: config.yaml 文件路径
        
    Returns:
        扁平化的参数字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 扁平化: 如果值是 dict 且有 'value' 键，提取它
    flat_config = {}
    
    # 需要跳过的 WandB 内部字段
    skip_keys = {'wandb_version', '_wandb', 'wandb_enabled', 'train', 'eval'}
    
    for key, val in config.items():
        if key in skip_keys:
            continue
        if isinstance(val, dict) and 'value' in val:
            flat_config[key] = val['value']
        else:
            flat_config[key] = val
    
    print(f"[load_wandb_config] 从 {config_path} 加载参数:")
    for k, v in flat_config.items():
        print(f"  {k}: {v}")
    
    return flat_config


def main(
    wandb_enabled: bool = False,
    train: bool = True,
    eval: int = 0,
    config_file: str = None,
    load_config_from: str = None,
    **kwargs,
):
    """主入口。
    
    Args:
        wandb_enabled: 是否启用 WandB
        train: 是否训练
        eval: 评估的 episode 数 (0 表示不评估)
        config_file: 环境配置文件
        load_config_from: 从 WandB config.yaml 加载参数 (可选)
        **kwargs: 覆盖默认 Args 的参数
    """
    # 如果指定了 wandb config 文件，先加载它
    if load_config_from is not None:
        wandb_config = load_wandb_config(load_config_from)
        # wandb config 优先级低于命令行参数
        for key, val in wandb_config.items():
            if key not in kwargs:
                kwargs[key] = val
    
    # 创建配置
    # kwargs["config_file"] = config_file
    args = Args.create(**kwargs)
    
    # 路径设置
    model_save_path = Path(__file__).parent /"checkpoints" / "ppo_racing.ckpt"
    model_eval_path = Path(__file__).parent /"checkpoints" / "best_model.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    jax_device = args.jax_device
    
    # 训练
    if train:
        train_ppo(args, model_save_path, device, jax_device, wandb_enabled)
    
    # 评估
    if eval > 0:
        episode_rewards, episode_lengths = evaluate_ppo(args, eval, model_eval_path)
        
        if wandb_enabled and wandb.run is not None:
            wandb.log({
                "eval/mean_reward": np.mean(episode_rewards),
                "eval/std_reward": np.std(episode_rewards),
                "eval/mean_length": np.mean(episode_lengths),
            })
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)