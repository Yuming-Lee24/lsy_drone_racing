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
from lsy_drone_racing.rl_training.wrappers.reward import RacingRewardWrapper


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
    config_file: str = "level2.toml"
    """环境配置文件 (相对于 config/ 目录)"""
    num_envs: int = 1024
    """并行环境数量"""
    # 门和障碍物数量从配置文件自动读取，无需手动指定
    
    # ---------- PPO 超参数 ----------
    total_timesteps: int = 2_000_000
    """总训练步数"""
    learning_rate: float = 3e-4
    """学习率"""
    num_steps: int = 32
    """每次 rollout 的步数"""
    gamma: float = 0.99
    """折扣因子"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    num_minibatches: int = 8
    """minibatch 数量"""
    update_epochs: int = 10
    """每次更新的 epoch 数"""
    clip_coef: float = 0.2
    """PPO clip 系数"""
    clip_vloss: bool = True
    """是否 clip value loss"""
    ent_coef: float = 0.01
    """熵系数"""
    vf_coef: float = 0.5
    """value function 系数"""
    max_grad_norm: float = 0.5
    """梯度裁剪"""
    target_kl: float = None
    """目标 KL 散度 (可选的早停)"""
    anneal_lr: bool = True
    """是否学习率退火"""
    norm_adv: bool = True
    """是否归一化 advantage"""
    
    # ---------- 网络结构 ----------
    hidden_dim: int = 256
    """隐藏层维度"""
    
    # ---------- 奖励系数 (可通过 WandB Sweep 调整) ----------
    coef_progress: float = 1.0
    """距离进度奖励系数"""
    coef_gate: float = 10.0
    """过门奖励"""
    coef_align: float = 0.5
    """速度对齐奖励系数"""
    coef_collision: float = 5.0
    """碰撞惩罚系数"""
    coef_smooth: float = 0.1
    """动作平滑惩罚系数"""
    coef_spin: float = 0.02
    """角速度惩罚系数"""
    
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
    env = RacingRewardWrapper(
        env,
        n_gates=n_gates,
        stage=1,  # Stage 1: 无障碍物
        coef_progress=args.coef_progress,
        coef_gate=args.coef_gate,
        coef_align=args.coef_align,
        coef_collision=args.coef_collision,
        coef_smooth=args.coef_smooth,
        coef_spin=args.coef_spin,
    )
    
    # 3. 包装观测 (将字典转为 58D 向量)
    env = RacingObservationWrapper(
        env,
        n_gates=n_gates,
        n_obstacles=n_obstacles,
        stage=2,  # Stage 1: 屏蔽障碍物
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
        
        # 动作标准差 (可学习参数)
        # 根据动作维度自适应初始化
        if action_dim == 4:
            # attitude 模式: [thrust, roll, pitch, yaw]
            init_logstd = torch.tensor([[-1.0, -1.0, -1.0, 0.0]])
        else:
            # state 模式或其他: 统一初始化
            init_logstd = torch.zeros(1, action_dim)
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
    
    # ========== 开始训练 ==========
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = next_obs.to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # 统计
    sum_rewards = torch.zeros(args.num_envs).to(device)
    sum_rewards_hist = []
    episode_count = 0
    
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
            print(f"Iter {iteration}/{args.num_iterations} | "
                  f"Steps: {global_step:,} | "
                  f"Episodes: {episode_count} | "
                  f"Avg Reward (last 100): {avg_reward:.2f} | "
                  f"Time: {iter_time:.2f}s")
    
    # ========== 保存模型 ==========
    train_time = time.time() - train_start_time
    print(f"\nTraining completed in {train_time:.2f}s ({global_step:,} steps)")
    
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
    """评估训练好的模型。
    
    Args:
        args: 配置
        n_eval: 评估 episode 数
        model_path: 模型路径
        render: 是否渲染
        
    Returns:
        episode_rewards: 各 episode 的总奖励
        episode_lengths: 各 episode 的长度
    """
    set_seeds(args.seed)
    device = torch.device("cpu")
    
    # 创建单环境
    args_eval = Args.create(**{**vars(args), "num_envs": 1})
    eval_env = make_env(args_eval, jax_device="cpu", torch_device=device)
    
    obs_dim = eval_env.single_observation_space.shape[0]
    action_dim = eval_env.single_action_space.shape[0]
    
    # 加载模型
    agent = Agent(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    episode_rewards = []
    episode_lengths = []
    
    with torch.no_grad():
        for episode in range(n_eval):
            obs, _ = eval_env.reset(seed=args.seed + episode)
            obs = obs.to(device)
            
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                action, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                obs = obs.to(device)
                
                if render:
                    eval_env.unwrapped.render()
                
                episode_reward += reward[0].item()
                steps += 1
                done = terminated[0].item() or truncated[0].item()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")
    
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f}")
    
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
    config_file: str = "level2.toml",
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
    kwargs["config_file"] = config_file
    args = Args.create(**kwargs)
    
    # 路径设置
    model_path = Path(__file__).parent / "ppo_racing.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    jax_device = args.jax_device
    
    # 训练
    if train:
        train_ppo(args, model_path, device, jax_device, wandb_enabled)
    
    # 评估
    if eval > 0:
        episode_rewards, episode_lengths = evaluate_ppo(args, eval, model_path)
        
        if wandb_enabled and wandb.run is not None:
            wandb.log({
                "eval/mean_reward": np.mean(episode_rewards),
                "eval/std_reward": np.std(episode_rewards),
                "eval/mean_length": np.mean(episode_lengths),
            })
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)