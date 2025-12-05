"""RL-based Racing Controller.

使用 PPO 训练的策略进行无人机竞速控制。
观测空间与训练时的 RacingObservationWrapper 一致 (58D)。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# Agent 网络定义 (与训练时一致)
# ============================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(torch.nn.Module):
    """PPO Agent 网络."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.critic = torch.nn.Sequential(
            layer_init(torch.nn.Linear(obs_dim, hidden_dim)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(hidden_dim, hidden_dim)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(hidden_dim, 1), std=1.0),
        )
        
        self.actor_mean = torch.nn.Sequential(
            layer_init(torch.nn.Linear(obs_dim, hidden_dim)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(hidden_dim, hidden_dim)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(hidden_dim, action_dim), std=0.01),
            torch.nn.Tanh(),
        )
        
        if action_dim == 4:
            init_logstd = torch.tensor([[-1.0, -1.0, -1.0, 0.0]])
        else:
            init_logstd = torch.zeros(1, action_dim)
        self.actor_logstd = torch.nn.Parameter(init_logstd)
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = action_mean if deterministic else probs.sample()
        
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


# ============================================================================
# 控制器
# ============================================================================

class RLRacingController(Controller):
    """使用 RL 策略的竞速控制器."""
    
    # 门的局部坐标系下 4 个角点偏移
    GATE_CORNERS_LOCAL = np.array([
        [0.0, -0.2,  0.2],  # 左上
        [0.0,  0.2,  0.2],  # 右上
        [0.0,  0.2, -0.2],  # 右下
        [0.0, -0.2, -0.2],  # 左下
    ], dtype=np.float32)
    
    def __init__(self, obs: dict[str, NDArray], info: dict, config: dict):
        """初始化控制器.
        
        Args:
            obs: 初始观测
            info: 环境信息
            config: 环境配置
        """
        super().__init__(obs, info, config)
        
        self.freq = config.env.freq
        self.n_gates = len(config.env.track.gates)
        self.n_obstacles = len(config.env.track.get("obstacles", []))
        
        # 动作范围 (attitude 模式)
        self.action_low = np.array([-1.0, -np.pi, -np.pi, -np.pi], dtype=np.float32)
        self.action_high = np.array([1.0, np.pi, np.pi, np.pi], dtype=np.float32)
        
        # 状态
        self.prev_action = np.zeros(4, dtype=np.float32)
        self._tick = 0
        self._finished = False
        
        # 加载 RL 策略
        self.device = torch.device("cpu")
        self.obs_dim = 58
        self.action_dim = 4
        
        self.agent = Agent(self.obs_dim, self.action_dim, hidden_dim=128).to(self.device)
        
        # 模型路径 (与训练脚本输出一致)
        model_path = Path(__file__).parent.parent / "rl_training" / "ppo_racing.ckpt"
        if model_path.exists():
            self.agent.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            print(f"[RLRacingController] 加载模型: {model_path}")
        else:
            print(f"[RLRacingController] 警告: 模型不存在 {model_path}，使用随机策略")
        
        self.agent.eval()
    
    def compute_control(
        self, 
        obs: dict[str, NDArray], 
        info: dict | None = None
    ) -> NDArray:
        """计算控制指令.
        
        Args:
            obs: 当前观测
            info: 额外信息
            
        Returns:
            [thrust, roll, pitch, yaw] 控制指令
        """
        # 检查是否完赛
        if obs["target_gate"] == -1:
            self._finished = True
            # 返回悬停指令
            return np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # 构建 58D 观测向量
        obs_vector = self._build_observation(obs)
        
        # 转换为 tensor
        obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(obs_tensor, deterministic=True)
            action = action.squeeze(0).cpu().numpy()
        
        # 缩放动作 [-1, 1] -> 实际范围
        action = self._scale_action(action)
        
        # 更新状态
        self.prev_action = action.copy()
        
        return action.astype(np.float32)
    
    def _build_observation(self, obs: dict[str, NDArray]) -> NDArray:
        """构建 58D 观测向量 (与 RacingObservationWrapper 一致).
        
        结构:
            [0:3]   - pos (世界坐标)
            [3:6]   - vel_body (机体坐标)
            [6:9]   - ang_vel
            [9:18]  - rot_matrix (展平)
            [18:30] - gate1_corners (当前门, 机体坐标)
            [30:42] - gate2_corners (下一门, 机体坐标)
            [42:46] - prev_action
            [46:58] - obstacles (机体坐标)
        """
        # 基础状态
        pos = obs["pos"]  # (3,)
        vel = obs["vel"]  # (3,)
        ang_vel = obs["ang_vel"]  # (3,)
        quat = obs["quat"]  # (4,)
        
        # 旋转矩阵
        rot = Rotation.from_quat(quat)
        rot_matrix = rot.as_matrix().flatten()  # (9,)
        
        # 速度转到机体坐标系
        vel_body = rot.inv().apply(vel)  # (3,)
        
        # 门角点 (机体坐标)
        target_gate = int(obs["target_gate"])
        gates_pos = obs["gates_pos"]  # (n_gates, 3)
        gates_quat = obs["gates_quat"]  # (n_gates, 4)
        
        gate1_corners = self._compute_gate_corners_body(
            target_gate, gates_pos, gates_quat, pos, rot
        )  # (12,)
        
        gate2_corners = self._compute_gate_corners_body(
            target_gate + 1, gates_pos, gates_quat, pos, rot
        )  # (12,)
        
        # 障碍物 (机体坐标)
        obstacles_body = self._compute_obstacles_body(
            obs.get("obstacles_pos", np.zeros((0, 3))), pos, rot
        )  # (12,)
        
        # 拼接
        obs_vector = np.concatenate([
            pos,                # 3
            vel_body,           # 3
            ang_vel,            # 3
            rot_matrix,         # 9
            gate1_corners,      # 12
            gate2_corners,      # 12
            self.prev_action,   # 4
            obstacles_body,     # 12
        ])  # Total: 58
        
        return obs_vector.astype(np.float32)
    
    def _compute_gate_corners_body(
        self,
        gate_idx: int,
        gates_pos: NDArray,
        gates_quat: NDArray,
        drone_pos: NDArray,
        drone_rot: Rotation,
    ) -> NDArray:
        """计算门的 4 个角点在机体坐标系下的位置."""
        # 边界检查
        if gate_idx < 0 or gate_idx >= len(gates_pos):
            return np.zeros(12, dtype=np.float32)
        
        gate_pos = gates_pos[gate_idx]  # (3,)
        gate_quat = gates_quat[gate_idx]  # (4,)
        
        # 门的旋转
        gate_rot = Rotation.from_quat(gate_quat)
        
        # 角点世界坐标
        corners_world = gate_pos + gate_rot.apply(self.GATE_CORNERS_LOCAL)  # (4, 3)
        
        # 转到机体坐标系
        rel_world = corners_world - drone_pos  # (4, 3)
        corners_body = drone_rot.inv().apply(rel_world)  # (4, 3)
        
        return corners_body.flatten().astype(np.float32)  # (12,)
    
    def _compute_obstacles_body(
        self,
        obstacles_pos: NDArray,
        drone_pos: NDArray,
        drone_rot: Rotation,
    ) -> NDArray:
        """计算障碍物在机体坐标系下的位置."""
        result = np.full(12, 10.0, dtype=np.float32)  # 默认很远
        
        if obstacles_pos.size == 0:
            return result
        
        # 相对位置
        rel_world = obstacles_pos - drone_pos  # (n_obs, 3)
        rel_body = drone_rot.inv().apply(rel_world)  # (n_obs, 3)
        
        # 按距离排序，取最近 4 个
        dists = np.linalg.norm(rel_body, axis=1)
        sorted_idx = np.argsort(dists)[:4]
        
        nearest = rel_body[sorted_idx]  # (<=4, 3)
        result[:nearest.size] = nearest.flatten()
        
        return result
    
    def _scale_action(self, action: NDArray) -> NDArray:
        """缩放动作从 [-1, 1] 到实际范围."""
        scale = (self.action_high - self.action_low) / 2.0
        mean = (self.action_high + self.action_low) / 2.0
        return np.clip(action, -1.0, 1.0) * scale + mean
    
    def step_callback(
        self,
        action: NDArray,
        obs: dict[str, NDArray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """步进回调."""
        self._tick += 1
        return self._finished
    
    def episode_callback(self):
        """Episode 结束回调."""
        self._tick = 0
        self._finished = False
        self.prev_action = np.zeros(4, dtype=np.float32)