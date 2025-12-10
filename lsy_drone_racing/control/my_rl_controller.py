"""RL-based Racing Controller.

使用 PPO 训练的策略进行无人机竞速控制。
观测空间与训练时的 RacingObservationWrapper 一致 (84D)。
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
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 512):
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
        [0.0, -0.2,  0.2],
        [0.0,  0.2,  0.2],
        [0.0,  0.2, -0.2],
        [0.0, -0.2, -0.2],
    ], dtype=np.float32)
    
    # 历史状态维度
    HISTORY_STATE_DIM = 13  # pos(3) + quat(4) + vel(3) + ang_vel(3)
    
    def __init__(self, obs: dict[str, NDArray], info: dict, config: dict):
        """初始化控制器."""
        super().__init__(obs, info, config)
        
        self.freq = config.env.freq
        self.n_gates = len(config.env.track.gates)
        self.n_obstacles = len(config.env.track.get("obstacles", []))
        
        # 历史帧数 (与训练时一致)
        self.n_history = 2
        
        # 动作范围 (attitude 模式)
        self.action_low = np.array([-1.0, -np.pi, -np.pi, -np.pi], dtype=np.float32)
        self.action_high = np.array([1.0, np.pi, np.pi, np.pi], dtype=np.float32)
        
        # 状态
        self.prev_action = np.zeros(4, dtype=np.float32)
        self._tick = 0
        self._finished = False
        
        # 历史状态缓冲区 (n_history, 13)
        self._history_buffer = np.zeros(
            (self.n_history, self.HISTORY_STATE_DIM), 
            dtype=np.float32
        )
        # 用初始状态填充历史缓冲区
        init_state = self._extract_basic_state(obs)
        for i in range(self.n_history):
            self._history_buffer[i] = init_state
        
        # 加载 RL 策略
        self.device = torch.device("cpu")
        self.obs_dim = 58 + self.n_history * self.HISTORY_STATE_DIM  # 84
        self.action_dim = 4
        
        self.agent = Agent(self.obs_dim, self.action_dim, hidden_dim=512).to(self.device)
        
        # 模型路径
        model_path = Path(__file__).parent.parent / "rl_training" / "ppo_racing.ckpt"
        if model_path.exists():
            self.agent.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            print(f"[RLRacingController] 加载模型: {model_path}")
        else:
            print(f"[RLRacingController] 警告: 模型不存在 {model_path}，使用随机策略")
        
        self.agent.eval()
    
    def _extract_basic_state(self, obs: dict) -> NDArray:
        """提取基础状态 (pos, quat, vel, ang_vel)。
        
        Returns:
            (13,) 基础状态向量
        """
        pos = np.array(obs["pos"]).flatten()[:3]
        quat = np.array(obs["quat"]).flatten()[:4]
        vel = np.array(obs["vel"]).flatten()[:3]
        ang_vel = np.array(obs["ang_vel"]).flatten()[:3]
        
        return np.concatenate([pos, quat, vel, ang_vel]).astype(np.float32)
    
    def _update_history_buffer(self, obs: dict):
        """更新历史状态缓冲区。"""
        current_state = self._extract_basic_state(obs)
        
        # 滚动：丢弃最旧的，添加最新的
        self._history_buffer = np.concatenate([
            self._history_buffer[1:],
            current_state[np.newaxis, :]
        ], axis=0)
    
    def compute_control(
        self, 
        obs: dict[str, NDArray], 
        info: dict | None = None
    ) -> NDArray:
        """计算控制指令."""
        # 检查是否完赛
        if obs["target_gate"] == -1:
            self._finished = True
            return np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # 构建 84D 观测向量
        obs_vector = self._build_observation(obs)
        
        # 转换为 tensor
        obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(obs_tensor, deterministic=True)
            action = action.squeeze(0).cpu().numpy()
        
        # 强制 yaw 为 0
        action[2] = 0.0
        
        # 缩放动作
        action = self._scale_action(action)
        
        # 更新状态
        self.prev_action = action.copy()
        self._update_history_buffer(obs)
        
        return action.astype(np.float32)
    
    def _build_observation(self, obs: dict[str, NDArray]) -> NDArray:
        """构建 84D 观测向量。
        
        结构:
            [0:3]   - pos (世界坐标)
            [3:6]   - vel_body (机体坐标)
            [6:9]   - ang_vel
            [9:18]  - rot_matrix (展平)
            [18:30] - gate1_corners
            [30:42] - gate2_corners
            [42:46] - prev_action
            [46:58] - obstacles
            [58:84] - history (2 * 13 = 26)
        """
        # 基础状态
        pos = obs["pos"]
        vel = obs["vel"]
        ang_vel = obs["ang_vel"]
        quat = obs["quat"]
        
        # 旋转矩阵
        rot = Rotation.from_quat(quat)
        rot_matrix = rot.as_matrix().flatten()
        
        # 速度转到机体坐标系
        vel_body = rot.inv().apply(vel)
        
        # 门角点
        target_gate = int(obs["target_gate"])
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
        
        gate1_corners = self._compute_gate_corners_body(
            target_gate, gates_pos, gates_quat, pos, rot
        )
        gate2_corners = self._compute_gate_corners_body(
            target_gate + 1, gates_pos, gates_quat, pos, rot
        )
        
        # 障碍物
        obstacles_body = self._compute_obstacles_body(
            obs.get("obstacles_pos", np.zeros((0, 3))), pos, rot
        )
        
        # 历史状态 (展平)
        history_flat = self._history_buffer.flatten()  # (26,)
        
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
            history_flat,       # 26
        ])  # Total: 84
        
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
        if gate_idx < 0 or gate_idx >= len(gates_pos):
            return np.zeros(12, dtype=np.float32)
        
        gate_pos = gates_pos[gate_idx]
        gate_quat = gates_quat[gate_idx]
        gate_rot = Rotation.from_quat(gate_quat)
        
        corners_world = gate_pos + gate_rot.apply(self.GATE_CORNERS_LOCAL)
        rel_world = corners_world - drone_pos
        corners_body = drone_rot.inv().apply(rel_world)
        
        return corners_body.flatten().astype(np.float32)
    
    def _compute_obstacles_body(
        self,
        obstacles_pos: NDArray,
        drone_pos: NDArray,
        drone_rot: Rotation,
    ) -> NDArray:
        """计算障碍物在机体坐标系下的位置."""
        result = np.full(12, 10.0, dtype=np.float32)
        
        if obstacles_pos.size == 0:
            return result
        
        rel_world = obstacles_pos - drone_pos
        rel_body = drone_rot.inv().apply(rel_world)
        
        dists = np.linalg.norm(rel_body, axis=1)
        sorted_idx = np.argsort(dists)[:4]
        
        nearest = rel_body[sorted_idx]
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
        self._history_buffer = np.zeros(
            (self.n_history, self.HISTORY_STATE_DIM), 
            dtype=np.float32
        )