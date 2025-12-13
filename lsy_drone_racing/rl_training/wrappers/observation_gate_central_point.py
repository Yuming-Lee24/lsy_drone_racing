"""
观测空间设计 (高效版 - 相对中心点 + 法向量):
    - [0:1]   Position Z (世界坐标): 1D
    - [1:4]   Linear Velocity (机体坐标): 3D
    - [4:7]   Angular Velocity: 3D
    - [7:16]  Rotation Matrix (展平): 9D
    - [16:22] Next Gate (机体坐标): 6D (3D相对位置 + 3D相对法向量)
    - [22:28] Next Next Gate (机体坐标): 6D (3D相对位置 + 3D相对法向量)
    - [28:32] Previous Action: 4D
    - [32:44] 4个障碍物位置 (机体坐标): 12D
    - [44:...] 历史状态 (n_history 帧)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorObservationWrapper
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RacingObservationWrapper(VectorObservationWrapper):
    """
    高效观测 Wrapper：使用门中心点和法向量代替角点。
    
    优势:
    1. 输入维度更低 (56 -> 44)，训练收敛更快。
    2. 显式提供"去哪里"(位置)和"怎么对齐"(法向量)，避免神经网络自己去猜几何形状。
    """
    
    # 观测维度常量 (更新版)
    # SELF_DIM (16) + GATES_DIM (6*2=12) + ACTION_DIM (4) + OBSTACLES_DIM (12) = 44
    BASE_OBS_DIM = 44
    SELF_DIM = 16        # pos_z(1) + vel(3) + ang_vel(3) + rot_mat(9)
    GATES_DIM = 12       # 2个门 × (3中心点 + 3法向量)
    ACTION_DIM = 4       # prev_action
    OBSTACLES_DIM = 12   # 4个障碍物 × 3坐标
    HISTORY_STATE_DIM = 16  # 历史状态保持不变
    
    def __init__(
        self, 
        env: VectorEnv,
        n_gates: int = 4,
        n_obstacles: int = 4,
        stage: int = 0,
        n_history: int = 2,
    ):
        super().__init__(env)
        
        self.n_gates = n_gates
        self.n_obstacles = n_obstacles
        self.stage = stage
        self.n_history = n_history
        
        # 计算总观测维度
        self.OBS_DIM = self.BASE_OBS_DIM + self.n_history * self.HISTORY_STATE_DIM
        
        # 内部状态：上一步动作
        self._prev_action = np.zeros((self.num_envs, self.ACTION_DIM), dtype=np.float32)
        
        # 内部状态：历史状态缓冲区
        if self.n_history > 0:
            self._history_buffer = np.zeros(
                (self.num_envs, self.n_history, self.HISTORY_STATE_DIM), 
                dtype=np.float32
            )
        
        # 定义新的观测空间
        self.single_observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.OBS_DIM,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, self.OBS_DIM),
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        """重置环境。"""
        obs, info = self.env.reset(**kwargs)
        
        # 重置内部状态
        self._prev_action = np.zeros((self.num_envs, self.ACTION_DIM), dtype=np.float32)
        
        # 重置历史缓冲区
        if self.n_history > 0:
            init_state = self._extract_basic_state(obs)
            for i in range(self.n_history):
                self._history_buffer[:, i, :] = init_state
        
        transformed_obs = self.observations(obs)
        return transformed_obs, info
    
    def step(self, action):
        """执行一步。"""
        current_obs_dict = self._get_current_obs_dict()
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 更新历史
        if self.n_history > 0 and current_obs_dict is not None:
            self._update_history_buffer(current_obs_dict)
        
        transformed_obs = self.observations(obs)
        
        # 更新 prev_action
        self._prev_action = np.array(action, dtype=np.float32).reshape(self.num_envs, -1)
        self._cached_obs = obs
        
        return transformed_obs, reward, terminated, truncated, info
    
    def _get_current_obs_dict(self):
        return getattr(self, '_cached_obs', None)
    
    def _extract_basic_state(self, obs: dict) -> NDArray:
        """提取基础状态用于历史堆叠。"""
        pos = np.array(obs["pos"])
        quat = np.array(obs["quat"])
        vel = np.array(obs["vel"])
        ang_vel = np.array(obs["ang_vel"])
        
        pos_z = pos[:, 2:3]
        rot_matrices = Rotation.from_quat(quat).as_matrix()
        rot_flat = rot_matrices.reshape(self.num_envs, 9)
        
        return np.concatenate([pos_z, rot_flat, vel, ang_vel], axis=1)
    
    def _update_history_buffer(self, obs: dict):
        """滚动更新历史缓冲区。"""
        current_state = self._extract_basic_state(obs)
        self._history_buffer = np.concatenate([
            self._history_buffer[:, 1:, :],
            current_state[:, np.newaxis, :]
        ], axis=1)
    
    def observations(self, obs: dict) -> NDArray:
        """核心函数：将观测字典转换为高效向量。"""
        num_envs = self.num_envs
        
        # ========== 1. 提取原始数据 ==========
        pos = np.array(obs["pos"])              # (N, 3)
        pos_z = pos[:, 2:3]                     # (N, 1)
        vel = np.array(obs["vel"])              # (N, 3)
        ang_vel = np.array(obs["ang_vel"])      # (N, 3)
        quat = np.array(obs["quat"])            # (N, 4)
        target_gate = np.array(obs["target_gate"])  # (N,)
        gates_pos = np.array(obs["gates_pos"])      # (N, n_gates, 3)
        gates_quat = np.array(obs["gates_quat"])    # (N, n_gates, 4)
        obstacles_pos = np.array(obs["obstacles_pos"]) 
        
        # ========== 2. 自身姿态计算 ==========
        rot_matrices = self._quat_to_rotation_matrix(quat)  # (N, 3, 3)
        rot_matrices_flat = rot_matrices.reshape(num_envs, 9)
        vel_body = self._world_to_body_batch(vel, rot_matrices)  # (N, 3)
        
        # ========== 3. 门信息计算 (核心修改) ==========
        # 获取当前门和下一个门的索引
        gate1_idx = np.clip(target_gate, 0, self.n_gates - 1)
        gate2_idx = np.clip(target_gate + 1, 0, self.n_gates - 1)
        
        # 处理完赛情况 (-1)
        finished_mask = (target_gate == -1)
        gate1_idx = np.where(finished_mask, self.n_gates - 1, gate1_idx)
        gate2_idx = np.where(finished_mask, self.n_gates - 1, gate2_idx)
        
        batch_idx = np.arange(num_envs)
        
        # 提取目标门的世界坐标信息
        g1_pos = gates_pos[batch_idx, gate1_idx]   # (N, 3)
        g1_quat = gates_quat[batch_idx, gate1_idx] # (N, 4)
        g2_pos = gates_pos[batch_idx, gate2_idx]   # (N, 3)
        g2_quat = gates_quat[batch_idx, gate2_idx] # (N, 4)
        
        # 计算 Gate 1 (当前门) 的机体坐标系信息
        # 相对位置 (3D)
        g1_center_body = self._compute_relative_pos_body(g1_pos, pos, rot_matrices)
        # 相对法向量 (3D)
        g1_normal_body = self._compute_gate_normal_body(g1_quat, rot_matrices)
        
        # 计算 Gate 2 (下一个门) 的机体坐标系信息
        g2_center_body = self._compute_relative_pos_body(g2_pos, pos, rot_matrices)
        g2_normal_body = self._compute_gate_normal_body(g2_quat, rot_matrices)
        
        # 拼接: 每个门变成 6维 (3 pos + 3 normal)
        gate1_obs = np.concatenate([g1_center_body, g1_normal_body], axis=1) # (N, 6)
        gate2_obs = np.concatenate([g2_center_body, g2_normal_body], axis=1) # (N, 6)
        
        # ========== 4. 障碍物计算 ==========
        obstacles_body = self._compute_obstacles_body(
            obstacles_pos, pos, rot_matrices
        )
        
        # ========== 5. 组装观测向量 ==========
        obs_parts = [
            pos_z,               # [0:1]   1D
            vel_body,            # [1:4]   3D
            ang_vel,             # [4:7]   3D
            rot_matrices_flat,   # [7:16]  9D
            gate1_obs,           # [16:22] 6D (当前门)
            gate2_obs,           # [22:28] 6D (下个门)
            self._prev_action,   # [28:32] 4D
            obstacles_body,      # [32:44] 12D
        ]
        
        # ========== 6. 添加历史信息 ==========
        if self.n_history > 0:
            history_flat = self._history_buffer.reshape(num_envs, -1)
            obs_parts.append(history_flat)
        
        obs_vector = np.concatenate(obs_parts, axis=1)
        
        # ========== 7. 课程学习 Masking ==========
        obs_vector = self._apply_stage_masking(obs_vector)
        
        return obs_vector.astype(np.float32)
    
    def set_stage(self, stage: int):
        self.stage = stage
        print(f"[RacingObservationWrapper] 切换到 Stage {stage}")
    
    # ========== 辅助函数 ==========
    
    def _quat_to_rotation_matrix(self, quat: NDArray) -> NDArray:
        return Rotation.from_quat(quat).as_matrix()
    
    def _world_to_body_batch(self, vec_world: NDArray, rot_matrices: NDArray) -> NDArray:
        """将世界坐标向量转换到机体坐标 (R^T * v)。"""
        return np.einsum('nij,nj->ni', rot_matrices.transpose(0, 2, 1), vec_world)
    
    def _compute_relative_pos_body(
        self, target_pos: NDArray, self_pos: NDArray, rot_matrices: NDArray
    ) -> NDArray:
        """计算目标在机体坐标系下的相对位置。"""
        # 1. 世界系相对向量
        rel_pos_world = target_pos - self_pos
        # 2. 旋转到机体系
        return self._world_to_body_batch(rel_pos_world, rot_matrices)

    def _compute_gate_normal_body(
        self, gate_quat: NDArray, drone_rot_matrices: NDArray
    ) -> NDArray:
        """计算门的法向量在无人机机体坐标系下的投影。"""
        # 1. 门的旋转矩阵 (Local -> World)
        gate_rot_matrices = Rotation.from_quat(gate_quat).as_matrix()
        
        # 2. 提取世界系法向量
        # 门在局部坐标系朝向 X 轴 [1, 0, 0]，对应矩阵的第一列
        gate_normal_world = gate_rot_matrices[:, :, 0] 
        
        # 3. 转换到无人机机体系
        gate_normal_body = self._world_to_body_batch(gate_normal_world, drone_rot_matrices)
        
        return gate_normal_body
    
    def _compute_obstacles_body(
        self, obstacles_pos: NDArray, drone_pos: NDArray, drone_rot: NDArray
    ) -> NDArray:
        """计算最近障碍物的相对位置 (保持原有鲁棒逻辑)。"""
        num_envs = drone_pos.shape[0]
        if obstacles_pos.size == 0 or obstacles_pos.shape[1] == 0:
            return np.full((num_envs, 12), 10.0, dtype=np.float32)
            
        obs_x = obstacles_pos[:, :, 0]
        obs_y = obstacles_pos[:, :, 1]
        obs_z_top = obstacles_pos[:, :, 2]
        drone_z = drone_pos[:, 2:3]
        
        effective_obs_z = np.minimum(obs_z_top, drone_z)
        effective_obs_pos = np.stack([obs_x, obs_y, effective_obs_z], axis=-1)
        
        rel_world = effective_obs_pos - drone_pos[:, np.newaxis, :]
        drone_rot_inv = drone_rot.transpose(0, 2, 1)
        rel_body = np.einsum('nij,nkj->nki', drone_rot_inv, rel_world)
        
        dists = np.linalg.norm(rel_body, axis=2)
        sorted_idx = np.argsort(dists, axis=1)
        
        n_obs_available = rel_body.shape[1]
        n_keep = min(n_obs_available, 4)
        
        result = np.full((num_envs, 12), 10.0, dtype=np.float32)
        batch_indices = np.arange(num_envs)[:, None]
        keep_indices = sorted_idx[:, :n_keep]
        
        result[:, :n_keep * 3] = rel_body[batch_indices, keep_indices].reshape(num_envs, -1)
        return result
    
    def _apply_stage_masking(self, obs_vector: NDArray) -> NDArray:
        """根据课程阶段 Masking (索引已更新)。"""
        obs_vector = obs_vector.copy()
        
        # 现在的布局:
        # [16:22] gate1 (6D)
        # [22:28] gate2 (6D)
        # [28:32] prev_action
        # [32:44] obstacles
        
        if self.stage == 0:
            # 屏蔽 Gate 2 (用 Gate 1 的信息覆盖它，或者全 0)
            # 这里我们让 Gate 2 看起来和 Gate 1 一样，或者直接给远处的假值
            obs_vector[:, 22:28] = obs_vector[:, 16:22] 
            # 屏蔽障碍物
            obs_vector[:, 32:44] = 10.0
            
        if self.stage == 1:
            # 只屏蔽障碍物
            obs_vector[:, 32:44] = 10.0
            
        return obs_vector