"""观测空间变换 Wrapper

将 VecDroneRaceEnv 的原始观测字典转换为 (58 + 13*n_history)D 向量，用于 RL 训练。

观测空间设计:
    - Position (世界坐标): 3D
    - Linear Velocity (机体坐标): 3D  
    - Angular Velocity: 3D
    - Rotation Matrix (展平): 9D
    - Next Gate 4角点 (机体坐标): 12D
    - Next Next Gate 4角点 (机体坐标): 12D
    - Previous Action: 4D
    - 4个障碍物位置 (机体坐标): 12D
    - 历史状态 (n_history 帧): n_history * 13D
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
    """将原始观测转换为向量的 Wrapper。
    
    核心功能:
    1. 世界坐标 → 机体坐标变换 (门、障碍物、速度)
    2. 门中心点 → 4角点计算
    3. 课程学习的观测 masking
    4. 维护 prev_action 状态
    5. 历史状态堆叠
    """
    
    # 门的局部坐标系下 4 个角点偏移 (门面朝 X 轴，YZ 平面是门框)
    GATE_CORNERS_LOCAL = np.array([
        [0.0, -0.2,  0.2],  # 左上
        [0.0,  0.2,  0.2],  # 右上
        [0.0,  0.2, -0.2],  # 右下
        [0.0, -0.2, -0.2],  # 左下
    ], dtype=np.float32)
    
    # 观测维度常量 (基础部分)
    BASE_OBS_DIM = 58
    SELF_DIM = 18        # pos(3) + vel(3) + ang_vel(3) + rot_mat(9)
    GATES_DIM = 24       # 2个门 × 4角点 × 3坐标
    ACTION_DIM = 4       # prev_action
    OBSTACLES_DIM = 12   # 4个障碍物 × 3坐标
    HISTORY_STATE_DIM = 13  # pos(3) + quat(4) + vel(3) + ang_vel(3)
    
    def __init__(
        self, 
        env: VectorEnv,
        n_gates: int = 4,
        n_obstacles: int = 4,
        stage: int = 0,
        n_history: int = 2,  # 新增：历史帧数
    ):
        """初始化 Wrapper。
        
        Args:
            env: 底层向量化环境 (VecDroneRaceEnv)
            n_gates: 赛道门的数量
            n_obstacles: 障碍物数量
            stage: 课程阶段 (0=屏蔽障碍物和下下个门, 1=屏蔽障碍物, 2=全开)
            n_history: 历史状态帧数 (0 表示不使用历史)
        """
        super().__init__(env)
        
        self.n_gates = n_gates
        self.n_obstacles = n_obstacles
        self.stage = stage
        self.n_history = n_history
        
        # 计算总观测维度
        self.OBS_DIM = self.BASE_OBS_DIM + self.n_history * self.HISTORY_STATE_DIM
        
        # 内部状态：上一步动作
        self._prev_action = np.zeros((self.num_envs, self.ACTION_DIM), dtype=np.float32)
        
        # 内部状态：历史状态缓冲区 (num_envs, n_history, 13)
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
        """重置环境，清零 prev_action 和历史缓冲区。"""
        obs, info = self.env.reset(**kwargs)
        
        # 重置内部状态
        self._prev_action = np.zeros((self.num_envs, self.ACTION_DIM), dtype=np.float32)
        
        # 重置历史缓冲区，用初始状态填充
        if self.n_history > 0:
            init_state = self._extract_basic_state(obs)  # (num_envs, 13)
            for i in range(self.n_history):
                self._history_buffer[:, i, :] = init_state
        
        # 转换观测
        transformed_obs = self.observations(obs)
        
        return transformed_obs, info
    
    def step(self, action):
        """执行一步，更新 prev_action 和历史缓冲区。"""
        # 先获取当前状态用于更新历史
        current_obs_dict = self._get_current_obs_dict()
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 更新历史缓冲区 (在获取新观测之前，用旧状态更新)
        if self.n_history > 0 and current_obs_dict is not None:
            self._update_history_buffer(current_obs_dict)
        
        # 转换观测 (使用当前的 prev_action)
        transformed_obs = self.observations(obs)
        
        # 更新 prev_action 为本次执行的动作 (供下一步使用)
        self._prev_action = np.array(action, dtype=np.float32).reshape(self.num_envs, -1)
        
        # 保存当前观测字典用于下一步更新历史
        self._cached_obs = obs
        
        return transformed_obs, reward, terminated, truncated, info
    
    def _get_current_obs_dict(self):
        """获取缓存的观测字典。"""
        return getattr(self, '_cached_obs', None)
    
    def _extract_basic_state(self, obs: dict) -> NDArray:
        """从观测字典中提取基础状态 (pos, quat, vel, ang_vel)。
        
        Args:
            obs: 观测字典
            
        Returns:
            (num_envs, 13) 基础状态向量
        """
        pos = np.array(obs["pos"])          # (num_envs, 3)
        quat = np.array(obs["quat"])        # (num_envs, 4)
        vel = np.array(obs["vel"])          # (num_envs, 3)
        ang_vel = np.array(obs["ang_vel"])  # (num_envs, 3)
        
        return np.concatenate([pos, quat, vel, ang_vel], axis=1)  # (num_envs, 13)
    
    def _update_history_buffer(self, obs: dict):
        """更新历史状态缓冲区。
        
        将当前状态加入缓冲区，移除最旧的状态。
        
        Args:
            obs: 当前观测字典
        """
        current_state = self._extract_basic_state(obs)  # (num_envs, 13)
        
        # 滚动缓冲区：丢弃最旧的，添加最新的
        # [:, 1:, :] 取第 1 到最后一帧，然后拼接新的一帧
        self._history_buffer = np.concatenate([
            self._history_buffer[:, 1:, :],
            current_state[:, np.newaxis, :]
        ], axis=1)
    
    def observations(self, obs: dict) -> NDArray:
        """将原始观测字典转换为向量。
        
        Args:
            obs: VecDroneRaceEnv 返回的观测字典
            
        Returns:
            (num_envs, OBS_DIM) 的观测向量
        """
        num_envs = self.num_envs
        
        # ========== 1. 提取原始数据 ==========
        pos = np.array(obs["pos"])              # (num_envs, 3) 世界坐标
        vel = np.array(obs["vel"])              # (num_envs, 3) 世界坐标
        ang_vel = np.array(obs["ang_vel"])      # (num_envs, 3)
        quat = np.array(obs["quat"])            # (num_envs, 4) [x, y, z, w] scipy 顺序
        target_gate = np.array(obs["target_gate"])  # (num_envs,)
        gates_pos = np.array(obs["gates_pos"])      # (num_envs, n_gates, 3)
        gates_quat = np.array(obs["gates_quat"])    # (num_envs, n_gates, 4)
        obstacles_pos = np.array(obs["obstacles_pos"])  # (num_envs, n_obstacles, 3)
        
        # ========== 2. 计算无人机姿态相关量 ==========
        rot_matrices = self._quat_to_rotation_matrix(quat)  # (num_envs, 3, 3)
        rot_matrices_flat = rot_matrices.reshape(num_envs, 9)
        vel_body = self._world_to_body_batch(vel, rot_matrices)  # (num_envs, 3)
        
        # ========== 3. 计算门角点 (机体坐标) ==========
        gate1_idx = np.clip(target_gate, 0, self.n_gates - 1)
        gate2_idx = np.clip(target_gate + 1, 0, self.n_gates - 1)
        
        finished_mask = (target_gate == -1)
        gate1_idx = np.where(finished_mask, self.n_gates - 1, gate1_idx)
        gate2_idx = np.where(finished_mask, self.n_gates - 1, gate2_idx)
        
        batch_idx = np.arange(num_envs)
        gate1_pos = gates_pos[batch_idx, gate1_idx]
        gate1_quat = gates_quat[batch_idx, gate1_idx]
        gate2_pos = gates_pos[batch_idx, gate2_idx]
        gate2_quat = gates_quat[batch_idx, gate2_idx]
        
        gate1_corners_body = self._compute_gate_corners_body(
            gate1_pos, gate1_quat, pos, rot_matrices
        )
        gate2_corners_body = self._compute_gate_corners_body(
            gate2_pos, gate2_quat, pos, rot_matrices
        )
        
        # ========== 4. 计算障碍物位置 (机体坐标) ==========
        obstacles_body = self._compute_obstacles_body(
            obstacles_pos, pos, rot_matrices
        )
        
        # ========== 5. 拼接基础观测向量 ==========
        obs_parts = [
            pos,                    # 3D  - 世界坐标位置
            vel_body,               # 3D  - 机体坐标速度
            ang_vel,                # 3D  - 角速度
            rot_matrices_flat,      # 9D  - 旋转矩阵
            gate1_corners_body,     # 12D - 当前门角点
            gate2_corners_body,     # 12D - 下一个门角点
            self._prev_action,      # 4D  - 上一步动作
            obstacles_body,         # 12D - 障碍物位置
        ]
        
        # ========== 6. 添加历史状态 ==========
        if self.n_history > 0:
            # 展平历史缓冲区: (num_envs, n_history, 13) -> (num_envs, n_history * 13)
            history_flat = self._history_buffer.reshape(num_envs, -1)
            obs_parts.append(history_flat)
        
        obs_vector = np.concatenate(obs_parts, axis=1)
        
        # ========== 7. 课程学习 Masking ==========
        obs_vector = self._apply_stage_masking(obs_vector)
        
        return obs_vector.astype(np.float32)
    
    def set_stage(self, stage: int):
        """设置课程学习阶段。"""
        self.stage = stage
        print(f"[RacingObservationWrapper] 切换到 Stage {stage}")
    
    # ========== 辅助函数 (保持不变) ==========
    
    def _quat_to_rotation_matrix(self, quat: NDArray) -> NDArray:
        """四元数转旋转矩阵 (批量)。"""
        rotations = Rotation.from_quat(quat)
        return rotations.as_matrix()
    
    def _world_to_body_batch(self, vec_world: NDArray, rot_matrices: NDArray) -> NDArray:
        """将世界坐标向量批量转换到机体坐标。"""
        return np.einsum('nij,nj->ni', rot_matrices.transpose(0, 2, 1), vec_world)
    
    def _compute_gate_corners_body(
        self,
        gate_pos: NDArray,
        gate_quat: NDArray,
        drone_pos: NDArray,
        drone_rot: NDArray,
    ) -> NDArray:
        """计算门的 4 个角点在机体坐标系下的位置。"""
        num_envs = gate_pos.shape[0]
        gate_rot = Rotation.from_quat(gate_quat).as_matrix()
        corners_world = np.einsum('nij,kj->nki', gate_rot, self.GATE_CORNERS_LOCAL)
        corners_world = corners_world + gate_pos[:, np.newaxis, :]
        corners_rel_world = corners_world - drone_pos[:, np.newaxis, :]
        drone_rot_inv = drone_rot.transpose(0, 2, 1)
        corners_body = np.einsum('nij,nkj->nki', drone_rot_inv, corners_rel_world)
        return corners_body.reshape(num_envs, -1)
    
    def _compute_obstacles_body(
        self,
        obstacles_pos: NDArray,
        drone_pos: NDArray,
        drone_rot: NDArray,
    ) -> NDArray:
        """计算障碍物在机体坐标系下的相对位置。"""
        num_envs = drone_pos.shape[0]
        
        if obstacles_pos.size == 0 or obstacles_pos.shape[1] == 0:
            return np.full((num_envs, 12), 10.0, dtype=np.float32)
        
        rel_world = obstacles_pos - drone_pos[:, np.newaxis, :]
        drone_rot_inv = drone_rot.transpose(0, 2, 1)
        rel_body = np.einsum('nij,nkj->nki', drone_rot_inv, rel_world)
        
        n_obs_actual = min(rel_body.shape[1], 4)
        result = np.full((num_envs, 12), 10.0, dtype=np.float32)
        result[:, :n_obs_actual * 3] = rel_body[:, :n_obs_actual, :].reshape(num_envs, -1)
        
        return result
    
    def _apply_stage_masking(self, obs_vector: NDArray) -> NDArray:
        """根据课程阶段对观测进行 masking。
        
        观测向量布局 (n_history=2 时):
            [0:3]   - pos
            [3:6]   - vel_body
            [6:9]   - ang_vel
            [9:18]  - rot_matrix
            [18:30] - gate1_corners
            [30:42] - gate2_corners  
            [42:46] - prev_action
            [46:58] - obstacles
            [58:84] - history (2 * 13 = 26)
        """
        obs_vector = obs_vector.copy()
        
        if self.stage == 0:
            # Stage 0: 屏蔽下下个门 + 屏蔽障碍物
            obs_vector[:, 30:42] = obs_vector[:, 18:30]
            obs_vector[:, 46:58] = 10.0
            
        elif self.stage == 1:
            # Stage 1: 只屏蔽障碍物
            obs_vector[:, 46:58] = 10.0
        
        return obs_vector