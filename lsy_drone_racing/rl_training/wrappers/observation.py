"""观测空间变换 Wrapper

将 VecDroneRaceEnv 的原始观测字典转换为 58D 向量，用于 RL 训练。

观测空间设计 (58D):
    - Position (世界坐标): 3D
    - Linear Velocity (机体坐标): 3D  
    - Angular Velocity: 3D
    - Rotation Matrix (展平): 9D
    - Next Gate 4角点 (机体坐标): 12D
    - Next Next Gate 4角点 (机体坐标): 12D
    - Previous Action: 4D
    - 4个障碍物位置 (机体坐标): 12D
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
    """将原始观测转换为 58D 向量的 Wrapper。
    
    核心功能:
    1. 世界坐标 → 机体坐标变换 (门、障碍物、速度)
    2. 门中心点 → 4角点计算
    3. 课程学习的观测 masking
    4. 维护 prev_action 状态
    """
    
    # 门的局部坐标系下 4 个角点偏移 (门面朝 X 轴，YZ 平面是门框)
    # 顺序: 左上 → 右上 → 右下 → 左下
    GATE_CORNERS_LOCAL = np.array([
        [0.0, -0.2,  0.2],  # 左上
        [0.0,  0.2,  0.2],  # 右上
        [0.0,  0.2, -0.2],  # 右下
        [0.0, -0.2, -0.2],  # 左下
    ], dtype=np.float32)
    
    # 观测维度常量
    OBS_DIM = 58
    SELF_DIM = 18        # pos(3) + vel(3) + ang_vel(3) + rot_mat(9)
    GATES_DIM = 24       # 2个门 × 4角点 × 3坐标
    ACTION_DIM = 4       # prev_action
    OBSTACLES_DIM = 12   # 4个障碍物 × 3坐标
    
    def __init__(
        self, 
        env: VectorEnv,
        n_gates: int = 4,
        n_obstacles: int = 4,
        stage: int = 0,
    ):
        """初始化 Wrapper。
        
        Args:
            env: 底层向量化环境 (VecDroneRaceEnv)
            n_gates: 赛道门的数量
            n_obstacles: 障碍物数量
            stage: 课程阶段 (0=屏蔽障碍物和下下个门, 1=屏蔽障碍物, 2=全开)
        """
        super().__init__(env)
        
        self.n_gates = n_gates
        self.n_obstacles = n_obstacles
        self.stage = stage
        
        # 内部状态：上一步动作
        self._prev_action = np.zeros((self.num_envs, self.ACTION_DIM), dtype=np.float32)
        
        # 定义新的观测空间 (58D 向量)
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
        """重置环境，清零 prev_action。"""
        obs, info = self.env.reset(**kwargs)
        
        # 重置内部状态
        self._prev_action = np.zeros((self.num_envs, self.ACTION_DIM), dtype=np.float32)
        
        # 转换观测
        transformed_obs = self.observations(obs)
        
        return transformed_obs, info
    
    def step(self, action):
        """执行一步，更新 prev_action。"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 转换观测 (使用当前的 prev_action)
        transformed_obs = self.observations(obs)
        
        # 更新 prev_action 为本次执行的动作 (供下一步使用)
        self._prev_action = np.array(action, dtype=np.float32).reshape(self.num_envs, -1)
        
        return transformed_obs, reward, terminated, truncated, info
    
    def observations(self, obs: dict) -> NDArray:
        """将原始观测字典转换为 58D 向量。
        
        Args:
            obs: VecDroneRaceEnv 返回的观测字典
            
        Returns:
            (num_envs, 58) 的观测向量
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
        # 四元数 → 旋转矩阵 (用于坐标变换)
        rot_matrices = self._quat_to_rotation_matrix(quat)  # (num_envs, 3, 3)
        rot_matrices_flat = rot_matrices.reshape(num_envs, 9)  # 展平为 9D
        
        # 速度从世界坐标转到机体坐标
        vel_body = self._world_to_body_batch(vel, rot_matrices)  # (num_envs, 3)
        
        # ========== 3. 计算门角点 (机体坐标) ==========
        # 获取当前门和下一个门的索引
        gate1_idx = np.clip(target_gate, 0, self.n_gates - 1)  # 当前门
        gate2_idx = np.clip(target_gate + 1, 0, self.n_gates - 1)  # 下一个门
        
        # 处理完赛情况 (target_gate == -1)
        finished_mask = (target_gate == -1)
        gate1_idx = np.where(finished_mask, self.n_gates - 1, gate1_idx)
        gate2_idx = np.where(finished_mask, self.n_gates - 1, gate2_idx)
        
        # 取出对应门的位置和朝向
        batch_idx = np.arange(num_envs)
        gate1_pos = gates_pos[batch_idx, gate1_idx]    # (num_envs, 3)
        gate1_quat = gates_quat[batch_idx, gate1_idx]  # (num_envs, 4)
        gate2_pos = gates_pos[batch_idx, gate2_idx]    # (num_envs, 3)
        gate2_quat = gates_quat[batch_idx, gate2_idx]  # (num_envs, 4)
        
        # 计算门的 4 个角点在机体坐标系下的位置
        gate1_corners_body = self._compute_gate_corners_body(
            gate1_pos, gate1_quat, pos, rot_matrices
        )  # (num_envs, 12)
        gate2_corners_body = self._compute_gate_corners_body(
            gate2_pos, gate2_quat, pos, rot_matrices
        )  # (num_envs, 12)
        
        # ========== 4. 计算障碍物位置 (机体坐标) ==========
        obstacles_body = self._compute_obstacles_body(
            obstacles_pos, pos, rot_matrices
        )  # (num_envs, 12)
        
        # ========== 5. 拼接观测向量 ==========
        obs_vector = np.concatenate([
            pos,                    # 3D  - 世界坐标位置
            vel_body,               # 3D  - 机体坐标速度
            ang_vel,                # 3D  - 角速度
            rot_matrices_flat,      # 9D  - 旋转矩阵
            gate1_corners_body,     # 12D - 当前门角点
            gate2_corners_body,     # 12D - 下一个门角点
            self._prev_action,      # 4D  - 上一步动作
            obstacles_body,         # 12D - 障碍物位置
        ], axis=1)  # (num_envs, 58)
        
        # ========== 6. 课程学习 Masking ==========
        obs_vector = self._apply_stage_masking(obs_vector)
        
        return obs_vector.astype(np.float32)
    
    def set_stage(self, stage: int):
        """设置课程学习阶段。
        
        Args:
            stage: 0=屏蔽障碍物和下下个门, 1=屏蔽障碍物, 2=全开
        """
        self.stage = stage
        print(f"[RacingObservationWrapper] 切换到 Stage {stage}")
    
    # ========== 辅助函数 ==========
    
    def _quat_to_rotation_matrix(self, quat: NDArray) -> NDArray:
        """四元数转旋转矩阵 (批量)。
        
        Args:
            quat: (num_envs, 4) 四元数 [x, y, z, w] scipy 顺序
            
        Returns:
            (num_envs, 3, 3) 旋转矩阵
        """
        # scipy.spatial.transform.Rotation 支持批量操作
        rotations = Rotation.from_quat(quat)
        return rotations.as_matrix()  # (num_envs, 3, 3)
    
    def _world_to_body_batch(self, vec_world: NDArray, rot_matrices: NDArray) -> NDArray:
        """将世界坐标向量批量转换到机体坐标。
        
        Args:
            vec_world: (num_envs, 3) 世界坐标下的向量
            rot_matrices: (num_envs, 3, 3) 无人机的旋转矩阵 (世界→机体需要转置)
            
        Returns:
            (num_envs, 3) 机体坐标下的向量
        """
        # R^T @ v 将世界坐标转到机体坐标
        # 使用 einsum 进行批量矩阵-向量乘法
        return np.einsum('nij,nj->ni', rot_matrices.transpose(0, 2, 1), vec_world)
    
    def _compute_gate_corners_body(
        self,
        gate_pos: NDArray,      # (num_envs, 3) 门中心世界坐标
        gate_quat: NDArray,     # (num_envs, 4) 门朝向
        drone_pos: NDArray,     # (num_envs, 3) 无人机位置
        drone_rot: NDArray,     # (num_envs, 3, 3) 无人机旋转矩阵
    ) -> NDArray:
        """计算门的 4 个角点在机体坐标系下的位置。
        
        步骤:
        1. 局部角点 → 门的旋转 → 世界角点
        2. 世界角点 - 无人机位置 → 相对位置 (世界系)
        3. 无人机旋转^-1 → 相对位置 (机体系)
        
        Returns:
            (num_envs, 12) 展平的角点坐标
        """
        num_envs = gate_pos.shape[0]
        
        # Step 1: 局部角点 → 世界角点
        # 门的旋转矩阵
        gate_rot = Rotation.from_quat(gate_quat).as_matrix()  # (num_envs, 3, 3)
        
        # 将局部角点旋转到世界坐标并加上门中心位置
        # corners_local: (4, 3), gate_rot: (num_envs, 3, 3)
        # 结果: (num_envs, 4, 3)
        corners_world = np.einsum('nij,kj->nki', gate_rot, self.GATE_CORNERS_LOCAL)
        corners_world = corners_world + gate_pos[:, np.newaxis, :]  # 广播加法
        
        # Step 2: 世界角点 → 相对位置 (世界系)
        corners_rel_world = corners_world - drone_pos[:, np.newaxis, :]  # (num_envs, 4, 3)
        
        # Step 3: 相对位置 (世界系) → 相对位置 (机体系)
        # 使用无人机旋转矩阵的转置
        drone_rot_inv = drone_rot.transpose(0, 2, 1)  # (num_envs, 3, 3)
        corners_body = np.einsum('nij,nkj->nki', drone_rot_inv, corners_rel_world)  # (num_envs, 4, 3)
        
        # 展平为 12D
        return corners_body.reshape(num_envs, -1)
    
    def _compute_obstacles_body(
        self,
        obstacles_pos: NDArray,  # (num_envs, n_obstacles, 3)
        drone_pos: NDArray,      # (num_envs, 3)
        drone_rot: NDArray,      # (num_envs, 3, 3)
    ) -> NDArray:
        """计算障碍物在机体坐标系下的相对位置。
        
        Returns:
            (num_envs, 12) 展平的障碍物位置 (4个障碍物 × 3坐标)
        """
        num_envs = drone_pos.shape[0]
        
        # 处理无障碍物的情况
        if obstacles_pos.size == 0 or obstacles_pos.shape[1] == 0:
            # 返回很远的默认值 (会被 stage masking 覆盖)
            return np.full((num_envs, 12), 10.0, dtype=np.float32)
        
        # 相对位置 (世界系)
        rel_world = obstacles_pos - drone_pos[:, np.newaxis, :]  # (num_envs, n_obstacles, 3)
        
        # 转换到机体坐标系
        drone_rot_inv = drone_rot.transpose(0, 2, 1)  # (num_envs, 3, 3)
        rel_body = np.einsum('nij,nkj->nki', drone_rot_inv, rel_world)  # (num_envs, n_obstacles, 3)
        
        # 只取前 4 个障碍物，展平为 12D
        # 如果障碍物少于 4 个，用 10.0 填充
        n_obs_actual = min(rel_body.shape[1], 4)
        result = np.full((num_envs, 12), 10.0, dtype=np.float32)
        result[:, :n_obs_actual * 3] = rel_body[:, :n_obs_actual, :].reshape(num_envs, -1)
        
        return result
    
    def _apply_stage_masking(self, obs_vector: NDArray) -> NDArray:
        """根据课程阶段对观测进行 masking。
        
        Args:
            obs_vector: (num_envs, 58) 原始观测向量
            
        Returns:
            (num_envs, 58) masking 后的观测向量
            
        观测向量布局:
            [0:3]   - pos (世界坐标)
            [3:6]   - vel_body
            [6:9]   - ang_vel
            [9:18]  - rot_matrix
            [18:30] - gate1_corners (当前门)
            [30:42] - gate2_corners (下一个门)  
            [42:46] - prev_action
            [46:58] - obstacles
        """
        obs_vector = obs_vector.copy()
        
        if self.stage == 0:
            # Stage 0: 屏蔽下下个门 + 屏蔽障碍物
            # 下下个门设为与当前门相同
            obs_vector[:, 30:42] = obs_vector[:, 18:30]
            # 障碍物设为很远的位置 (机体坐标下)
            obs_vector[:, 46:58] = 10.0
            
        elif self.stage == 1:
            # Stage 1: 只屏蔽障碍物
            obs_vector[:, 46:58] = 10.0
            
        # Stage 2: 不做任何 masking
        
        return obs_vector