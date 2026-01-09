"""
观测空间设计 (更新版):
    - Position Z (世界坐标): 1D  <-- 修改这里
    - Linear Velocity (机体坐标): 3D  
    - Angular Velocity: 3D
    - Rotation Matrix (展平): 9D
    - Next Gate 4角点 (机体坐标): 12D
    - Next Next Gate 4角点 (机体坐标): 12D
    - Previous Action: 4D
    - 4个障碍物位置 (机体坐标): 12D
    - 历史状态 (n_history 帧): n_history * 16D (Pos_Z + Rot + Vel + AngVel)
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
    BASE_OBS_DIM = 56
    SELF_DIM = 16        # pos(1) + vel(3) + ang_vel(3) + rot_mat(9)
    GATES_DIM = 24       # 2个门 × 4角点 × 3坐标
    ACTION_DIM = 4       # prev_action
    OBSTACLES_DIM = 12   # 4个障碍物 × 3坐标
    HISTORY_STATE_DIM = 16  # pos(1) + quat(4) + vel(3) + ang_vel(3)
    
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
        current_obs_dict = self._get_current_obs_dict()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        done_mask = terminated | truncated
        if np.any(done_mask):
            # print(f"\n[DEBUG] Autoreset - Raw obs from base env:")
            for idx in np.where(done_mask)[0]:
                quat = obs["quat"][idx]
                rot = Rotation.from_quat(quat).as_matrix()
                # print(f"  Env {idx} raw quat: {quat}")
                # print(f"  Env {idx} raw rot[0]: {rot[0, :3]}")
        # 🔧 关键：条件更新历史
        if self.n_history > 0 and current_obs_dict is not None:
            if np.any(done_mask):
                # 对于 autoreset 的环境：直接用新状态填充，不滚动旧状态
                init_state = self._extract_basic_state(obs)
                    # 手动清零速度部分（16维中的最后6维）
                init_state[done_mask, 10:13] = 0.0  # hist_vel
                init_state[done_mask, 13:16] = 0.0  # hist_ang_vel
                for i in range(self.n_history):
                    self._history_buffer[done_mask, i, :] = init_state[done_mask]
                
                # 对于正常环境：正常滚动
                active_mask = ~done_mask
                if np.any(active_mask):
                    # 只更新正常环境的历史
                    old_state = self._extract_basic_state(current_obs_dict)
                    self._history_buffer[active_mask] = np.concatenate([
                        self._history_buffer[active_mask, 1:, :],
                        old_state[active_mask, np.newaxis, :]
                    ], axis=1)
            else:
                # 所有环境都正常：使用原来的逻辑
                self._update_history_buffer(current_obs_dict)
        
        # 重置 prev_action
        if np.any(done_mask):
            self._prev_action[done_mask] = 0.0
        
        transformed_obs = self.observations(obs)
        if np.any(done_mask):
            transformed_obs[done_mask, 1:4] = 0.0   # vel_body
            transformed_obs[done_mask, 4:7] = 0.0   # ang_vel
            # print(f"\n[FINAL CHECK before return]")
            # for idx in np.where(done_mask)[0]:
            #     history_raw = transformed_obs[idx, 56:88]  # 32维
            #     frame_0 = history_raw[:16]
            #     frame_1 = history_raw[16:]
                
            #     print(f"  Env {idx}:")
            #     print(f"    Frame 0 (前4): {frame_0[:4]}")
            #     print(f"    Frame 1 (前4): {frame_1[:4]}")
            #     print(f"    Equal? {np.allclose(frame_0, frame_1)}")
            #     print(f"    Total history norm: {np.linalg.norm(history_raw):.4f}")
        self._prev_action = np.array(action, dtype=np.float32).reshape(self.num_envs, -1)
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
            (num_envs, 16) 基础状态向量
        """
        pos = np.array(obs["pos"])          # (num_envs, 3)
        quat = np.array(obs["quat"])        # (num_envs, 4)
        vel = np.array(obs["vel"])          # (num_envs, 3)
        ang_vel = np.array(obs["ang_vel"])  # (num_envs, 3)
        
        
        pos_z = pos[:, 2:3]  # 只取 z 坐标 (num_envs, 1)
        rot_matrices = Rotation.from_quat(quat).as_matrix() # (num_envs, 3, 3)
        rot_flat = rot_matrices.reshape(self.num_envs, 9)   # (num_envs, 9)
        # 1 + 9 + 3 + 3 = 16
        return np.concatenate([pos_z, rot_flat, vel, ang_vel], axis=1)  # (num_envs, 16)
    
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
        pos_z = pos[:, 2:3]            # (num_envs, 1) 高度
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
            pos_z,                    # 1D  - 世界坐标z
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
            # 展平历史缓冲区: (num_envs, n_history, 16) -> (num_envs, n_history * 16)
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
        """计算障碍物在机体坐标系下的相对位置。
        
        改进策略: 
        不直接使用障碍物顶点，而是使用障碍物轴线上距离无人机最近的点。
        这能防止无人机在低空飞行时，因为看顶点觉得"障碍物在很高的地方"而发生碰撞。
        """
        num_envs = drone_pos.shape[0]
        
        if obstacles_pos.size == 0 or obstacles_pos.shape[1] == 0:
            return np.full((num_envs, 12), 10.0, dtype=np.float32)
        
        # ========== 1. 构建"有效"障碍物坐标 (关键修改) ==========
        # 分解障碍物坐标 (N, n_obs, 3)
        obs_x = obstacles_pos[:, :, 0]
        obs_y = obstacles_pos[:, :, 1]
        obs_z_top = obstacles_pos[:, :, 2] # 这是杆子的顶点高度
        
        # 获取无人机高度 (N, 1) - 扩展维度以便广播
        drone_z = drone_pos[:, 2:3]
        
        # 计算有效高度: min(杆顶高度, 无人机高度)
        # 假设杆子是立在地上的 (base=0)。
        # 如果无人机在杆子下面，视作障碍物就在同高度 (Z差值为0)
        # 如果无人机在杆子上面，视作障碍物在下方 (Z差值为负)
        effective_obs_z = np.minimum(obs_z_top, drone_z)
        
        # 重新堆叠为 (N, n_obs, 3)
        # 此时 effective_obs_pos 是杆子轴线上离无人机最近的点
        effective_obs_pos = np.stack([obs_x, obs_y, effective_obs_z], axis=-1)
        
        # ========== 2. 计算相对位置并旋转 ==========
        # 计算世界坐标系下的相对向量
        rel_world = effective_obs_pos - drone_pos[:, np.newaxis, :]
        
        # 旋转到机体坐标系
        # drone_rot 是 (N, 3, 3) 旋转矩阵
        drone_rot_inv = drone_rot.transpose(0, 2, 1)
        
        # Einsum: 批量矩阵乘法
        # nij: env i, row j (inv matrix)
        # nkj: env i, obs k, row j (vector)
        # -> nki: env i, obs k, row i (result)
        rel_body = np.einsum('nij,nkj->nki', drone_rot_inv, rel_world)
        
        # ========== 3. 排序并取最近的 n 个 ==========
        # 计算距离 (用于排序)
        dists = np.linalg.norm(rel_body, axis=2)
        
        # 排序取前 4 个
        # argsort 默认是升序，所以取前面的是最近的
        sorted_idx = np.argsort(dists, axis=1)
        
        # 取出最近的 4 个障碍物 (如果不足 4 个则处理)
        n_obs_available = rel_body.shape[1]
        n_keep = min(n_obs_available, 4) # 假设观测空间固定为 4 个
        
        # 创建结果容器 (默认填充 10.0 表示很远)
        result = np.full((num_envs, 12), 10.0, dtype=np.float32)
        
        # 这种 gather 操作在 numpy 里需要利用 fancy indexing
        # 创建 batch 索引: [[0,0,0,0], [1,1,1,1], ...]
        batch_indices = np.arange(num_envs)[:, None]
        keep_indices = sorted_idx[:, :n_keep]
        
        nearest_rel_body = rel_body[batch_indices, keep_indices] # (num_envs, n_keep, 3)
        
        # 填入结果
        result[:, :n_keep * 3] = nearest_rel_body.reshape(num_envs, -1)
        
        return result
    
    def _apply_stage_masking(self, obs_vector: NDArray) -> NDArray:
        """根据课程阶段对观测进行 masking。
        
        观测向量布局 (更新后):
            [0:1]   - pos_z (1D)
            [1:4]   - vel_body
            [4:7]   - ang_vel
            [7:16]  - rot_matrix
            [16:28] - gate1_corners
            [28:40] - gate2_corners  <-- 索引变更 (-2)
            [40:44] - prev_action
            [44:56] - obstacles      <-- 索引变更 (-2)
            [56:...] - history
        """
        obs_vector = obs_vector.copy()
        
        if self.stage == 0:
            # Stage 0: 屏蔽下下个门 + 屏蔽障碍物
            # 将 gate2 (28:40) 替换为 gate1 (16:28)，模拟只有当前门可见
            obs_vector[:, 28:40] = obs_vector[:, 16:28]
            # 屏蔽障碍物
            obs_vector[:, 44:56] = 10.0
            
        if self.stage == 1:
            # Stage 1: 只屏蔽障碍物
            obs_vector[:, 44:56] = 10.0
        elif self.stage == 2:
            # Stage 2: 全开
            pass
        
        return obs_vector