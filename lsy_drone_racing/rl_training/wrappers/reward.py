"""奖励塑形 Wrapper

为 RL 训练提供 dense reward，替代原始环境的稀疏奖励。

奖励组成:
    R_total = R_progress + R_gate + R_align - P_collision - P_smooth - P_spin

其中:
    - R_progress: 距离差奖励 (靠近门)
    - R_gate: 过门稀疏奖励
    - R_align: 速度方向与门法向量对齐奖励
    - P_collision: 碰撞惩罚
    - P_smooth: 动作平滑惩罚
    - P_spin: 角速度惩罚
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium.vector import VectorEnv, VectorWrapper
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RacingRewardWrapper(VectorWrapper):
    """奖励塑形 Wrapper，计算 dense reward。
    
    核心功能:
    1. 计算各项奖励和惩罚
    2. 维护距离、动作等历史状态
    3. 支持课程学习的系数调整
    """
    
    # 门局部坐标系下的法向量 (门面朝 X 轴)
    GATE_NORMAL_LOCAL = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    def __init__(
        self,
        env: VectorEnv,
        n_gates: int = 4,
        stage: int = 0,
        # 奖励系数
        coef_progress: float = 1.0,
        coef_gate: float = 10.0,
        coef_align: float = 0.5,
        coef_collision: float = 5.0,
        coef_smooth: float = 0.1,
        coef_spin: float = 0.02,
        coef_angle: float = 0.06,  # 新增

    ):
        """初始化 Wrapper。
        
        Args:
            env: 底层向量化环境
            n_gates: 赛道门的数量
            stage: 课程阶段
            coef_*: 各项奖励/惩罚的系数
        """
        super().__init__(env)
        
        self.n_gates = n_gates
        self.stage = stage
        
        # 保存基础系数 (用于 set_stage 时恢复)
        self._base_coefs = {
            "progress": coef_progress,
            "gate": coef_gate,
            "align": coef_align,
            "collision": coef_collision,
            "smooth": coef_smooth,
            "spin": coef_spin,
            "angle": coef_angle,  # 新增
        }
        
        # 当前使用的系数
        self.coefs = self._base_coefs.copy()
        
        # 根据初始 stage 调整系数
        self._apply_stage_coefs()
        
        # 内部状态
        self._last_dist_to_gate = np.zeros(self.num_envs, dtype=np.float32)
        self._last_action = np.zeros((self.num_envs, 4), dtype=np.float32)
        self._last_target_gate = np.zeros(self.num_envs, dtype=np.int32)
    
    def reset(self, **kwargs):
        """重置环境，初始化内部状态。"""
        obs, info = self.env.reset(**kwargs)
        
        # 计算初始距离
        self._last_dist_to_gate = self._compute_dist_to_gate(obs)
        
        # 重置其他状态
        self._last_action = np.zeros((self.num_envs, 4), dtype=np.float32)
        self._last_target_gate = np.array(obs["target_gate"], dtype=np.int32)
        
        return obs, info
    
    def step(self, action):
        """执行一步，计算 shaped reward。"""
        # 保存当前 action (用于下一步计算 smooth)
        action_array = np.array(action, dtype=np.float32).reshape(self.num_envs, -1)
        # action = action.at[..., 2].set(0.0)  # 强制 yaw 为 0

        # 执行环境 step
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # 计算 shaped reward
        shaped_reward = self._compute_reward(obs, action_array, terminated)
        
        # 更新内部状态
        self._update_state(obs, action_array)
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _compute_reward(
        self, 
        obs: dict, 
        action: NDArray,
        terminated: NDArray,
    ) -> NDArray:
        """计算 shaped reward。
        
        Args:
            obs: 当前观测字典
            action: 当前动作
            terminated: 终止标志
            
        Returns:
            (num_envs,) shaped reward
        """
        target_gate = np.array(obs["target_gate"])
        
        # ========== 1. R_progress: 距离差奖励 ==========
        curr_dist = self._compute_dist_to_gate(obs)
        r_progress = np.exp(-2.0 * curr_dist)
        
        # 处理过门时的距离跳变：过门后 target 变了，这一步 progress 设为 0
        gate_changed = (target_gate != self._last_target_gate)
        r_progress = np.where(gate_changed, 0.0, r_progress)
        
        # 处理完赛情况：target_gate == -1 时不计算
        pos = np.array(obs["pos"])
        z = pos[:, 2]  # 高度

        # 1. 起飞奖励：强烈鼓励离开地面
        r_altitude = 1.0 * np.clip(z / 0.5, 0, 1.0)  # z=0→0, z=0.5m→1.0
        finished = (target_gate == -1)
        airborne = z > 0.15  # 离地 15cm 以上
        r_progress = np.where(airborne, np.exp(-2.0 * curr_dist), 0.0)
        r_progress = np.where(finished, 0.0, r_progress)
        
        r_progress = self.coefs["progress"] * r_progress
        
        # ========== 2. R_gate: 过门奖励 ==========
        # 检测过门：target_gate 增加了 (且不是因为 reset)
        passed_gate = (target_gate > self._last_target_gate) & (self._last_target_gate >= 0)
        r_gate = np.where(passed_gate, self.coefs["gate"], 0.0)
        
        # ========== 3. R_align: 速度对齐奖励 ==========
        r_align = self._compute_align_reward(obs)
        r_align = self.coefs["align"] * r_align
        
        # ========== 4. P_collision: 碰撞惩罚 ==========
        # 碰撞 = 终止了但没完赛
        collision = terminated & (target_gate != -1)
        p_collision = np.where(collision, self.coefs["collision"], 0.0)
        
        # ========== 5. P_smooth: 动作平滑惩罚 ==========
        action_diff = action - self._last_action
        p_smooth = self.coefs["smooth"] * np.sum(action_diff ** 2, axis=1)
        
        # ========== 6. P_spin: 角速度惩罚 ==========
        ang_vel = np.array(obs["ang_vel"])
        p_spin = self.coefs["spin"] * np.sum(ang_vel ** 2, axis=1)

        # ========== 7. P_angle: 姿态惩罚 ==========
        quat = np.array(obs["quat"])  # (num_envs, 4)
        rpy = Rotation.from_quat(quat).as_euler("xyz")  # (num_envs, 3)
        rpy_norm = np.linalg.norm(rpy, axis=-1)  # (num_envs,)
        p_angle = self.coefs["angle"] * rpy_norm

        # ========== 8. R_altitude: 高度奖励 ==========


        # ========== 汇总 ==========
        # reward = r_progress + r_gate + r_align - p_collision - p_smooth - p_spin
        reward = r_progress + r_gate + r_align + r_altitude- p_smooth - p_angle - p_collision

        return reward.astype(np.float32)
    
    def _compute_dist_to_gate(self, obs: dict) -> NDArray:
        """计算无人机到当前目标门的距离。
        
        Args:
            obs: 观测字典
            
        Returns:
            (num_envs,) 距离
        """
        pos = np.array(obs["pos"])              # (num_envs, 3)
        gates_pos = np.array(obs["gates_pos"])  # (num_envs, n_gates, 3)
        target_gate = np.array(obs["target_gate"])  # (num_envs,)
        
        # 处理完赛情况
        safe_idx = np.clip(target_gate, 0, self.n_gates - 1)
        
        # 取出目标门位置
        batch_idx = np.arange(self.num_envs)
        gate_pos = gates_pos[batch_idx, safe_idx]  # (num_envs, 3)
        
        # 计算欧氏距离
        dist = np.linalg.norm(pos - gate_pos, axis=1)
        
        return dist.astype(np.float32)
    
    def _compute_align_reward(self, obs: dict) -> NDArray:
        """计算速度方向与门法向量的对齐奖励。
        
        R_align = (v / |v|) · n_gate
        
        Args:
            obs: 观测字典
            
        Returns:
            (num_envs,) 对齐奖励 (范围 [-1, 1])
        """
        vel = np.array(obs["vel"])              # (num_envs, 3)
        gates_quat = np.array(obs["gates_quat"])  # (num_envs, n_gates, 4)
        target_gate = np.array(obs["target_gate"])  # (num_envs,)
        
        # 计算速度大小
        vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)  # (num_envs, 1)
        
        # 速度太小时不计算 (避免除零)
        min_vel = 0.1
        vel_valid = (vel_norm.squeeze() > min_vel)
        
        # 速度方向 (归一化) - 先用安全值避免除零警告
        safe_vel_norm = np.maximum(vel_norm, 1e-8)  # 避免除零
        vel_dir = vel / safe_vel_norm  # (num_envs, 3)
        
        # 获取目标门的法向量
        safe_idx = np.clip(target_gate, 0, self.n_gates - 1)
        batch_idx = np.arange(self.num_envs)
        gate_quat = gates_quat[batch_idx, safe_idx]  # (num_envs, 4)
        
        # 将门局部法向量旋转到世界坐标
        gate_rot = Rotation.from_quat(gate_quat)
        gate_normal = gate_rot.apply(self.GATE_NORMAL_LOCAL)  # (num_envs, 3)
        
        # 计算点积 (对齐程度)
        align = np.sum(vel_dir * gate_normal, axis=1)  # (num_envs,)
        
        # 速度太小或已完赛时，对齐奖励为 0
        finished = (target_gate == -1)
        align = np.where(vel_valid & ~finished, align, 0.0)
        
        return align.astype(np.float32)
    
    def _update_state(self, obs: dict, action: NDArray):
        """更新内部状态。"""
        self._last_dist_to_gate = self._compute_dist_to_gate(obs)
        self._last_action = action.copy()
        self._last_target_gate = np.array(obs["target_gate"], dtype=np.int32)
    
    def set_stage(self, stage: int):
        """设置课程学习阶段，调整奖励系数。
        
        Args:
            stage: 0=初级, 1=中级, 2=高级
        """
        self.stage = stage
        self._apply_stage_coefs()
        print(f"[RacingRewardWrapper] 切换到 Stage {stage}, 系数: {self.coefs}")
    
    def _apply_stage_coefs(self):
        """根据当前 stage 调整奖励系数。"""
        # 不同阶段的系数调整
        stage_multipliers = {
            # Stage 0: 低碰撞惩罚 (看不到障碍物)，低平滑惩罚 (允许探索)
            0: {"collision": 0.2, "smooth": 0.5, "spin": 0.5},
            # Stage 1: 中等惩罚
            1: {"collision": 0.6, "smooth": 0.8, "spin": 0.8},
            # Stage 2: 完整惩罚
            2: {"collision": 1.0, "smooth": 1.0, "spin": 1.0},
        }
        
        multipliers = stage_multipliers.get(self.stage, stage_multipliers[2])
        
        # 应用乘数
        self.coefs = self._base_coefs.copy()
        for key, mult in multipliers.items():
            self.coefs[key] = self._base_coefs[key] * mult