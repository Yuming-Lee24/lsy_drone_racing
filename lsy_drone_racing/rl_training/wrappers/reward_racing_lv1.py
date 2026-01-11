"""奖励塑形 Wrapper - 竞速专用版

针对固定赛道优化：
1. 使用差分奖励 (Potential-based) 替代绝对距离奖励，防止悬停刷分。
2. 加入时间惩罚，鼓励快速完赛。
3. 简化姿态惩罚，允许大机动过弯。
4. 逐渐增加过门奖励
Reward = (d_t-1 - d_t) * C_prog + R_gate + R_finish - P_time - P_crash - P_smooth
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium.vector import VectorEnv, VectorWrapper
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RacingRewardWrapper(VectorWrapper):
    """竞速专用奖励 Wrapper。"""
    
    # 门局部坐标系下的法向量 (门面朝 X 轴)
    GATE_NORMAL_LOCAL = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    def __init__(
        self,
        env: VectorEnv,
        n_gates: int = 4,
        stage: int = 0, # 兼容接口，实际可以忽略
        # ========== 关键系数调整 ==========
        coef_progress: float = 20.0,   # 差分奖励系数，通常要大 (因为差值很小)
        coef_gate: float = 10.0,       # 过单门奖励
        coef_finish: float = 50.0,     # 完赛大奖
        coef_time: float = 0.05,       # 时间惩罚 (每步扣分)
        coef_collision: float = 10.0,  # 碰撞惩罚 (撞了就重开，惩罚大点)
        coef_smooth: float = 0.1,      # 动作平滑
        coef_spin: float = 0.1,        # 角速度惩罚 (防震荡)
        coef_align: float = 0.5,       # 对齐奖励 (辅助引导)
        coef_angle: float = 0.02,
    ):
        super().__init__(env)
        
        self.n_gates = n_gates
        
        self.coefs = {
            "progress": coef_progress,
            "gate": coef_gate,
            "finish": coef_finish,
            "time": coef_time,
            "collision": coef_collision,
            "smooth": coef_smooth,
            "spin": coef_spin,
            "align": coef_align,
            "angle": coef_angle,
        }
        
        # 内部状态
        self._last_dist_to_gate = np.zeros(self.num_envs, dtype=np.float32)
        self._last_action = np.zeros((self.num_envs, 4), dtype=np.float32)
        self._last_target_gate = np.zeros(self.num_envs, dtype=np.int32)

        # 新增：累积奖励追踪
        self._ep_rewards = {
            "progress": np.zeros(self.num_envs, dtype=np.float32),
            "gate": np.zeros(self.num_envs, dtype=np.float32),
            "finish": np.zeros(self.num_envs, dtype=np.float32),
            "align": np.zeros(self.num_envs, dtype=np.float32),
            "time": np.zeros(self.num_envs, dtype=np.float32),
            "ground": np.zeros(self.num_envs, dtype=np.float32),
            "collision": np.zeros(self.num_envs, dtype=np.float32),
            "smooth": np.zeros(self.num_envs, dtype=np.float32),
            "spin": np.zeros(self.num_envs, dtype=np.float32),
            "angle": np.zeros(self.num_envs, dtype=np.float32),
            "total": np.zeros(self.num_envs, dtype=np.float32),
        }
        # Rollout 期间完成的 episode 统计
        self._finished_episodes = {k: [] for k in self._ep_rewards.keys()}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 初始化距离
        self._last_dist_to_gate = self._compute_dist_to_gate(obs)
        
        # 重置状态
        self._last_action = np.zeros((self.num_envs, 4), dtype=np.float32)
        self._last_target_gate = np.array(obs["target_gate"], dtype=np.int32)
        
        return obs, info
    
    def step(self, action):
        # 转换动作格式
        action_array = np.array(action, dtype=np.float32).reshape(self.num_envs, -1)
        
        # 环境步进
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # 计算新奖励
        reward = self._compute_reward(obs, action_array, terminated, truncated)
        
        # 更新状态
        self._update_state(obs, action_array)
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(
        self, 
        obs: dict, 
        action: NDArray,
        terminated: NDArray,
        truncated: NDArray
    ) -> NDArray:
        target_gate = np.array(obs["target_gate"])
        pos = np.array(obs["pos"])
        
        # 1. 计算距离
        curr_dist = self._compute_dist_to_gate(obs)
        
        # ========== 核心: 差分进度奖励 (Potential-based) ==========
        # (上一步距离 - 当前距离)
        # 正值 = 靠近目标; 负值 = 远离目标
        dist_diff = self._last_dist_to_gate - curr_dist
        
        # 处理过门跳变: 
        # 如果过门了，目标变了，距离会突然变大。这一步我们不计算差分奖励，避免巨大的负惩罚。
        # 完赛时 (target=-1) 也不计算
        gate_changed = (target_gate != self._last_target_gate)
        finished = (target_gate == -1)
        
        r_progress = np.where(gate_changed | finished, 0.0, dist_diff)
        r_progress = r_progress * self.coefs["progress"]
        
        # ========== 2. 过门与完赛奖励 (修改版) ==========
        # 检测是否刚过门 (target_gate 增加代表过门)
        passed_gate = (target_gate > self._last_target_gate) & (self._last_target_gate >= 0)

        # 计算递增奖励: (门索引 + 1) * 20
        # 穿过第0个门(索引0)奖励 20，第1个门奖励 40... 以此类推
        incremental_reward = (self._last_target_gate.astype(np.float32) + 1) * self.coefs["gate"]
        r_gate = np.where(passed_gate, incremental_reward, 0.0)

        # 完赛奖励 (如果是最后一个门)
        just_finished = (target_gate == -1) & (self._last_target_gate != -1)
        # 如果第四个门(索引3)是最后一个，完赛奖励可以设为 100 或者更高
        r_finish = np.where(just_finished, self.coefs["finish"], 0.0)
        
        # ========== 3. 时间惩罚 (Step Penalty) ==========
        # 只要没完赛，每一步都扣分，逼它快跑
        p_time = np.where(finished, 0.0, self.coefs["time"])
        
        # ========== 4. 起飞引导 (Altitude) ==========
        # 如果在地上蹭 (z < 0.1)，给一个额外的负分，逼它飞起来
        z = pos[:, 2]
        on_ground = (z < 0.1) & ~finished
        p_ground = np.where(on_ground, 0.1, 0.0)
        
        # ========== 5. 对齐奖励 (辅助) ==========
        # 仅当速度足够大时计算，引导它对准门
        r_align = self._compute_align_reward(obs) * self.coefs["align"]
        
        # ========== 6. 惩罚项 ==========
        # 碰撞 (终止了但不是因为完赛或超时)
        # 注意: VecDroneRaceEnv 在 step 限制到期时也会 terminated=True，需要排除
        is_crash = terminated & ~truncated & ~finished
        p_collision = np.where(is_crash, self.coefs["collision"], 0.0)
        
        # 动作平滑 (防抖)
        action_diff = action - self._last_action
        p_smooth = self.coefs["smooth"] * np.sum(action_diff ** 2, axis=1)
        
        # 角速度 (防震荡)
        ang_vel = np.array(obs["ang_vel"])
        p_spin = self.coefs["spin"] * np.sum(ang_vel ** 2, axis=1)
        
        # 姿态惩罚 (轻微): 只惩罚过大的 Roll/Pitch，防止翻车，但允许侧倾过弯
        quat = np.array(obs["quat"])
        rpy = Rotation.from_quat(quat).as_euler("xyz")
        p_angle = self.coefs["angle"] * np.linalg.norm(rpy[:, :2], axis=-1) # 系数给很小
        
        # ========== 总计 ==========
        reward = (
            r_progress 
            + r_gate 
            + r_finish 
            + r_align
            - p_time 
            - p_ground
            - p_collision 
            - p_smooth 
            - p_spin
            - p_angle
        )
        self._ep_rewards["progress"] += r_progress
        self._ep_rewards["gate"] += r_gate
        self._ep_rewards["finish"] += r_finish
        self._ep_rewards["align"] += r_align
        self._ep_rewards["time"] -= p_time
        self._ep_rewards["ground"] -= p_ground
        self._ep_rewards["collision"] -= p_collision
        self._ep_rewards["smooth"] -= p_smooth
        self._ep_rewards["spin"] -= p_spin
        self._ep_rewards["angle"] -= p_angle
        self._ep_rewards["total"] += reward
        
        # Episode 结束时，记录并重置
        done_mask = terminated | truncated
        if np.any(done_mask):
            for key in self._ep_rewards:
                # 记录结束的 episode 的累积值
                self._finished_episodes[key].extend(self._ep_rewards[key][done_mask].tolist())
                # 重置这些环境
                self._ep_rewards[key][done_mask] = 0.0

        # print(f"last_dist={self._last_dist_to_gate[0]:.3f}, curr_dist={curr_dist[0]:.3f}, dist_diff={dist_diff[0]:.3f},  "
        #         f"r_prog={r_progress[0]:.3f}, r_gate={r_gate[0]:.3f}, "
        #         f"r_finish={r_finish[0]:.3f}, r_align={r_align[0]:.3f}, "
        #         f"p_time={p_time[0]:.3f}, p_collision={p_collision[0]:.3f}, "
        #         f"p_smooth={p_smooth[0]:.3f}, p_spin={p_spin[0]:.3f}")
        return reward.astype(np.float32)

    def _compute_dist_to_gate(self, obs: dict) -> NDArray:
        pos = np.array(obs["pos"])
        gates_pos = np.array(obs["gates_pos"])
        target_gate = np.array(obs["target_gate"])
        
        safe_idx = np.clip(target_gate, 0, self.n_gates - 1)
        batch_idx = np.arange(self.num_envs)
        
        # 获取当前目标门位置
        current_gate_pos = gates_pos[batch_idx, safe_idx]
        
        # 计算距离
        dist = np.linalg.norm(pos - current_gate_pos, axis=1)
        return dist

    def _compute_align_reward(self, obs: dict) -> NDArray:
        """
        简化版对齐奖励：仅鼓励速度与门法向量对齐
        
        让策略自己学习如何穿门，而不是手工设计路径。
        """
        # 1. 提取数据
        vel = np.array(obs["vel"])                      # (N, 3)
        gates_quat = np.array(obs["gates_quat"])        # (N, n_gates, 4)
        target_gate_idx = np.array(obs["target_gate"])  # (N,)
        
        # 处理完赛状态
        valid_mask = (target_gate_idx != -1)
        safe_idx = np.clip(target_gate_idx, 0, self.n_gates - 1)
        
        # 2. 获取目标门的法向量
        batch_indices = np.arange(self.num_envs)
        curr_gate_quat = gates_quat[batch_indices, safe_idx]  # (N, 4)
        
        gate_rot = Rotation.from_quat(curr_gate_quat)
        gate_normal = gate_rot.apply(np.array([1.0, 0.0, 0.0]))  # (N, 3)
        
        # 3. 计算速度与门法向量的对齐度
        # Reward = v · n (速度在正确方向上的投影)
        align_reward = np.sum(vel * gate_normal, axis=1)
        
        # 4. 可选：缩放系数
        align_reward *= 0.5
        
        # 5. 完赛后不给奖励
        align_reward = np.where(valid_mask, align_reward, 0.0)
        
        return align_reward

    def _update_state(self, obs: dict, action: NDArray):
        self._last_dist_to_gate = self._compute_dist_to_gate(obs)
        self._last_action = action.copy()
        self._last_target_gate = np.array(obs["target_gate"], dtype=np.int32)
        
    def set_stage(self, stage: int):
        pass # 固定场景不需要 Stage 调整