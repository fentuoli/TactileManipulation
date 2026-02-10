# TactileManipulation DOF Mapping 问题记录与修复

## 背景

TactileManipulation 是将 ManipTrans（IsaacGym）迁移到 IsaacLab 的版本。迁移过程中 DOF mapping 逻辑引入了多个 bug。

## 核心概念

### DOF Mapping 是什么

ManipTrans 的目标是把人手（MANO 模型，20 个关节）的动作迁移到灵巧机械手上。不同机械手结构差异巨大（Inspire 12 DOF, Shadow 22 DOF, Allegro 16 DOF），因此需要一张"翻译表"（`hand2dex_mapping`）在 MANO 关节和机械手 body/joint 之间建立映射。

### 三层名字对应关系

| 代码概念 | URDF 对应 | 含义 |
|---------|----------|------|
| `body_names` | `<link>` | 刚体零件，有位置/姿态 |
| `dof_names` | `<joint>` (revolute/prismatic) | 可控关节轴，有角度值 |
| `hand2dex_mapping` | — | MANO 关节名 → 机械手 body 名 |

名字必须与 URDF 中的 link/joint 名完全一致。

### 三个系统的 DOF 顺序不同

同一个 URDF，三个系统给出不同的 DOF 索引顺序：

| 系统 | 遍历方式 | Inspire RH 示例 |
|------|---------|----------------|
| `dexhand.dof_names` | 手动定义 | index(0-1), middle(2-3), pinky(4-5), ring(6-7), thumb(8-11) |
| `pytorch_kinematics` | URDF 文件顺序 | thumb(0-3), index(4-5), middle(6-7), ring(8-9), pinky(10-11) |
| IsaacLab (USD/PhysX) | 广度优先 | index_prox, middle_prox, pinky_prox, ring_prox, thumb_yaw, index_inter, ... |

因此需要 `isaac2chain_order` 和 `demo_to_isaaclab_dof_mapping` 做索引转换。

### URDF → USD 转换的影响

IsaacLab 会将 URDF 自动转为 USD 格式。转换**不改变名字，但会改变顺序**。`merge_fixed_joints` 必须设为 `False`，否则指尖 body 会被合并消失。

---

## 发现的问题与修复

### 问题 1（严重）：`isaac2chain_order` 建了但没用于 FK 重排

**文件**: `main/dataset/mano2dexhand.py`

**原代码**:
```python
# 构建时用模糊子串匹配（可能错配）
for joint_name in self.joint_names:
    for i, dof_name in enumerate(dexhand.dof_names):
        if dof_name in joint_name or joint_name in dof_name:  # 模糊匹配
            ...

# FK 调用时没有用 isaac2chain_order 重排，只截断了长度
chain_dof_pos = opt_dof_pos_clamped[:, :len(self.isaac2chain_order)]
ret = self.chain.forward_kinematics(chain_dof_pos)  # 错！没重排
```

**后果**: thumb 的 DOF 值被喂给了 index finger 的 FK 输入，retargeting 结果完全错误。

**修复**:
```python
# 构建时用精确名字匹配
dof_name_list = list(dexhand.dof_names)
self.isaac2chain_order = []
for j in self.joint_names:
    if j in dof_name_list:
        self.isaac2chain_order.append(dof_name_list.index(j))
    else:
        raise ValueError(f"FK chain joint '{j}' not found in dexhand.dof_names: {dof_name_list}")

# FK 调用时用 isaac2chain_order 重排（与原版 ManipTrans 一致）
ret = self.chain.forward_kinematics(opt_dof_pos_clamped[:, self.isaac2chain_order])
```

对于 Inspire RH，正确的 `isaac2chain_order` 应为 `[8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5]`。

### 问题 2（中等）：关节限位硬编码

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

**原代码**: Inspire 手硬编码了 12 个限位值（按 IsaacLab 广度优先顺序），其他手 fallback 到默认值。

**修复**: 统一从 URDF 解析 `<limit lower=... upper=...>`，按 IsaacLab 的 `actual_joint_names` 顺序构建。

### 问题 3（中等）：retargeting 关节限位默认 [-π, π]

**文件**: `main/dataset/mano2dexhand.py`

**原代码**: `torch.full((n_dofs,), -np.pi, ...)`

**修复**: 用 `xml.etree.ElementTree` 解析 URDF，按 `dexhand.dof_names` 顺序提取每个 joint 的真实限位。

### 问题 4（低）：wrist body 索引硬编码为 0

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

**原代码**: `self.apply_forces[:, 0, :]`，假设 wrist 是 body 列表中的第一个。

**修复**: 通过 `self.dexhand.to_dex("wrist")[0]` 查名字，再从 `body_name_to_idx` 获取真实索引存为 `self.wrist_body_idx`。

### 问题 5（致命）：手腕外力/力矩未实际施加到仿真

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

**原代码**: `_apply_pid_control` 和 `_apply_force_control` 计算了 `self.apply_forces` 和 `self.apply_torques`，但没有调用任何 API 将力施加到仿真中。只留了一行注释：
```python
# Note: In IsaacLab, force application is handled differently
# This may need adjustment based on the actual IsaacLab API
```

**后果**: 手腕完全没有受到力控制，只靠初始 root state 位置。手无法跟踪 demo 轨迹中的腕部运动。

**修复**: 调用 IsaacLab 的 `set_external_force_and_torque()` API（等价于 IsaacGym 的 `gym.apply_rigid_body_force_tensors()`）：
```python
self.hand.set_external_force_and_torque(
    forces=self.apply_forces,
    torques=self.apply_torques,
    body_ids=None,       # 所有 body（只有 wrist_body_idx 处有非零值）
    env_ids=None,        # 所有环境
    is_global=True,      # ENV_SPACE = 全局坐标系
)
```
IsaacLab 的 `DirectRLEnv` 会在 `_apply_action()` 之后自动调用 `scene.write_data_to_sim()` 将缓冲的力写入仿真。

### 问题 6（严重）：奖励函数缺失 15 项（19 项只实现了 4 项）

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

**原代码**: 只有 4 个奖励项：
```python
reward = (
    0.1 * reward_eef_pos
    + 0.6 * reward_eef_rot
    + 5.0 * reward_obj_pos
    + 1.0 * reward_obj_rot
)
```

**缺失项及权重**:

| 缺失奖励项 | 权重 | 含义 |
|-----------|------|------|
| `reward_thumb_tip_pos` | 0.9 | 拇指指尖位置 |
| `reward_index_tip_pos` | 0.8 | 食指指尖位置 |
| `reward_middle_tip_pos` | 0.75 | 中指指尖位置 |
| `reward_ring_tip_pos` | 0.6 | 无名指指尖位置 |
| `reward_pinky_tip_pos` | 0.6 | 小指指尖位置 |
| `reward_level_1_pos` | 0.5 | 近端关节 |
| `reward_level_2_pos` | 0.3 | 其他关节 |
| `reward_eef_vel` | 0.1 | 手腕线速度 |
| `reward_eef_ang_vel` | 0.05 | 手腕角速度 |
| `reward_joints_vel` | 0.1 | 关节速度 |
| `reward_obj_vel` | 0.1 | 物体线速度 |
| `reward_obj_ang_vel` | 0.1 | 物体角速度 |
| `reward_finger_tip_force` | 1.0 | 指尖接触力 |
| `reward_power` | 0.5 | 关节功率惩罚 |
| `reward_wrist_power` | 0.5 | 手腕功率惩罚 |

**后果**: 手指完全没有跟踪激励（缺失总权重 5.0），无抓握激励，无能量约束。

**修复**: 补全所有 19 个奖励项，权重与原版 ManipTrans 完全一致。新增了：
- `demo_body_to_isaaclab_indices` 映射：将 IsaacLab body 位置重排到 demo 顺序，用于关节位置比较
- `contact_body_indices` 映射：提取指尖接触力
- `weight_idx` 分组：按 `dexhand.weight_idx` 定义分组计算关节误差

**额外修复**: 四元数角度提取 bug — `diff_eef_rot[:, 3]` 改为 `diff_eef_rot[:, 0]`（IsaacLab wxyz 格式中 w 在 index 0）。

### 问题 7（严重）：终止条件过于简化

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

**原代码**: 只检查物体位置误差：
```python
terminated = diff_obj_pos_dist > 0.1
```

**原版 ManipTrans 的完整终止条件**:
- 8 个逐关节阈值检查（物体位置、拇指、食指、中指、小指、无名指、level_1、level_2）
- 物体旋转阈值（>30°）
- 指尖穿透检测（指尖距离 <0.005m 但无接触力）
- 6 个速度安全检查（error_buf）
- `running_progress_buf >= 8` 保护（前 8 步不判 fail）
- 课程学习 `scale_factor`（exp_decay 从 1.0 衰减到 0.7）

**修复**: 补全所有终止条件，并通过 `_reward_cache` 字典复用奖励函数中已计算的中间值避免重复计算。

### 问题 8（中等）：物理参数与原版差异大

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

| 参数 | 原值（错误） | 修复后（与 ManipTrans 一致） |
|------|------------|--------------------------|
| `orientation_scale` | 1.0 | **0.1**（力矩缩放 10 倍差异会导致手腕旋转不稳定） |
| `act_moving_average` | 0.9 | **0.4**（与训练 CLI 参数一致） |
| `tighten_steps` | 100000 | **3200**（课程学习速度） |
| `max_episode_length` | 1000 | **1200**（与原版一致） |
| `physics_material friction` | 1.0 | **4.0**（与原版手摩擦力一致，防止物体滑落） |

### 问题 9（低）：观测中关节 delta 未使用实际 body state

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

**原代码**:
```python
delta_joints_pos = target_joints_pos.reshape(nE, -1)  # Simplified for now
delta_joints_vel = target_joints_vel.reshape(nE, -1)  # Simplified for now
```

**修复**: 使用实际 body state 与 demo 的差值：
```python
cur_joints_pos = self.hand_body_pos  # (nE, n_bodies-1, 3) 已按 demo 顺序排列
delta_joints_pos = (target_joints_pos - cur_joints_pos[:, None]).reshape(nE, -1)
```

### 问题 10（低）：`running_progress_buf` 从未递增

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

**原代码**: `running_progress_buf` 在 `_init_buffers` 中初始化为 0，在 `_reset_idx` 中重置为 0，但从未递增。

**后果**: 终止条件中的 `running_progress_buf >= 8` 保护永远不生效（始终为 False），导致第 1 步就可能判定失败。

**修复**: 在 `_get_dones()` 中添加 `self.running_progress_buf += 1`。

---

## 修改的文件

1. `main/dataset/mano2dexhand.py` — 修复 isaac2chain_order 构建/使用、URDF 关节限位解析
2. `maniptrans_envs/tasks/dexhand_manip_env.py` — 修复力施加、奖励函数、终止条件、物理参数、观测 delta、progress 计数

## 验证方法

1. **Retargeting**:
   ```bash
   cd /path/to/TactileManipulation
   PYTHONPATH=. python main/dataset/mano2dexhand.py --dexhand inspire --data_idx g0 --side right
   ```
   - 检查打印的 `isaac2chain_order` 是否为 `[8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5]`
   - 检查 loss 是否正常收敛

2. **训练环境**:
   - 检查 `[INFO] Demo to IsaacLab DOF mapping` 无 WARNING
   - 检查 `[INFO] Joint limits` 与 URDF 一致
   - 检查 `[INFO] Wrist body index` 指向正确的 body
   - 检查 `[INFO] Demo body to IsaacLab indices` 无 WARNING
   - 检查 `[INFO] Contact body indices` 包含 5 个有效索引
   - 检查奖励日志中 `reward_joints_pos`、`reward_power`、`reward_finger_tip_force` 均有非零值
   - 检查手腕在仿真中有实际运动（不是静止不动）

## 注意事项

- 修复后需要**重新跑 retargeting** 生成正确的 `opt_dof_pos` 数据，之前生成的数据因 DOF 顺序错误不可用
- `merge_fixed_joints` 必须保持 `False`，否则指尖 body 会被合并
- Inspire 手真实硬件是 6 主动自由度（6 电机驱动 12 关节），代码中建模为 12 个独立 DOF（仿真中的理想化处理）
- IsaacLab 的 `set_external_force_and_torque()` 是缓冲式 API，需要 `write_data_to_sim()` 才生效，`DirectRLEnv` 会自动调用
- 四元数在 IsaacLab 中统一使用 **wxyz** 格式（w 在 index 0），与 IsaacGym 的 xyzw 不同
- 接触力通过 `root_physx_view.get_net_contact_forces()` 获取，如果 API 不可用会 fallback 到零值
