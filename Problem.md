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

### 问题 11（严重）：四元数格式不兼容 — IsaacGym 训练的 imitator 在 IsaacLab 中无法使用

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

**原代码**: 观测中的四元数统一使用 IsaacLab 的 wxyz 格式，但预训练的 base imitator（behavioral cloning 模型）是在 IsaacGym 环境下训练的，期望 xyzw 格式输入。

**后果**: 喂给 imitator 的四元数格式错误，base policy 输出完全错误的动作，残差策略无法在此基础上学习有效的修正。

**修复**:
1. 新增配置项 `use_isaacgym_imitator: bool = True`，控制观测中四元数的格式
2. 新增辅助函数：
   ```python
   def wxyz_to_xyzw(quat):         # IsaacLab → IsaacGym 格式转换
   def quat_mul_xyzw(a, b):        # xyzw 格式的四元数乘法
   def quat_conjugate_xyzw(q):     # xyzw 格式的四元数共轭
   ```
3. 在 `_compute_target_observations()` 中，对手腕四元数和物体四元数的计算分别做条件转换：
   - `use_isaacgym_imitator=True`：将 wxyz 转为 xyzw，用 `quat_mul_xyzw` 计算 delta
   - `use_isaacgym_imitator=False`：保持 wxyz，用 IsaacLab 原生 `quat_mul` 计算 delta

**注意**: 此选项仅影响**观测中**的四元数格式（喂给 imitator 的输入）。奖励函数和终止条件中的四元数计算始终使用 wxyz 格式（IsaacLab 内部一致性）。

### 问题 12（严重）：缺少重力课程学习（Gravity Curriculum）

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

**原版 ManipTrans 行为**: 通过 IsaacGym 的 `apply_randomizations` 框架实现重力渐进：
- 重力从 0 线性增加到 -9.81 m/s²，经过 1920 步完成
- 公式：`effective_gravity = -9.81 * min(step, 1920) / 1920`
- 手的 `disable_gravity=True`，不受重力课程影响
- 只有物体受到重力变化的影响
- 更新频率：每 32 个 env step 更新一次

**IsaacLab 中的实现方式**: 采用**补偿力**方案（而非直接修改 PhysX 全局重力），物理上等价：
```python
# 向上补偿力 = 物体质量 × 9.81 × (1 - gravity_scale)
# gravity_scale 从 0 → 1 渐变
compensation_z = self.manip_obj_mass * 9.81 * (1.0 - self.gravity_scale)
self.object_gravity_force[:, 0, 2] = compensation_z
self.object.set_external_force_and_torque(forces=self.object_gravity_force, ...)
```

**为什么用补偿力而非修改 PhysX 重力**: IsaacLab 的 GPU pipeline 中直接修改全局重力的 API 不确定是否可靠（缓存问题），而补偿力方案利用已验证的 `set_external_force_and_torque()` API，效果完全等价（因为手本身 `disable_gravity=True`）。

### 问题 13（严重）：缺少物体摩擦力课程学习（Friction Curriculum）

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

**原版 ManipTrans 行为**: 物体摩擦力从 3× 默认值线性衰减到 1× 默认值：
- 公式：`friction_scale = 3 × sched_scaling + 1 × (1 - sched_scaling)`
- `sched_scaling = 1 - min(step, 1920) / 1920`（linear_decay）
- 训练初期高摩擦力防止物体滑落，逐步降低到真实值

**后果**: 缺少摩擦力课程导致训练初期物体容易从手中滑落，增加学习难度。

**修复**: 通过 PhysX tensor API 动态更新物体材质属性：
```python
mat_props = self.object.root_physx_view.get_material_properties()
mat_props[:, :, 0] = original_static_friction * friction_scale   # static
mat_props[:, :, 1] = original_dynamic_friction * friction_scale  # dynamic
self.object.root_physx_view.set_material_properties(mat_props)
```

### 问题 14（中等）：缺少 `manip_obj_weight` 观测

**文件**: `maniptrans_envs/tasks/dexhand_manip_env.py`

**原版 ManipTrans**: 观测向量中包含 `manip_obj_weight = mass × |gravity.z|`（1 维），告知策略当前有效重力大小。

**后果**: 策略无法感知重力课程的当前阶段，无法根据重力大小调整抓握力度。

**修复**:
```python
manip_obj_weight = (self.manip_obj_mass * 9.81 * self.gravity_scale).unsqueeze(-1)
obs = torch.cat([prop_obs, target_obs, manip_obj_weight], dim=-1)
```
同时更新 `num_observations` 计算，增加 1 维。

### 新增配置项（问题 12-14 相关）

```python
# Domain Randomization / Curriculum
enable_gravity_curriculum: bool = True   # 重力从 0 → -9.81 渐变
gravity_curriculum_steps: int = 1920     # 达到完整重力的步数
enable_friction_curriculum: bool = True  # 物体摩擦力从 3× → 1× 渐变
friction_curriculum_steps: int = 1920    # 达到正常摩擦力的步数
friction_curriculum_init_scale: float = 3.0  # 初始摩擦力倍率
dr_frequency: int = 32                  # DR 更新频率（每 N 个 env step）
```

### 实现细节

- **延迟初始化** (`_dr_initialized` flag)：PhysX views 在第一个仿真步之前不可用，因此在首次调用 `_apply_domain_randomization()` 时才读取物体质量和原始摩擦力
- **训练/测试区分**：DR 仅在训练时生效（`self.training` 检查），测试时 `gravity_scale=1.0`、`friction_scale=1.0`
- **物体质量上限**：`clamp(max=0.5)` 防止异常大质量导致补偿力过大
- **set_external_force_and_torque 的叠加**：手腕力和物体重力补偿力分别通过 `self.hand.set_external_force_and_torque()` 和 `self.object.set_external_force_and_torque()` 施加，互不干扰

---

## 修改的文件

1. `main/dataset/mano2dexhand.py` — 修复 isaac2chain_order 构建/使用、URDF 关节限位解析
2. `maniptrans_envs/tasks/dexhand_manip_env.py` — 修复力施加、奖励函数、终止条件、物理参数、观测 delta、progress 计数、四元数双格式支持、重力/摩擦力课程学习、`manip_obj_weight` 观测

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
   - 检查 `[INFO] Object mass:` 打印正确的物体质量
   - 检查 `[INFO] Original object friction:` 打印正确的摩擦力值
   - 检查训练初期重力补偿力非零（物体不会立即坠落），1920 步后归零
   - 检查 `use_isaacgym_imitator=True` 时 imitator 行为正常（手跟踪 demo 轨迹）

## 注意事项

- 修复后需要**重新跑 retargeting** 生成正确的 `opt_dof_pos` 数据，之前生成的数据因 DOF 顺序错误不可用
- `merge_fixed_joints` 必须保持 `False`，否则指尖 body 会被合并
- Inspire 手真实硬件是 6 主动自由度（6 电机驱动 12 关节），代码中建模为 12 个独立 DOF（仿真中的理想化处理）
- IsaacLab 的 `set_external_force_and_torque()` 是缓冲式 API，需要 `write_data_to_sim()` 才生效，`DirectRLEnv` 会自动调用
- 四元数在 IsaacLab 中统一使用 **wxyz** 格式（w 在 index 0），与 IsaacGym 的 xyzw 不同
- 接触力通过 `root_physx_view.get_net_contact_forces()` 获取，如果 API 不可用会 fallback 到零值
- IsaacGym 训练的 imitator 使用 **xyzw** 四元数格式，需要设 `use_isaacgym_imitator=True`
- 重力课程采用补偿力方案实现，物理等价于修改全局重力（因手 `disable_gravity=True`）
- `set_external_force_and_torque()` 是缓冲式 API，手腕力和物体补偿力分别对 `self.hand` 和 `self.object` 调用，互不覆盖
- 摩擦力课程通过 `root_physx_view.get/set_material_properties()` tensor API 实现，每 32 步更新一次
- DR 相关 PhysX views 在首步之前不可用，需延迟初始化（`_dr_initialized` flag）

---

## 第二轮逐行对比发现的问题（问题 15-26）

以下问题通过逐行对比 IsaacLab 迁移代码与原版 ManipTrans（IsaacGym）代码发现，涉及 `dexhand_manip_env.py` 和 `dexhand_imitator_env.py` 两个文件。

### 问题 15（严重）：ROBOT_HEIGHT 值错误

**文件**: `dexhand_manip_env.py`, `dexhand_imitator_env.py`

**原代码**: `ROBOT_HEIGHT = 0.15`

**正确值**: `ROBOT_HEIGHT = 0.00214874`（来自原版 `main/cfg/config.py`）

**后果**: 手腕高度偏差约 148mm，导致手在桌面上方过高位置初始化，与 demo 轨迹不匹配。所有依赖手腕高度的计算（wrist_pos 观测、reset 位置等）均受影响。

**修复**: 两个文件均改为 `ROBOT_HEIGHT = 0.00214874`。

### 问题 16（中等）：桌子 X 方向偏移缺失

**文件**: `dexhand_manip_env.py`, `dexhand_imitator_env.py`

**原代码**: `translation=(0.0, 0.0, 0.4)`

**正确值**: `translation=(-0.1, 0.0, 0.4)`（原版 `table_width_offset / 2 = -0.1`）

**后果**: 桌子中心相对于手的位置偏移了 10cm，影响物体放置位置和手-桌交互。

**修复**: 两个文件均改为 `translation=(-0.1, 0.0, 0.4)`。

### 问题 17（中等）：桌面摩擦力错误

**文件**: `dexhand_manip_env.py`, `dexhand_imitator_env.py`

**原代码**: 桌子使用全局 `physics_material`（摩擦力 4.0，与手相同）

**正确值**: 原版桌面摩擦力为 **0.1**，远低于手的 4.0

**后果**: 桌面摩擦力过高（40 倍）会导致物体在桌面上难以滑动，影响推/拉操作的物理行为。

**修复**: 在 CuboidCfg 中设置独立的 `physics_material`:
```python
spawn=sim_utils.CuboidCfg(
    ...
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=0.1,
        dynamic_friction=0.1,
        restitution=0.0,
    ),
)
```

### 问题 18（致命）：`_apply_action` 被调用 `decimation` 次导致双重移动平均

**文件**: `dexhand_manip_env.py`, `dexhand_imitator_env.py`

**原代码**: 在 `_apply_action()` 中计算 DOF targets、moving average、wrist force/torque。

**原版行为**: 原版 ManipTrans 的 `pre_physics_step()` 每个策略步仅调用一次，在其中完成所有计算。

**IsaacLab 差异**: IsaacLab 的 `_apply_action()` 每个策略步被调用 `decimation` 次（默认 2 次），而 `_pre_physics_step()` 仅被调用 1 次。

**后果**:
- Moving average 被应用 2 次：`target = α*action + (1-α)*prev`，第二次又混合一次，导致动作过度平滑
- 手腕力 PD 控制器被执行 2 次，目标位置被更新 2 次
- 关节目标也被设置 2 次

**修复**: 将所有计算逻辑从 `_apply_action()` 移到 `_pre_physics_step()`（仅执行 1 次），`_apply_action()` 只负责将预计算的值写入仿真：
```python
def _pre_physics_step(self, actions):
    self.actions = actions.clone().clamp(-1.0, 1.0)
    # 所有计算：moving average, DOF target scaling, force/torque computation
    ...

def _apply_action(self):
    # 仅写入仿真
    self.hand.set_joint_position_target(self.curr_targets)
    self.hand.set_external_force_and_torque(...)
```

### 问题 19（严重）：力缩放使用错误的时间步长

**文件**: `dexhand_manip_env.py`, `dexhand_imitator_env.py`

**原代码**: 力/力矩计算中使用 `self.physics_dt`（= 1/120）

**正确值**: 应使用 `self.step_dt`（= physics_dt × decimation = 1/60），对应原版的 `self.dt`

**原版代码**:
```python
# 原版 ManipTrans (IsaacGym)
force = (self.kp * p_error + self.kd * d_error) / self.dt  # self.dt = 1/60
```

**后果**: 使用 1/120 而非 1/60，力被缩放了 2 倍，导致手腕力控制过强，可能引起手腕振荡或不稳定。

**修复**: 两个文件均改为 `self.step_dt`：
```python
force = (Kp * pos_error + Kd * vel_error) / self.step_dt
```

### 问题 20（中等）：DOF 速度在 reset 时未夹紧到 URDF 限位

**文件**: `dexhand_manip_env.py`, `dexhand_imitator_env.py`

**原代码**: reset 时直接从 demo 加载 DOF 速度，不做限位检查。

**原版行为**: reset 时会将 DOF 速度夹紧到 URDF 中定义的 `velocity` 限位。

**后果**: 如果 demo 数据中某些关节速度超过 URDF 定义的物理限位，PhysX 可能产生异常行为。

**修复**:
1. 在 `_init_buffers()` 中从 URDF 解析每个关节的 `velocity` 限位并存储为 `dof_speed_limits`
2. 在 `_reset_idx()` 中夹紧：
```python
speed_limits_demo = self.dof_speed_limits[self.demo_to_isaaclab_dof_mapping]
dof_vel = torch.clamp(dof_vel_demo, -speed_limits_demo, speed_limits_demo)
```

### 问题 21（中等）：reset 后 PD 控制器初始目标不匹配

**文件**: `dexhand_manip_env.py`, `dexhand_imitator_env.py`

**原代码**: reset 时设置 `dof_pos` 但未设置对应的位置目标（position target）。

**后果**: PD 控制器在 reset 后的第一帧使用旧的目标位置（或默认 0），与新的 dof_pos 不匹配，产生瞬间大力矩导致手抖动。

**修复**: reset 时同步设置位置目标：
```python
self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)  # 新增
```

### 问题 22（严重）：本体感觉观测中 base 位置未清零

**文件**: `dexhand_manip_env.py`（仅残差环境）

**原代码**: 直接使用 `hand_root_state[:, :13]` 作为 base_state（包含 xyz 位置）。

**原版行为**: base_state 的前 3 个分量（位置 xyz）被清零：
```python
base_state = self.base_state.clone()
base_state[:, :3] = 0  # 清零位置
```

**原因**: 手腕绝对位置信息已通过 target 观测中的 `delta_wrist_pos` 隐式提供（相对当前位置的目标差值），保留绝对位置会引入与环境坐标系相关的偏差。

**修复**:
```python
zeroed_root_state = self.hand_root_state[:, :13].clone()
zeroed_root_state[:, :3] = 0  # 清零位置，保留姿态/速度
```

### 问题 23（严重）：特权观测 (privileged obs) 内容不完整

**文件**: `dexhand_manip_env.py`（仅残差环境）

**原代码**: 特权观测仅包含 `hand_dof_vel`（n_dofs 维）。

**原版 ManipTrans 特权观测**（对应 `spaces.Dict` 中的 `privileged` key）:

| 分量 | 维度 | 说明 |
|------|------|------|
| `dq`（DOF 速度） | n_dofs | 关节角速度 |
| `obj_pos_rel` | 3 | 物体相对手腕位置 |
| `obj_quat` | 4 | 物体四元数 |
| `obj_vel` | 3 | 物体线速度 |
| `obj_ang_vel` | 3 | 物体角速度 |
| `tip_force_3d` | n_tips×3 | 指尖 3D 接触力 |
| `tip_force_mag` | n_tips×1 | 指尖力大小 |
| `obj_com_rel` | 3 | 物体质心相对手腕位置 |
| `obj_weight` | 1 | 物体重量（mass × gravity_scale × 9.81） |

**以 Inspire 手为例**: n_dofs=12, n_tips=5 → 特权观测维度 = 12+3+4+3+3+15+5+3+1 = **49**

**后果**: 缺失物体状态和接触力信息使得残差策略无法感知：
- 物体当前位姿和速度
- 指尖是否接触物体
- 当前有效重力大小

**修复**: 补全所有特权观测分量，并更新 `num_observations` 计算。

### 问题 24（严重）：`obj_to_joints` 距离计算使用错误的 body 集合

**文件**: `dexhand_manip_env.py`（仅残差环境）

**原代码**: 使用 5 个指尖位置 `self.fingertip_pos` 计算物体到关节的距离。

**原版行为**: 使用**所有** body 位置（n_bodies 个，Inspire=18）计算物体到每个 body 的距离。

**后果**:
- 目标观测中的 `obj_to_joints` 维度错误：5×3=15 vs 应为 18×3=54
- 只关注指尖而忽略手掌和近端关节，丢失了物体与手掌接触的重要信息

**修复**: 使用所有 body 位置：
```python
all_hand_body_pos = self.hand_body_pos_all[:, self.demo_body_to_isaaclab_indices]
obj_to_joints = (manip_obj_pos.unsqueeze(1) - all_hand_body_pos)
```

### 问题 25（中等）：`obs_future_length` 默认值错误

**文件**: `dexhand_manip_env.py`（仅残差环境）

**原代码**: `obs_future_length: int = 3`

**正确值**: `obs_future_length: int = 1`（原版默认值）

**后果**: 3 帧未来观测使得目标观测维度膨胀 3 倍，增加网络输入维度和计算量，且与原版训练配置不一致。

### 问题 26（中等）：`tighten_method` 默认值错误

**文件**: `dexhand_manip_env.py`（仅残差环境）

**原代码**: `tighten_method: str = "None"`

**正确值**: `tighten_method: str = "exp_decay"`（原版默认值）

**后果**: 课程学习未启用，训练初期终止阈值不会逐步收紧。原版使用指数衰减：
```python
scale = (e**2)**(-step/tighten_steps) * (1 - tighten_factor) + tighten_factor
```
scale 从 1.0 衰减到 `tighten_factor`（0.7），逐步提高跟踪精度要求。

---

## 第二轮修改的文件

1. `maniptrans_envs/tasks/dexhand_manip_env.py` — 修复 ROBOT_HEIGHT、桌子偏移/摩擦力、_pre_physics_step/_apply_action 拆分、力缩放 step_dt、DOF 速度夹紧、reset 位置目标、base 位置清零、特权观测补全、obj_to_joints 全 body、obs_future_length、tighten_method
2. `maniptrans_envs/tasks/dexhand_imitator_env.py` — 修复 ROBOT_HEIGHT、桌子偏移/摩擦力、_pre_physics_step/_apply_action 拆分、力缩放 step_dt、DOF 速度夹紧、reset 位置目标

## 第二轮验证方法

1. **观测维度检查**: 启动时会打印 `[INFO] Manip obs dims: prop=X, priv=X, target=X, total=X`，对照原版检查各部分维度
2. **力缩放检查**: 确认 `step_dt` 打印值为 ~0.01667（1/60），而非 0.00833（1/120）
3. **Reset 检查**: 确认 reset 后第一帧手不会抖动（PD 目标与 dof_pos 匹配）
4. **桌面交互**: 确认物体在桌面上可以正常滑动（摩擦力 0.1 而非 4.0）
5. **课程学习**: 确认 tighten_method="exp_decay" 时 scale_factor 随步数递减
