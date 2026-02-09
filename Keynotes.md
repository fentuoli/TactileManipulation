# ManipTrans Migration Keynotes

## Migration Overview
- **Source**: ManipTrans-old (Python 3.8 + IsaacGym Preview 4)
- **Target**: ManipTrans-new (Python 3.11 + IsaacLab/IsaacSim 5.1.0)
- **Conda Environment**: `env_isaacsim`
- **IsaacLab Location**: `~/Code/IsaacLab`

## Key Migration Changes

### 1. Environment Structure Changes
- Move from flat task directories to structured folders
- Configuration: YAML configs -> Python dataclasses with `@configclass` decorator
- Need `agents/` subdirectory for RL configs
- Need `__init__.py` for Gymnasium environment registration

### 2. API Changes (gymapi -> IsaacLab)
| IsaacGym | IsaacLab |
|----------|----------|
| `gym.create_sim()` | `_setup_scene()` method |
| `gym.add_ground()` | `spawn_ground_plane()` with `GroundPlaneCfg` |
| `gym.create_actor()` | `Articulation` objects from configs |
| `gym.get_asset_dof_properties()` | `ImplicitActuatorCfg` |
| `gymtorch.wrap_tensor()` | Direct access via `asset.data.joint_pos` |

### 3. Method Renames
| IsaacGym | IsaacLab |
|----------|----------|
| `pre_physics_step()` | `_pre_physics_step()` + `_apply_action()` |
| `compute_observations()` | `_get_observations()` |
| `compute_reward()` | `_get_rewards()` |
| `compute_dones()` | `_get_dones()` |

### 4. URDF Conversion
Use `UrdfConverter` from `isaaclab.sim.converters`:
```python
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
cfg = UrdfConverterCfg(asset_path="/path/to/robot.urdf")
converter = UrdfConverter(cfg)
usd_path = converter.usd_path
```

### 5. Critical Differences
- **Quaternion convention**: Both use wxyz, but Isaac Gym Preview used xyzw - need conversions
- **Joint ordering**: IsaacSim uses breadth-first, IsaacGym used depth-first

### 6. pytorch3d Replacement
- `chamfer_distance` already installed in env_isaacsim
- Use `chamfer_distance` package for Chamfer distance calculations
- For other mesh operations, consider `trimesh` or `open3d`

## Files to Migrate (Single Hand Mode Only)
- `maniptrans_envs/lib/envs/core/vec_task.py`
- `maniptrans_envs/lib/envs/tasks/dexhandmanip_sh.py`
- `maniptrans_envs/lib/envs/tasks/dexhandimitator.py`
- `main/dataset/mano2dexhand.py`
- `main/rl/train.py`

## Files NOT to Migrate
- `DexManipNet/dexmanip_bih.py` (bimanual mode, per instructions)
- `maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py`

## Progress Log
- [x] Created ManipTrans-new directory
- [x] Copied files from ManipTrans-old
- [x] Researched IsaacLab migration guide
- [x] Researched UrdfConverter API
- [x] Install required packages in env_isaacsim
- [x] Replace pytorch3d usage
  - Added custom rotation conversion functions in `main/dataset/transform.py`
  - Replaced `pytorch3d.ops.sample_points_from_meshes` with `trimesh.sample.sample_surface`
  - Updated `main/dataset/base.py`, `grab_dataset_dexhand.py`, `oakink2_dataset_dexhand_rh.py`, `oakink2_dataset_dexhand_lh.py`
- [x] Migrate environment core code (COMPLETED)
  - Created `maniptrans_envs/tasks/dexhand_manip_env.py` - IsaacLab-based environment
  - Migrated VecTask to DirectRLEnv
  - Created configuration classes with @configclass decorator
  - Copied dexhands module and utilities
- [x] Migrate preprocessing script (mano2dexhand.py)
  - Removed IsaacGym dependencies
  - Rewrote using pure PyTorch + pytorch_kinematics for forward kinematics
  - No simulation needed for retargeting optimization
- [x] Update URDF conversion to use IsaacLab UrdfConverter ✓
  - Not needed separately - `UrdfFileCfg` in `ArticulationCfg` handles conversion automatically
  - Preprocessing uses `pytorch_kinematics` directly (no USD needed)
- [x] Test preprocessing command ✓
  - Command: `PYTHONPATH=. python main/dataset/mano2dexhand.py --data_idx g0 --dexhand inspire --iter 100`
  - Output verified: data/retargeting/grab_demo/mano2inspire_rh/102_sv_dict.pkl
  - Contains: opt_wrist_pos, opt_wrist_rot, opt_dof_pos, opt_joints_pos
- [x] Migrate training script (main/rl/train.py) ✓
  - Rewrote to use IsaacLab's AppLauncher pattern
  - Uses RlGamesVecEnvWrapper for rl-games integration
  - Registers environment via gymnasium
  - Created `maniptrans_envs/tasks/agents/rl_games_ppo_cfg.yaml`
- [x] Test training command ✓
  - Isaac Sim starts successfully
  - URDF converts to USD automatically
  - Environment initializes and loads demo data
  - rl_games training loop starts
  - Command: `PYTHONPATH=. python main/rl/train.py --task ManipTrans-DexHand-RH-Direct-v0 --num_envs 512 --headless`
  - Note: May need further debugging for full training convergence

## Environment Migration Details

### New File Structure
```
ManipTrans-new/
├── maniptrans_envs/
│   ├── __init__.py
│   ├── tasks/
│   │   ├── __init__.py               # Includes gymnasium registration
│   │   ├── dexhand_manip_env.py      # Main IsaacLab DirectRLEnv
│   │   └── agents/
│   │       ├── __init__.py
│   │       └── rl_games_ppo_cfg.yaml # RL-Games PPO config
│   └── lib/
│       ├── envs/
│       │   └── dexhands/             # Copied from old repo
│       └── utils/                    # Copied from old repo
├── main/
│   ├── dataset/                      # Updated for pytorch3d replacement
│   └── rl/
│       └── train.py                  # New IsaacLab training script
└── Keynotes.md
```

### Key API Mappings Used
```python
# IsaacGym -> IsaacLab
VecTask -> DirectRLEnv
gymapi.create_env() -> InteractiveSceneCfg
gymtorch.wrap_tensor() -> self.hand.data.joint_pos (direct access)
gym.load_asset() -> ArticulationCfg with UrdfFileCfg
pre_physics_step() -> _pre_physics_step() + _apply_action()
compute_observations() -> _get_observations()
compute_reward() -> _get_rewards()
reset_idx() -> _reset_idx()
```

### Remaining Work
1. ~~Test URDF loading with IsaacLab's UrdfFileCfg (for RL training)~~ ✓ Working
2. Debug any remaining runtime issues in `_get_observations()`, `_get_rewards()`, `_reset_idx()`
3. Tune hyperparameters for training convergence
4. Handle dynamic object loading (per-environment objects) if needed for diverse training

### Known Issues / Notes
- ~~URDF fingertip links are being merged during conversion~~ ✓ Fixed: Added `merge_fixed_joints=False` to UrdfFileCfg
- ~~Need `merge_fixed_joints: False` in UrdfFileCfg if fingertip tracking is needed~~ ✓ Fixed
- ~~Observation dimension mismatch~~ ✓ Fixed: Corrected obs dim calculation (obj_to_joints=5, tips inside multiplier)
- ~~Action dimension mismatch~~ ✓ Fixed: Doubled action space for base + residual (36 dims)
- ~~sim_dt attribute missing~~ ✓ Fixed: Changed to `physics_dt` (IsaacLab API)
- ~~Object mesh: placeholder cube~~ ✓ Fixed: Object URDF converted to USD via UrdfConverter, then loaded as RigidObject
- ~~Hand orientation wrong~~ ✓ Fixed: Removed incorrect base rotation - wxyz quaternions used directly without extra transforms
- ~~Hand flying into sky~~ ✓ Fixed: Added damping settings (linear_damping=20, angular_damping=20) matching original
- Added dynamic body/joint index computation from IsaacLab articulation (handles body reordering)
- ~~DOF ordering mismatch~~ ✓ Fixed: Added DOF index mapping between demo order (depth-first) and IsaacLab order (breadth-first)
- Joint limits are set to default values; may need calibration from URDF

### Quaternion/Rotation Handling
- IsaacLab uses wxyz quaternion format (same as pytorch3d/scipy)
- IsaacGym used xyzw quaternion format - original code converted wxyz→xyzw
- Demo data `opt_wrist_rot` is in axis-angle format, converted to wxyz via `aa_to_quat()`
- For IsaacLab, we use wxyz directly without any additional rotation:
  ```python
  opt_wrist_rot = aa_to_quat(self.demo_data["opt_wrist_rot"][...])  # Already in wxyz, no conversion needed
  ```

### DOF Ordering
- IsaacLab may reorder joints during URDF conversion (breadth-first vs depth-first)
- Demo data DOF order follows `dexhand.dof_names` (depth-first)
- Dynamic mapping computed at runtime in `_compute_body_indices()`:
  ```python
  # Map from demo order to IsaacLab order
  demo_to_isaaclab_dof_mapping[demo_idx] = isaaclab_idx
  # Map from IsaacLab order to demo order (for observations)
  isaaclab_to_demo_dof_mapping[isaaclab_idx] = demo_idx
  ```
- Actions: Policy outputs DOF targets in demo order → reorder to IsaacLab order using `scatter_()`
- Observations: IsaacLab DOF positions → reorder to demo order for policy using gather indexing

### Joint Limits (CRITICAL)
Joint limits must be in IsaacLab order and match URDF values. Wrong limits cause incorrect joint positions:
```python
# IsaacLab order for Inspire hand:
# [index_prox, middle_prox, pinky_prox, ring_prox, thumb_yaw,
#  index_inter, middle_inter, pinky_inter, ring_inter,
#  thumb_pitch, thumb_inter, thumb_distal]
lower = [0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
upper = [1.7, 1.7, 1.7, 1.7, 1.3, 1.7, 1.7, 1.7, 1.7, 0.5, 0.8, 1.2]
```

## Key Changes Summary

### Preprocessing (mano2dexhand.py)
- **Original**: Used IsaacGym for simulation + pytorch_kinematics for FK
- **New**: Pure PyTorch + pytorch_kinematics only
- **Benefit**: No simulation environment needed, faster execution, simpler dependencies

### Environment (dexhand_manip_env.py)
- **Original**: VecTask (IsaacGym) with gymapi/gymtorch
- **New**: DirectRLEnv (IsaacLab) with isaaclab.sim/assets
- **Key Changes**:
  - Configuration via `@configclass` decorators
  - Scene setup via `_setup_scene()` method
  - Direct tensor access via `self.hand.data.joint_pos` (no wrap_tensor)
  - Observations via `_get_observations()` returning dict with "policy" key
  - Rewards via `_get_rewards()` returning tensor
  - Reset via `_reset_idx()` with IsaacLab's write_*_to_sim methods

### Training Script (main/rl/train.py)
- **Original**: Hydra-based config with `isaacgym` import, custom rl_games wrappers
- **New**: IsaacLab AppLauncher pattern with built-in rl_games integration
- **Key Changes**:
  - Uses `AppLauncher` to initialize Isaac Sim before imports
  - Registers environment via `gymnasium.register()`
  - Uses `RlGamesVecEnvWrapper` from `isaaclab_rl`
  - Uses argparse instead of Hydra (IsaacLab convention)

**Original Command (ManipTrans-old with Hydra):**
```bash
python main/rl/train.py task=ResDexHand dexhand=inspire side=RH headless=true num_envs=4096 \
    learning_rate=2e-4 test=false randomStateInit=true dataIndices=[g0] \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    actionsMovingAverage=0.4 experiment=cross_g0_inspire
```

**New Command (ManipTrans-new with IsaacLab):**
```bash
PYTHONPATH=. python main/rl/train.py --dexhand inspire --side right --num_envs 4096 \
    --learning_rate 2e-4 --random_state_init --data_indices g0 \
    --rh_base_checkpoint assets/imitator_rh_inspire.pth \
    --actions_moving_average 0.4 --experiment cross_g0_inspire --headless
```

### Dataset (transform.py, base.py, etc.)
- Replaced pytorch3d with custom implementations
- Uses trimesh for mesh point sampling
- All rotation conversion functions now in pure PyTorch
