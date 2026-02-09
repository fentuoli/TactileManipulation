"""
DexHand Manipulation Environment for IsaacLab.

Migrated from ManipTrans IsaacGym implementation to IsaacLab/IsaacSim 5.1.0.
"""

from __future__ import annotations

import os
import torch
import numpy as np
from collections.abc import Sequence
from dataclasses import MISSING
from typing import Dict, List, Tuple, Any

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, UrdfFileCfg, UsdFileCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_mul, sample_uniform, saturate

from bps_torch.bps import bps_torch
from tqdm import tqdm

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_aa, rotmat_to_quat, rot6d_to_aa


# Quaternion format conversion helpers
# IsaacLab uses wxyz format, but the pre-trained model was trained with IsaacGym's xyzw format
def wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from wxyz format to xyzw format."""
    return quat[..., [1, 2, 3, 0]]

def xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from xyzw format to wxyz format."""
    return quat[..., [3, 0, 1, 2]]

def quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication for xyzw format (IsaacGym convention)."""
    # Extract components (xyzw format)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    # Hamilton product
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([x, y, z, w], dim=-1)

def quat_conjugate_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for xyzw format."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


# Configuration constants
ROBOT_HEIGHT = 0.15


@configclass
class DexHandManipEnvCfg(DirectRLEnvCfg):
    """Configuration for the DexHand manipulation environment."""

    # Environment settings
    # decimation=1 means 1 physics step per action (control_freq_inv=1 in original)
    # The original IsaacGym uses dt=1/60 with substeps=2, so effective physics dt = 1/120
    # With decimation=1 and dt=1/60, we get same control frequency
    decimation = 1
    episode_length_s = 20.0  # Will be overridden by max_episode_length

    # Action/Observation space (default values for Inspire hand with 12 DOFs)
    # Will be recalculated in __init__ based on actual hand configuration
    action_space = 36  # (6 wrist + 12 dofs) * 2 (base + residual) for inspire
    observation_space = 794  # Computed dynamically in __init__ based on dexhand configuration
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        # Original IsaacGym: dt=1/60 with substeps=2 -> effective physics dt = 1/120
        # In IsaacLab, we set dt directly to the physics timestep
        # With decimation=1, control frequency matches physics frequency
        dt=1 / 60,  # Physics timestep - IsaacLab handles substeps internally
        render_interval=1,
        physics_material=RigidBodyMaterialCfg(
            # Original code: hand friction=2.0, object friction=4.0
            # Use high friction for better grasping
            static_friction=3.0,
            dynamic_friction=3.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            # Note: In IsaacLab, solver iterations are set per-actor in ArticulationCfg,
            # not globally here. These are min/max bounds for the scene.
            min_position_iteration_count=4,
            max_position_iteration_count=255,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=255,
        ),
        gravity=(0.0, 0.0, -9.81),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
    )

    # Task-specific configuration
    dexhand_type: str = "inspire"
    side: str = "right"  # "right" or "left"
    data_indices: List[str] | None = None  # Will use default ["g0"] if None

    # Action/Observation settings
    use_quat_rot: bool = True
    action_scale: float = 1.0
    act_moving_average: float = 0.9
    translation_scale: float = 1.0
    orientation_scale: float = 1.0
    use_pid_control: bool = False

    # Training settings
    training: bool = True
    obs_future_length: int = 3
    rollout_state_init: bool = False
    random_state_init: bool = True

    # Tightening curriculum
    tighten_method: str = "None"  # "None", "const", "linear_decay", "exp_decay", "cos"
    tighten_factor: float = 0.7
    tighten_steps: int = 100000

    # Rollout settings (optional)
    rollout_len: int | None = None
    rollout_begin: int | None = None

    # Max episode length
    max_episode_length: int = 1000


@configclass
class DexHandManipRHEnvCfg(DexHandManipEnvCfg):
    """Configuration for right-hand manipulation."""
    side: str = "right"


@configclass
class DexHandManipLHEnvCfg(DexHandManipEnvCfg):
    """Configuration for left-hand manipulation."""
    side: str = "left"


class DexHandManipEnv(DirectRLEnv):
    """
    DexHand Manipulation Environment using IsaacLab.

    This environment implements dexterous hand manipulation for imitation learning,
    tracking reference trajectories from motion capture data.
    """

    cfg: DexHandManipEnvCfg

    def __init__(self, cfg: DexHandManipEnvCfg, render_mode: str | None = None, **kwargs):
        # Initialize dexhand before calling super().__init__
        self.dexhand = DexHandFactory.create_hand(cfg.dexhand_type, cfg.side)
        self.side = cfg.side

        # Pre-load object URDF path for scene setup (before super().__init__)
        self.data_indices = cfg.data_indices if cfg.data_indices is not None else ["g0"]
        self._preload_object_info()

        # Calculate action and observation space dimensions
        n_dofs = self.dexhand.n_dofs
        use_quat_rot = cfg.use_quat_rot
        # Action space: base + residual, each has (wrist control + finger DOFs)
        # For non-quat: 6 (wrist force/torque) + n_dofs (fingers) for each
        # For quat: 1 (flag?) + 6 (wrist) + n_dofs (fingers) for each
        single_action_dim = (1 + 6 + n_dofs) if use_quat_rot else (6 + n_dofs)
        cfg.num_actions = single_action_dim * 2  # base + residual

        # Target observation dimension
        n_fingertips = 5  # Number of fingertips for obj_to_joints
        n_joint_bodies = self.dexhand.n_bodies - 1  # Bodies excluding wrist for joint tracking
        target_obs_dim = (
            128  # BPS features
            + n_fingertips  # obj_to_joints (NOT multiplied by obs_future_length)
            + (
                3   # delta_wrist_pos
                + 3   # wrist_vel
                + 3   # delta_wrist_vel
                + 4   # wrist_quat
                + 4   # delta_wrist_quat
                + 3   # wrist_ang_vel
                + 3   # delta_wrist_ang_vel
                + n_joint_bodies * 9  # delta_joints_pos + joints_vel + delta_joints_vel (17*9=153)
                + 3   # delta_manip_obj_pos
                + 3   # manip_obj_vel
                + 3   # delta_manip_obj_vel
                + 4   # manip_obj_quat
                + 4   # delta_manip_obj_quat
                + 3   # manip_obj_ang_vel
                + 3   # delta_manip_obj_ang_vel
                + n_fingertips  # gt_tips_distance (per frame)
            )
            * cfg.obs_future_length
        )

        # Proprioception dimension: q, cos_q, sin_q, base_state
        prop_obs_dim = n_dofs * 3 + 13  # q, cos(q), sin(q), base_state

        cfg.num_observations = prop_obs_dim + target_obs_dim
        cfg.num_states = 0  # No asymmetric states for now

        super().__init__(cfg, render_mode, **kwargs)

        # Store configuration
        self.use_quat_rot = cfg.use_quat_rot
        self._max_episode_length_cfg = cfg.max_episode_length  # Store config value separately
        self.action_scale = cfg.action_scale
        self.act_moving_average = cfg.act_moving_average
        self.translation_scale = cfg.translation_scale
        self.orientation_scale = cfg.orientation_scale
        self.use_pid_control = cfg.use_pid_control
        self.training = cfg.training
        self.obs_future_length = cfg.obs_future_length
        self.rollout_state_init = cfg.rollout_state_init
        self.random_state_init = cfg.random_state_init
        self.tighten_method = cfg.tighten_method
        self.tighten_factor = cfg.tighten_factor
        self.tighten_steps = cfg.tighten_steps
        self.rollout_len = cfg.rollout_len
        self.rollout_begin = cfg.rollout_begin
        self.data_indices = cfg.data_indices if cfg.data_indices is not None else ["g0"]

        # Initialize buffers
        self._init_buffers()

        # Load demo data
        self._load_demo_data()

        # Initialize BPS layer
        self.bps_feat_type = "dists"
        self.bps_layer = bps_torch(
            bps_type="grid_sphere",
            n_bps_points=128,
            radius=0.2,
            randomize=False,
            device=self.device,
        )

        # Compute BPS features for object
        obj_verts = self.demo_data["obj_verts"]
        self.obj_bps = self.bps_layer.encode(obj_verts, feature_type=self.bps_feat_type)[self.bps_feat_type]

        # Default DOF positions
        default_pose = torch.ones(self.dexhand.n_dofs, device=self.device) * np.pi / 12
        if cfg.dexhand_type == "inspire":
            default_pose[8] = 0.3
            default_pose[9] = 0.01
        self.dexhand_default_dof_pos = default_pose

    def _preload_object_info(self):
        """Pre-load object URDF path before scene setup.

        This is called before super().__init__() to get the object mesh path
        for spawning in _setup_scene().
        """
        from main.dataset.factory import ManipDataFactory
        import os

        # Get the first data index to determine object
        first_idx = self.data_indices[0]
        dataset_type = ManipDataFactory.dataset_type(first_idx)

        # Get the project root directory (maniptrans_envs/../)
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(this_file_dir, "../.."))

        # Create a minimal dataset instance just to get object path
        # We don't need full data loading here, just the URDF path
        if dataset_type in ("grab_demo", "grabdemo"):
            # GRAB demo uses a fixed object - find from working directory
            # Try multiple possible locations with the project root
            possible_paths = [
                os.path.join(project_root, "data/grab_demo/102/obj.urdf"),
                os.path.join(os.getcwd(), "data/grab_demo/102/obj.urdf"),
                os.path.join(this_file_dir, "../../../data/grab_demo/102/obj.urdf"),
                "/home/bore/Code/ManipTrans-new/data/grab_demo/102/obj.urdf",
            ]
            self._object_urdf_path = None
            for path in possible_paths:
                resolved_path = os.path.abspath(path)
                if os.path.exists(resolved_path):
                    self._object_urdf_path = resolved_path
                    break
        else:
            # For OakInk2, we'd need to look up the object path
            # For now, use a placeholder - this should be extended for OakInk2
            self._object_urdf_path = None

        print(f"[INFO] Pre-loaded object URDF path: {self._object_urdf_path}")

    def _compute_body_indices(self):
        """Compute body indices based on actual IsaacLab body names.

        IsaacLab may reorder bodies during URDF conversion. This method
        creates a mapping from the expected body names to actual indices.
        """
        # Get actual body names from IsaacLab articulation
        actual_body_names = self.hand.body_names
        actual_joint_names = self.hand.joint_names
        print(f"[INFO] IsaacLab num_bodies: {self.hand.num_bodies}")
        print(f"[INFO] IsaacLab num_joints: {self.hand.num_joints}")
        print(f"[INFO] IsaacLab body names ({len(actual_body_names)}): {actual_body_names}")
        print(f"[INFO] IsaacLab joint names ({len(actual_joint_names)}): {actual_joint_names}")
        print(f"[INFO] Expected body names ({len(self.dexhand.body_names)}): {self.dexhand.body_names}")
        print(f"[INFO] Expected joint names ({len(self.dexhand.dof_names)}): {self.dexhand.dof_names}")
        print(f"[INFO] Expected n_dofs: {self.dexhand.n_dofs}")

        # Create mapping from expected names to actual indices
        self.body_name_to_idx = {}
        for i, name in enumerate(actual_body_names):
            self.body_name_to_idx[name] = i

        # Create DOF index mapping - use identity (no reordering)
        self.joint_name_to_idx = {}
        for i, name in enumerate(actual_joint_names):
            self.joint_name_to_idx[name] = i

        # Find fingertip indices in actual ordering
        # Fingertip names from dexhand definition
        fingertip_names = []
        prefix = "R_" if self.side == "right" else "L_"
        for tip_suffix in ["index_tip", "middle_tip", "pinky_tip", "ring_tip", "thumb_tip"]:
            fingertip_names.append(prefix + tip_suffix)

        self.fingertip_indices = []
        for name in fingertip_names:
            if name in self.body_name_to_idx:
                self.fingertip_indices.append(self.body_name_to_idx[name])
            else:
                print(f"[WARNING] Fingertip body '{name}' not found in articulation!")
                print(f"[WARNING] Available bodies: {actual_body_names}")

        print(f"[INFO] Computed fingertip indices: {self.fingertip_indices}")

        # Find wrist body index for force/torque application
        wrist_body_names = self.dexhand.to_dex("wrist")  # e.g., ["R_hand_base_link"] for right Inspire
        self.wrist_body_idx = None
        for wrist_name in wrist_body_names:
            if wrist_name in self.body_name_to_idx:
                self.wrist_body_idx = self.body_name_to_idx[wrist_name]
                print(f"[INFO] Wrist body '{wrist_name}' found at index {self.wrist_body_idx}")
                break
        if self.wrist_body_idx is None:
            # Fallback: assume wrist is the first body (root)
            self.wrist_body_idx = 0
            print(f"[WARNING] Wrist body not found, defaulting to body index 0")
            print(f"[WARNING] Available bodies: {actual_body_names}")

        # Update the actual number of bodies and joints
        self._actual_num_bodies = len(actual_body_names)
        self._actual_num_joints = len(actual_joint_names)

        # Verify joint count matches expected
        if self._actual_num_joints != self.dexhand.n_dofs:
            print(f"[WARNING] Joint count mismatch! IsaacLab has {self._actual_num_joints} joints, expected {self.dexhand.n_dofs}")
            print(f"[WARNING] This may cause indexing errors. Demo data assumes {self.dexhand.n_dofs} DOFs.")

    def _init_buffers(self):
        """Initialize tensor buffers."""
        # Compute fingertip indices based on actual body names from IsaacLab
        # This is needed because URDF conversion may reorder bodies
        self._compute_body_indices()

        # Use actual joint count from IsaacLab articulation
        num_joints = self._actual_num_joints if hasattr(self, '_actual_num_joints') else self.dexhand.n_dofs

        # Read joint limits from articulation (parsed from URDF)
        if hasattr(self.hand.data, 'default_joint_pos_limits') and self.hand.data.default_joint_pos_limits is not None:
            self.dexhand_dof_lower_limits = self.hand.data.default_joint_pos_limits[0, :, 0].clone()
            self.dexhand_dof_upper_limits = self.hand.data.default_joint_pos_limits[0, :, 1].clone()
        elif hasattr(self.hand.data, 'soft_joint_pos_limits') and self.hand.data.soft_joint_pos_limits is not None:
            self.dexhand_dof_lower_limits = self.hand.data.soft_joint_pos_limits[0, :, 0].clone()
            self.dexhand_dof_upper_limits = self.hand.data.soft_joint_pos_limits[0, :, 1].clone()
        else:
            self.dexhand_dof_lower_limits = torch.full((num_joints,), 0.0, device=self.device, dtype=torch.float32)
            self.dexhand_dof_upper_limits = torch.full((num_joints,), 1.7, device=self.device, dtype=torch.float32)

        # Action-related buffers
        self.prev_targets = torch.zeros(
            (self.num_envs, num_joints), dtype=torch.float, device=self.device
        )
        self.curr_targets = torch.zeros(
            (self.num_envs, num_joints), dtype=torch.float, device=self.device
        )

        # Force application buffers - use actual number of bodies from IsaacLab
        num_bodies = self._actual_num_bodies if hasattr(self, '_actual_num_bodies') else self.dexhand.n_bodies
        self.apply_forces = torch.zeros(
            (self.num_envs, num_bodies, 3), dtype=torch.float, device=self.device
        )
        self.apply_torques = torch.zeros(
            (self.num_envs, num_bodies, 3), dtype=torch.float, device=self.device
        )

        # Progress tracking
        self.running_progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failure_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.error_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.total_rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Contact history
        CONTACT_HISTORY_LEN = 3
        self.tips_contact_history = torch.ones(
            self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device
        ).bool()

        # PID control buffers
        if self.use_pid_control:
            self.prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

            self.Kp_rot = self.dexhand.Kp_rot
            self.Ki_rot = self.dexhand.Ki_rot
            self.Kd_rot = self.dexhand.Kd_rot
            self.Kp_pos = self.dexhand.Kp_pos
            self.Ki_pos = self.dexhand.Ki_pos
            self.Kd_pos = self.dexhand.Kd_pos

        # Best rollout tracking (for curriculum)
        self.best_rollout_len = 0
        self.best_rollout_begin = 0

    def _load_demo_data(self):
        """Load demonstration data for imitation learning."""
        # Compute transformation matrix from MuJoCo to Gym coordinate frame
        table_surface_z = 0.4 + 0.015  # table_pos.z + table_half_height
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, table_surface_z])
        self.mujoco2gym_transf = torch.tensor(
            mujoco2gym_transf, device=self.device, dtype=torch.float32
        )

        # Get unique dataset types
        dataset_list = list(set([ManipDataFactory.dataset_type(idx) for idx in self.data_indices]))

        # Create dataset instances
        self.demo_dataset_dict = {}
        for dataset_type in dataset_list:
            self.demo_dataset_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side=self.side,
                device=self.device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=self.dexhand,
                embodiment=self.cfg.dexhand_type,
            )

        # Load data for each environment
        def segment_data(k):
            idx = self.data_indices[k % len(self.data_indices)]
            return self.demo_dataset_dict[ManipDataFactory.dataset_type(idx)][idx]

        demo_data_list = [segment_data(i) for i in tqdm(range(self.num_envs), desc="Loading demo data")]
        self.demo_data = self._pack_data(demo_data_list)

    def _pack_data(self, data: List[Dict]) -> Dict[str, torch.Tensor]:
        """Pack list of data dicts into batched tensors."""
        packed_data = {}
        packed_data["seq_len"] = torch.tensor(
            [len(d["obj_trajectory"]) for d in data], device=self.device
        )
        max_len = packed_data["seq_len"].max()
        assert max_len <= self.max_episode_length, "max_len should be less than max_episode_length"

        def fill_data(stack_data):
            for i in range(len(stack_data)):
                if len(stack_data[i]) < max_len:
                    stack_data[i] = torch.cat(
                        [
                            stack_data[i],
                            stack_data[i][-1]
                            .unsqueeze(0)
                            .repeat(max_len - len(stack_data[i]), *[1 for _ in stack_data[i].shape[1:]]),
                        ],
                        dim=0,
                    )
            return torch.stack(stack_data).squeeze()

        for k in data[0].keys():
            if k == "mano_joints" or k == "mano_joints_velocity":
                mano_joints = []
                for d in data:
                    mano_joints.append(
                        torch.concat(
                            [
                                d[k][self.dexhand.to_hand(j_name)[0]]
                                for j_name in self.dexhand.body_names
                                if self.dexhand.to_hand(j_name)[0] != "wrist"
                            ],
                            dim=-1,
                        )
                    )
                packed_data[k] = fill_data(mano_joints)
            elif isinstance(data[0][k], torch.Tensor):
                stack_data = [d[k] for d in data]
                if k != "obj_verts":
                    packed_data[k] = fill_data(stack_data)
                else:
                    packed_data[k] = torch.stack(stack_data).squeeze()
            elif isinstance(data[0][k], np.ndarray):
                raise RuntimeError("Using np is very slow.")
            else:
                packed_data[k] = [d[k] for d in data]

        return packed_data

    def _setup_scene(self):
        """Set up the simulation scene with robot and objects."""
        # Get URDF path for dexhand
        dexhand_urdf_path = self.dexhand.urdf_path
        urdf_dir, urdf_file = os.path.split(dexhand_urdf_path)

        # Create robot configuration
        # Note: In IsaacLab, we use USD files. URDF is converted to USD automatically.
        self.robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=UrdfFileCfg(
                asset_path=dexhand_urdf_path,
                fix_base=False,  # Free-floating hand, pose controlled via write_root_state_to_sim
                merge_fixed_joints=False,  # Keep fingertip links for tracking
                joint_drive=UrdfFileCfg.JointDriveCfg(
                    drive_type="force",
                    target_type="position",
                    gains=UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=500.0,
                        damping=30.0,
                    ),
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=10.0,
                    linear_damping=20.0,   # Prevent flying (same as original)
                    angular_damping=20.0,  # Prevent flying (same as original)
                    max_linear_velocity=50.0,
                    max_angular_velocity=100.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=self.dexhand.self_collision,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.4, 0.0, 0.4 + 0.015 + ROBOT_HEIGHT),
                # Quaternion from euler_zyx(0, -pi/2, 0): rotate -90 deg about Y axis
                # w = cos(-pi/4) ≈ 0.7071, x = 0, y = sin(-pi/4) ≈ -0.7071, z = 0
                rot=(0.7071, 0.0, -0.7071, 0.0),  # wxyz format
            ),
            actuators={
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=500.0,
                    damping=30.0,
                ),
            },
        )

        # Create the robot articulation
        self.hand = Articulation(self.robot_cfg)

        # Create table
        table_cfg = sim_utils.CuboidCfg(
            size=(1.0, 1.6, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        )
        table_cfg.func("/World/envs/env_.*/Table", table_cfg, translation=(0.0, 0.0, 0.4))

        # Spawn object from URDF if available
        # Note: Single-link URDFs without joints cannot be loaded as Articulations
        # We need to convert to USD first, then load as RigidObject
        self._object_is_articulation = False
        if hasattr(self, '_object_urdf_path') and self._object_urdf_path is not None and os.path.exists(self._object_urdf_path):
            print(f"[INFO] Loading object from URDF: {self._object_urdf_path}")
            # Convert URDF to USD first (no joint_drive needed for single-link object)
            urdf_converter_cfg = UrdfConverterCfg(
                asset_path=self._object_urdf_path,
                fix_base=False,
                joint_drive=None,  # No joints in this URDF
            )
            urdf_converter = UrdfConverter(urdf_converter_cfg)
            usd_path = urdf_converter.usd_path
            print(f"[INFO] Converted URDF to USD: {usd_path}")

            # Load as RigidObject using the converted USD
            # Original code: friction=4.0, mass capped at 0.5kg
            self.object_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object",
                spawn=UsdFileCfg(
                    usd_path=usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=10.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=0.1,  # Light object for easy grasping (original caps at 0.5kg)
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.005,
                        rest_offset=0.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
            )
            self.object = RigidObject(self.object_cfg)
        else:
            print(f"[WARNING] Object URDF not found, using placeholder cube")
            self.object_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object",
                spawn=sim_utils.CuboidCfg(
                    size=(0.05, 0.05, 0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=10.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=0.1,  # Light object
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.005,
                        rest_offset=0.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
            )
            self.object = RigidObject(self.object_cfg)

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Register to scene
        self.scene.articulations["robot"] = self.hand
        if self._object_is_articulation:
            self.scene.articulations["object"] = self.object
        else:
            self.scene.rigid_objects["object"] = self.object

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Note: Joint limits will be initialized in _init_buffers() after physics views are available

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        root_control_dim = 9 if self.use_pid_control else 6
        res_split_idx = (
            self.actions.shape[1] // 2
            if not self.use_pid_control
            else ((self.actions.shape[1] - (root_control_dim - 6)) // 2 + (root_control_dim - 6))
        )

        base_action = self.actions[:, :res_split_idx]
        residual_action = self.actions[:, res_split_idx:] * 2

        # DOF position targets
        dof_pos = (
            1.0 * base_action[:, root_control_dim:root_control_dim + self.dexhand.n_dofs]
            + residual_action[:, 6:6 + self.dexhand.n_dofs]
        )
        dof_pos = torch.clamp(dof_pos, -1, 1)

        # Scale to joint limits
        self.curr_targets = scale(
            dof_pos,
            self.dexhand_dof_lower_limits,
            self.dexhand_dof_upper_limits,
        )

        # Apply moving average
        self.curr_targets = (
            self.act_moving_average * self.curr_targets
            + (1.0 - self.act_moving_average) * self.prev_targets
        )

        # Clamp to limits
        self.curr_targets = saturate(
            self.curr_targets,
            self.dexhand_dof_lower_limits,
            self.dexhand_dof_upper_limits,
        )

        # Apply wrist forces/torques
        if self.use_pid_control:
            self._apply_pid_control(base_action, residual_action, root_control_dim)
        else:
            self._apply_force_control(base_action, residual_action)

        # Set joint position targets
        self.prev_targets[:] = self.curr_targets[:]
        self.hand.set_joint_position_target(self.curr_targets)

        # Apply external forces to wrist using IsaacLab API
        # set_external_force_and_torque expects shapes: (num_envs, len(body_ids), 3)
        # We're applying forces only to the wrist body (body index stored at self.wrist_body_idx)
        # Extract only the wrist forces/torques and reshape for single body
        wrist_forces = self.apply_forces[:, self.wrist_body_idx:self.wrist_body_idx+1, :]  # (num_envs, 1, 3)
        wrist_torques = self.apply_torques[:, self.wrist_body_idx:self.wrist_body_idx+1, :]  # (num_envs, 1, 3)

        self.hand.set_external_force_and_torque(
            forces=wrist_forces,
            torques=wrist_torques,
            body_ids=[self.wrist_body_idx],
            is_global=True,  # ENV_SPACE in original IsaacGym = world/environment frame
        )

    def _apply_pid_control(self, base_action, residual_action, root_control_dim):
        """Apply PID-based wrist control."""
        position_error = base_action[:, 0:3]
        self.pos_error_integral += position_error * self.physics_dt
        self.pos_error_integral = torch.clamp(self.pos_error_integral, -1, 1)
        pos_derivative = (position_error - self.prev_pos_error) / self.physics_dt
        force = (
            self.Kp_pos * position_error
            + self.Ki_pos * self.pos_error_integral
            + self.Kd_pos * pos_derivative
        )
        self.prev_pos_error = position_error
        force = force + residual_action[:, 0:3] * self.physics_dt * self.translation_scale * 500

        rotation_error = base_action[:, 3:root_control_dim]
        rotation_error = rot6d_to_aa(rotation_error)
        self.rot_error_integral += rotation_error * self.physics_dt
        self.rot_error_integral = torch.clamp(self.rot_error_integral, -1, 1)
        rot_derivative = (rotation_error - self.prev_rot_error) / self.physics_dt
        torque = (
            self.Kp_rot * rotation_error
            + self.Ki_rot * self.rot_error_integral
            + self.Kd_rot * rot_derivative
        )
        self.prev_rot_error = rotation_error
        torque = torque + residual_action[:, 3:6] * self.physics_dt * self.orientation_scale * 200

        self.apply_forces[:, self.wrist_body_idx, :] = (
            self.act_moving_average * force
            + (1.0 - self.act_moving_average) * self.apply_forces[:, self.wrist_body_idx, :]
        )
        self.apply_torques[:, self.wrist_body_idx, :] = (
            self.act_moving_average * torque
            + (1.0 - self.act_moving_average) * self.apply_torques[:, self.wrist_body_idx, :]
        )

    def _apply_force_control(self, base_action, residual_action):
        """Apply direct force control to wrist."""
        force = (
            1.0 * (base_action[:, 0:3] * self.physics_dt * self.translation_scale * 500)
            + (residual_action[:, 0:3] * self.physics_dt * self.translation_scale * 500)
        )
        torque = (
            1.0 * (base_action[:, 3:6] * self.physics_dt * self.orientation_scale * 200)
            + (residual_action[:, 3:6] * self.physics_dt * self.orientation_scale * 200)
        )

        self.apply_forces[:, self.wrist_body_idx, :] = (
            self.act_moving_average * force
            + (1.0 - self.act_moving_average) * self.apply_forces[:, self.wrist_body_idx, :]
        )
        self.apply_torques[:, self.wrist_body_idx, :] = (
            self.act_moving_average * torque
            + (1.0 - self.act_moving_average) * self.apply_torques[:, self.wrist_body_idx, :]
        )

    def _get_observations(self) -> dict:
        """Compute and return observations."""
        self._compute_intermediate_values()

        # Proprioception observations
        prop_obs = torch.cat([
            self.hand_dof_pos,
            torch.cos(self.hand_dof_pos),
            torch.sin(self.hand_dof_pos),
            self.hand_root_state,
        ], dim=-1)

        # Target observations (from demo data)
        target_obs = self._compute_target_observations()

        # Combine observations
        obs = torch.cat([prop_obs, target_obs], dim=-1)

        # Check for NaN values
        if torch.isnan(obs).any():
            nan_count = torch.isnan(obs).sum().item()
            print(f"[WARNING] NaN detected in observations! Count: {nan_count}")
            # Replace NaN with zeros to prevent crash
            obs = torch.nan_to_num(obs, nan=0.0)

        return {"policy": obs}

    def _compute_intermediate_values(self):
        """Compute intermediate state values."""
        # Hand data - reorder from IsaacLab order to demo order for consistent observation space
        hand_dof_pos_isaaclab = self.hand.data.joint_pos
        hand_dof_vel_isaaclab = self.hand.data.joint_vel

        # Check for NaN in raw data
        if torch.isnan(hand_dof_pos_isaaclab).any() or torch.isnan(hand_dof_vel_isaaclab).any():
            print(f"[WARNING] NaN in hand DOF data!")
            hand_dof_pos_isaaclab = torch.nan_to_num(hand_dof_pos_isaaclab, nan=0.0)
            hand_dof_vel_isaaclab = torch.nan_to_num(hand_dof_vel_isaaclab, nan=0.0)

        self.hand_dof_pos = hand_dof_pos_isaaclab
        self.hand_dof_vel = hand_dof_vel_isaaclab
        root_pos = self.hand.data.root_pos_w - self.scene.env_origins
        root_quat = self.hand.data.root_quat_w
        root_lin_vel = self.hand.data.root_lin_vel_w
        root_ang_vel = self.hand.data.root_ang_vel_w

        # Check for physics instability (hand flying away or NaN)
        if torch.isnan(root_pos).any() or (root_pos.abs() > 10.0).any():
            nan_count = torch.isnan(root_pos).sum().item()
            far_count = (root_pos.abs() > 10.0).sum().item()
            # Only print warning occasionally to avoid spam
            if not hasattr(self, '_instability_warn_count'):
                self._instability_warn_count = 0
            self._instability_warn_count += 1
            if self._instability_warn_count <= 5 or self._instability_warn_count % 100 == 0:
                print(f"[WARNING] Physics instability! NaN: {nan_count}, Far: {far_count} (warning #{self._instability_warn_count})")
            root_pos = torch.nan_to_num(root_pos, nan=0.0)
            root_lin_vel = torch.nan_to_num(root_lin_vel, nan=0.0)
            root_ang_vel = torch.nan_to_num(root_ang_vel, nan=0.0)

        # Store root state with quaternion in wxyz format (IsaacLab native)
        self.hand_root_state = torch.cat([
            root_pos,
            root_quat,  # wxyz format
            root_lin_vel,
            root_ang_vel,
        ], dim=-1)

        # Fingertip data
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.fingertip_indices]
        self.fingertip_pos -= self.scene.env_origins.unsqueeze(1)

        # Object data
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def _compute_target_observations(self) -> torch.Tensor:
        """Compute target observations from demo data."""
        nE = self.num_envs
        nF = self.obs_future_length

        cur_idx = self.episode_length_buf + 1
        cur_idx = torch.clamp(cur_idx, torch.zeros_like(self.demo_data["seq_len"]), self.demo_data["seq_len"] - 1)

        cur_idx = torch.stack([cur_idx + t for t in range(nF)], dim=-1)
        cur_idx = torch.clamp(cur_idx, max=self.demo_data["seq_len"].unsqueeze(-1) - 1)

        nT = self.demo_data["wrist_pos"].shape[1]

        def indicing(data, idx):
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)

        # Wrist target
        target_wrist_pos = indicing(self.demo_data["wrist_pos"], cur_idx)
        cur_wrist_pos = self.hand_root_state[:, :3]
        delta_wrist_pos = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1)

        target_wrist_vel = indicing(self.demo_data["wrist_velocity"], cur_idx)
        cur_wrist_vel = self.hand_root_state[:, 7:10]
        wrist_vel = target_wrist_vel.reshape(nE, -1)
        delta_wrist_vel = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1)

        target_wrist_rot = indicing(self.demo_data["wrist_rot"], cur_idx)
        cur_wrist_rot_wxyz = self.hand_root_state[:, 3:7]  # wxyz format (IsaacLab native)
        cur_wrist_rot = wxyz_to_xyzw(cur_wrist_rot_wxyz)  # Convert to xyzw for pre-trained model
        # Convert target quaternion to xyzw format for compatibility with pre-trained IsaacGym model
        # All quaternion math done in xyzw to match original behavior exactly
        wrist_quat_wxyz = aa_to_quat(target_wrist_rot.reshape(nE * nF, -1))
        wrist_quat = wxyz_to_xyzw(wrist_quat_wxyz)  # xyzw format
        # Compute delta in xyzw format using xyzw quaternion operations
        delta_wrist_quat = quat_mul_xyzw(
            cur_wrist_rot[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate_xyzw(wrist_quat),
        ).reshape(nE, -1)
        wrist_quat = wrist_quat.reshape(nE, -1)

        target_wrist_ang_vel = indicing(self.demo_data["wrist_angular_velocity"], cur_idx)
        cur_wrist_ang_vel = self.hand_root_state[:, 10:13]
        wrist_ang_vel = target_wrist_ang_vel.reshape(nE, -1)
        delta_wrist_ang_vel = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        # Joint targets
        target_joints_pos = indicing(self.demo_data["mano_joints"], cur_idx).reshape(nE, nF, -1, 3)
        # Note: cur_joint_pos requires body state access which needs proper body indexing
        delta_joints_pos = target_joints_pos.reshape(nE, -1)  # Simplified for now

        target_joints_vel = indicing(self.demo_data["mano_joints_velocity"], cur_idx).reshape(nE, nF, -1, 3)
        joints_vel = target_joints_vel.reshape(nE, -1)
        delta_joints_vel = target_joints_vel.reshape(nE, -1)  # Simplified for now

        # Object targets
        target_obj_transf = indicing(self.demo_data["obj_trajectory"], cur_idx)
        target_obj_transf = target_obj_transf.reshape(nE * nF, 4, 4)
        delta_manip_obj_pos = (
            target_obj_transf[:, :3, 3].reshape(nE, nF, -1) - self.object_pos[:, None]
        ).reshape(nE, -1)

        target_obj_vel = indicing(self.demo_data["obj_velocity"], cur_idx)
        manip_obj_vel = target_obj_vel.reshape(nE, -1)
        delta_manip_obj_vel = (target_obj_vel - self.object_linvel[:, None]).reshape(nE, -1)

        manip_obj_quat_wxyz = rotmat_to_quat(target_obj_transf[:, :3, :3])  # Returns wxyz
        # Convert to xyzw for consistency with IsaacGym pre-trained model
        manip_obj_quat = wxyz_to_xyzw(manip_obj_quat_wxyz)  # xyzw format
        cur_obj_rot_xyzw = wxyz_to_xyzw(self.object_rot)  # Convert current object rot to xyzw
        # Compute delta in xyzw format using xyzw quaternion operations
        delta_manip_obj_quat = quat_mul_xyzw(
            cur_obj_rot_xyzw[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate_xyzw(manip_obj_quat),
        ).reshape(nE, -1)
        manip_obj_quat = manip_obj_quat.reshape(nE, -1)

        target_obj_ang_vel = indicing(self.demo_data["obj_angular_velocity"], cur_idx)
        manip_obj_ang_vel = target_obj_ang_vel.reshape(nE, -1)
        delta_manip_obj_ang_vel = (target_obj_ang_vel - self.object_angvel[:, None]).reshape(nE, -1)

        # Object to joints distance
        obj_to_joints = torch.norm(
            self.object_pos[:, None] - self.fingertip_pos, dim=-1
        ).reshape(nE, -1)

        # Tips distance from demo
        gt_tips_distance = indicing(self.demo_data["tips_distance"], cur_idx).reshape(nE, -1)

        # BPS features
        bps = self.obj_bps

        # Concatenate all target observations
        # Note: quaternions are in xyzw format for compatibility with pre-trained IsaacGym model
        target_obs = torch.cat([
            delta_wrist_pos,
            wrist_vel,
            delta_wrist_vel,
            wrist_quat,  # xyzw format
            delta_wrist_quat,  # xyzw format
            wrist_ang_vel,
            delta_wrist_ang_vel,
            delta_joints_pos,
            joints_vel,
            delta_joints_vel,
            delta_manip_obj_pos,
            manip_obj_vel,
            delta_manip_obj_vel,
            manip_obj_quat,  # xyzw format
            delta_manip_obj_quat,  # xyzw format
            manip_obj_ang_vel,
            delta_manip_obj_ang_vel,
            obj_to_joints,
            gt_tips_distance,
            bps,
        ], dim=-1)

        return target_obs

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on imitation error."""
        # Get current and target states
        cur_idx = self.episode_length_buf

        # Current states (wxyz format - IsaacLab native)
        current_eef_pos = self.hand_root_state[:, :3]
        current_eef_quat = self.hand_root_state[:, 3:7]  # wxyz format for IsaacLab quat_mul
        current_eef_vel = self.hand_root_state[:, 7:10]
        current_eef_ang_vel = self.hand_root_state[:, 10:13]

        current_obj_pos = self.object_pos
        current_obj_quat = self.object_rot  # Already wxyz from IsaacLab
        current_obj_vel = self.object_linvel
        current_obj_ang_vel = self.object_angvel

        # Target states from demo data
        target_eef_pos = self.demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]
        target_eef_quat = aa_to_quat(self.demo_data["wrist_rot"][torch.arange(self.num_envs), cur_idx])
        target_obj_transf = self.demo_data["obj_trajectory"][torch.arange(self.num_envs), cur_idx]
        target_obj_pos = target_obj_transf[:, :3, 3]
        target_obj_quat = rotmat_to_quat(target_obj_transf[:, :3, :3])
        # Position rewards
        diff_eef_pos_dist = torch.norm(target_eef_pos - current_eef_pos, dim=-1)
        reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)

        diff_obj_pos_dist = torch.norm(target_obj_pos - current_obj_pos, dim=-1)
        reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

        # Rotation rewards
        # Note: IsaacLab uses wxyz quaternion format, so w component is at index 0
        diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
        diff_eef_rot_angle = 2.0 * torch.acos(torch.clamp(torch.abs(diff_eef_rot[:, 0]), max=1.0))
        reward_eef_rot = torch.exp(-1 * diff_eef_rot_angle.abs())

        diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
        diff_obj_rot_angle = 2.0 * torch.acos(torch.clamp(torch.abs(diff_obj_rot[:, 0]), max=1.0))
        reward_obj_rot = torch.exp(-3 * diff_obj_rot_angle.abs())

        # Combined reward
        reward = (
            0.1 * reward_eef_pos
            + 0.6 * reward_eef_rot
            + 5.0 * reward_obj_pos
            + 1.0 * reward_obj_rot
        )

        # Update tracking buffers
        self.total_rew_buf += reward

        # Log reward components
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["reward_eef_pos"] = reward_eef_pos.mean()
        self.extras["log"]["reward_eef_rot"] = reward_eef_rot.mean()
        self.extras["log"]["reward_obj_pos"] = reward_obj_pos.mean()
        self.extras["log"]["reward_obj_rot"] = reward_obj_rot.mean()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions."""
        self._compute_intermediate_values()

        max_length = torch.clip(self.demo_data["seq_len"], 0, self.max_episode_length)

        # Termination: task failed (large tracking error or physics error)
        cur_idx = self.episode_length_buf
        target_obj_transf = self.demo_data["obj_trajectory"][torch.arange(self.num_envs), cur_idx]
        target_obj_pos = target_obj_transf[:, :3, 3]
        diff_obj_pos_dist = torch.norm(target_obj_pos - self.object_pos, dim=-1)

        terminated = diff_obj_pos_dist > 0.1  # Large object tracking error

        # Timeout: reached max episode length or demo sequence end
        time_out = self.episode_length_buf >= max_length - 1

        # Update success/failure buffers
        self.success_buf = time_out & ~terminated
        self.failure_buf = terminated

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        super()._reset_idx(env_ids)

        # Reset tracking buffers
        self.running_progress_buf[env_ids] = 0
        self.success_buf[env_ids] = False
        self.failure_buf[env_ids] = False
        self.error_buf[env_ids] = False
        self.total_rew_buf[env_ids] = 0
        self.prev_targets[env_ids] = 0
        self.curr_targets[env_ids] = 0
        self.apply_forces[env_ids] = 0
        self.apply_torques[env_ids] = 0
        self.tips_contact_history[env_ids] = True

        if self.use_pid_control:
            self.prev_pos_error[env_ids] = 0
            self.prev_rot_error[env_ids] = 0
            self.pos_error_integral[env_ids] = 0
            self.rot_error_integral[env_ids] = 0

        # Determine initial state index
        if self.random_state_init:
            if self.rollout_begin is not None:
                seq_idx = (
                    torch.floor(
                        self.rollout_len * 0.98 * torch.rand(len(env_ids), device=self.device)
                    ).long()
                    + self.rollout_begin
                )
                seq_idx = torch.clamp(
                    seq_idx,
                    torch.zeros(1, device=self.device).long(),
                    torch.floor(self.demo_data["seq_len"][env_ids] * 0.98).long(),
                )
            else:
                seq_idx = torch.floor(
                    self.demo_data["seq_len"][env_ids]
                    * 0.98
                    * torch.rand(len(env_ids), device=self.device)
                ).long()
        else:
            if self.rollout_begin is not None:
                seq_idx = self.rollout_begin * torch.ones(len(env_ids), device=self.device).long()
            else:
                seq_idx = torch.zeros(len(env_ids), device=self.device).long()

        # Reset hand state from demo data
        # Convert env_ids to tensor if needed
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.long()
        else:
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Debug: print shapes to help identify indexing issues
        if len(env_ids_tensor) > 0 and env_ids_tensor.max() >= self.demo_data["opt_dof_pos"].shape[0]:
            print(f"[ERROR] env_ids_tensor max {env_ids_tensor.max()} >= demo_data dim {self.demo_data['opt_dof_pos'].shape[0]}")
        if seq_idx.max() >= self.demo_data["opt_dof_pos"].shape[1]:
            print(f"[ERROR] seq_idx max {seq_idx.max()} >= demo_data seq dim {self.demo_data['opt_dof_pos'].shape[1]}")

        # Demo data DOF positions - use directly without reordering
        dof_pos = self.demo_data["opt_dof_pos"][env_ids_tensor, seq_idx]
        dof_vel = self.demo_data["opt_dof_velocity"][env_ids_tensor, seq_idx]

        # Clamp DOF positions to limits
        dof_pos = saturate(dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)

        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # Reset wrist pose
        opt_wrist_pos = self.demo_data["opt_wrist_pos"][env_ids_tensor, seq_idx]
        # aa_to_quat returns wxyz format, IsaacLab also uses wxyz - no conversion needed
        # Note: The original IsaacGym code converts wxyz->xyzw because IsaacGym uses xyzw.
        # For IsaacLab, we keep wxyz format without any conversion or extra rotation.
        opt_wrist_rot_aa = self.demo_data["opt_wrist_rot"][env_ids_tensor, seq_idx]
        opt_wrist_rot = aa_to_quat(opt_wrist_rot_aa)

        # Apply correction rotation if needed for IsaacLab coordinate system
        # This compensates for differences in how IsaacGym vs IsaacLab interpret root orientation
        # Set USE_ROTATION_CORRECTION = True and adjust CORRECTION_QUAT if hands face wrong direction
        # NOTE: After testing, the original demo quaternions work correctly without correction
        # when forces are properly applied to the wrist. Only enable if hands still face wrong way.
        USE_ROTATION_CORRECTION = False
        if USE_ROTATION_CORRECTION:
            # -90° about X axis to fix "palms facing up" issue (wxyz format)
            # This rotates the hand so palms face forward/toward object instead of up
            CORRECTION_QUAT = torch.tensor([0.7071068, -0.7071068, 0.0, 0.0], device=self.device, dtype=opt_wrist_rot.dtype)
            opt_wrist_rot = quat_mul(opt_wrist_rot, CORRECTION_QUAT.unsqueeze(0).expand(opt_wrist_rot.shape[0], -1))

        opt_wrist_vel = self.demo_data["opt_wrist_velocity"][env_ids_tensor, seq_idx]
        opt_wrist_ang_vel = self.demo_data["opt_wrist_angular_velocity"][env_ids_tensor, seq_idx]

        root_state = torch.cat([
            opt_wrist_pos + self.scene.env_origins[env_ids],
            opt_wrist_rot,
            opt_wrist_vel,
            opt_wrist_ang_vel,
        ], dim=-1)
        self.hand.write_root_state_to_sim(root_state, env_ids=env_ids)

        # Reset object state from demo data
        obj_pos_init = self.demo_data["obj_trajectory"][env_ids_tensor, seq_idx, :3, 3]
        # rotmat_to_quat returns wxyz format, IsaacLab also uses wxyz - no conversion needed
        obj_rot_init = rotmat_to_quat(self.demo_data["obj_trajectory"][env_ids_tensor, seq_idx, :3, :3])
        obj_vel = self.demo_data["obj_velocity"][env_ids_tensor, seq_idx]
        obj_ang_vel = self.demo_data["obj_angular_velocity"][env_ids_tensor, seq_idx]

        obj_state = torch.cat([
            obj_pos_init + self.scene.env_origins[env_ids],
            obj_rot_init,
            obj_vel,
            obj_ang_vel,
        ], dim=-1)
        self.object.write_root_state_to_sim(obj_state, env_ids=env_ids)

        # Update episode length buffer to match demo sequence position
        self.episode_length_buf[env_ids] = seq_idx

        self._compute_intermediate_values()


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Scale tensor from [-1, 1] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Unscale tensor from [lower, upper] to [-1, 1]."""
    return (2.0 * x - upper - lower) / (upper - lower)
