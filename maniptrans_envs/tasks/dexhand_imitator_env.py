"""
DexHand Imitator Training Environment for IsaacLab.

This environment trains an imitator policy to track human hand trajectories
(hand only, no object). It is the first stage of the two-stage ManipTrans pipeline:
  Stage 1: Imitator (this env) - learns to track MANO hand motion via PPO
  Stage 2: Residual (dexhand_manip_env.py) - adds object manipulation on top

Key differences from the residual env (dexhand_manip_env.py):
  - NO object in simulation (hand + table only)
  - Single policy actions (not base+residual split)
  - Imitation-only rewards (14 terms, no object/contact rewards)
  - Tighten curriculum (gradually tighten error tolerance)
  - No gravity/friction domain randomization
  - Longer episodes (2000 steps)
  - obs_future_length = 1 (not 3)
  - Reset with noisy default DOF pose (not IK-solved)
"""

from __future__ import annotations

import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import Dict, List, Tuple, Any

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, UrdfFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_mul, saturate

from tqdm import tqdm

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_quat, aa_to_rotmat, rotmat_to_quat, rotmat_to_aa, rot6d_to_aa


# Configuration constants (must match original ManipTrans)
ROBOT_HEIGHT = 0.00214874


@configclass
class DexHandImitatorEnvCfg(DirectRLEnvCfg):
    """Configuration for the DexHand imitator training environment."""

    # Environment settings
    decimation = 2
    episode_length_s = 40.0  # Generous; max_episode_length overridden in __init__

    # Action/Observation space (defaults for Inspire, recalculated in __init__)
    action_space = 18  # 6 wrist + 12 dofs for inspire
    observation_space = 237  # Computed dynamically
    state_space = 0

    # Simulation (same physics as residual env)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics_material=RigidBodyMaterialCfg(
            static_friction=4.0,
            dynamic_friction=4.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.0005,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            enable_external_forces_every_iteration=True,  # Required for wrist force control
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
    act_moving_average: float = 0.4
    translation_scale: float = 1.0
    orientation_scale: float = 0.1
    use_pid_control: bool = False

    # Training settings
    training: bool = True
    obs_future_length: int = 1  # Imitator default (residual uses 3)
    random_state_init: bool = True

    # Tightening curriculum (imitator uses exp_decay with 128k steps)
    tighten_method: str = "exp_decay"  # "None", "const", "linear_decay", "exp_decay", "cos"
    tighten_factor: float = 0.7
    tighten_steps: int = 128000

    # Max episode length (longer than residual's 1200)
    max_episode_length: int = 2000


@configclass
class DexHandImitatorRHEnvCfg(DexHandImitatorEnvCfg):
    """Configuration for right-hand imitator training."""
    side: str = "right"


@configclass
class DexHandImitatorLHEnvCfg(DexHandImitatorEnvCfg):
    """Configuration for left-hand imitator training."""
    side: str = "left"


class DexHandImitatorEnv(DirectRLEnv):
    """
    DexHand Imitator Training Environment using IsaacLab.

    Trains a policy to imitate human hand trajectories (without object).
    The trained imitator serves as the base policy for residual learning.
    """

    cfg: DexHandImitatorEnvCfg

    def __init__(self, cfg: DexHandImitatorEnvCfg, render_mode: str | None = None, **kwargs):
        # Initialize dexhand before super().__init__
        self.dexhand = DexHandFactory.create_hand(cfg.dexhand_type, cfg.side)
        self.side = cfg.side
        self.data_indices = cfg.data_indices if cfg.data_indices is not None else ["g0"]

        # Calculate action space: single policy (no base+residual split)
        n_dofs = self.dexhand.n_dofs
        # Action: wrist force/torque(6) + DOF targets(n_dofs)
        single_action_dim = 6 + n_dofs
        if cfg.use_pid_control:
            single_action_dim += 3  # extra 3 for PID rot6d
        cfg.num_actions = single_action_dim
        cfg.action_space = cfg.num_actions

        # Calculate observation space
        n_joint_bodies = self.dexhand.n_bodies - 1  # Bodies excluding wrist

        # Target observation per future frame (no object obs)
        target_obs_per_frame = (
            3   # delta_wrist_pos
            + 3   # wrist_vel
            + 3   # delta_wrist_vel
            + 4   # wrist_quat (wxyz)
            + 4   # delta_wrist_quat (wxyz)
            + 3   # wrist_ang_vel
            + 3   # delta_wrist_ang_vel
            + n_joint_bodies * 3  # delta_joints_pos
            + n_joint_bodies * 3  # joints_vel
            + n_joint_bodies * 3  # delta_joints_vel
        )
        target_obs_dim = target_obs_per_frame * cfg.obs_future_length

        # Proprioception: q, cos_q, sin_q, base_state (pos zeroed out)
        prop_obs_dim = n_dofs * 3 + 13

        # Privileged: dq (joint velocities only, no object data)
        priv_obs_dim = n_dofs

        cfg.num_observations = prop_obs_dim + priv_obs_dim + target_obs_dim
        cfg.observation_space = cfg.num_observations
        cfg.num_states = 0

        print(f"[INFO] Imitator obs dims: prop={prop_obs_dim}, priv={priv_obs_dim}, "
              f"target={target_obs_dim} ({target_obs_per_frame}*{cfg.obs_future_length}), "
              f"total={cfg.num_observations}")
        print(f"[INFO] Imitator action dim: {cfg.num_actions} (6 wrist + {n_dofs} dofs)")

        super().__init__(cfg, render_mode, **kwargs)

        # Override max_episode_length (DirectRLEnv computes from episode_length_s)
        self.max_episode_length = cfg.max_episode_length

        # Store configuration
        self.use_quat_rot = cfg.use_quat_rot
        self.action_scale = cfg.action_scale
        self.act_moving_average = cfg.act_moving_average
        self.translation_scale = cfg.translation_scale
        self.orientation_scale = cfg.orientation_scale
        self.use_pid_control = cfg.use_pid_control
        self.training = cfg.training
        self.obs_future_length = cfg.obs_future_length
        self.random_state_init = cfg.random_state_init
        self.tighten_method = cfg.tighten_method
        self.tighten_factor = cfg.tighten_factor
        self.tighten_steps = cfg.tighten_steps

        # Initialize buffers
        self._init_buffers()

        # Load demo data
        self._load_demo_data()

        # Default DOF positions (in demo/dexhand.dof_names order)
        # Imitator uses pi/36 ≈ 5 deg (different from residual's pi/12 ≈ 15 deg)
        default_pose = torch.ones(self.dexhand.n_dofs, device=self.device) * np.pi / 36
        if cfg.dexhand_type == "inspire":
            default_pose[8] = 0.3   # thumb_proximal_yaw_joint
            default_pose[9] = 0.01  # thumb_proximal_pitch_joint
        self.dexhand_default_dof_pos = default_pose

    def _compute_body_indices(self):
        """Compute body/joint index mappings between demo data order and IsaacLab order.

        IsaacLab may reorder bodies during URDF conversion. This method creates
        all necessary mappings for consistent indexing.
        """
        actual_body_names = self.hand.body_names
        actual_joint_names = self.hand.joint_names
        print(f"[INFO] IsaacLab num_bodies: {self.hand.num_bodies}")
        print(f"[INFO] IsaacLab num_joints: {self.hand.num_joints}")
        print(f"[INFO] IsaacLab body names ({len(actual_body_names)}): {actual_body_names}")
        print(f"[INFO] IsaacLab joint names ({len(actual_joint_names)}): {actual_joint_names}")
        print(f"[INFO] Expected body names ({len(self.dexhand.body_names)}): {self.dexhand.body_names}")
        print(f"[INFO] Expected joint names ({len(self.dexhand.dof_names)}): {self.dexhand.dof_names}")

        # Body name -> index mapping
        self.body_name_to_idx = {}
        for i, name in enumerate(actual_body_names):
            self.body_name_to_idx[name] = i

        # Joint name -> index mapping
        self.joint_name_to_idx = {}
        for i, name in enumerate(actual_joint_names):
            self.joint_name_to_idx[name] = i

        # DOF reorder mapping: demo order -> IsaacLab order
        self.demo_to_isaaclab_dof_mapping = []
        for demo_dof_name in self.dexhand.dof_names:
            if demo_dof_name in self.joint_name_to_idx:
                self.demo_to_isaaclab_dof_mapping.append(self.joint_name_to_idx[demo_dof_name])
            else:
                print(f"[WARNING] DOF '{demo_dof_name}' not found in IsaacLab joints!")
                self.demo_to_isaaclab_dof_mapping.append(0)

        self.demo_to_isaaclab_dof_mapping = torch.tensor(
            self.demo_to_isaaclab_dof_mapping, device=self.device, dtype=torch.long
        )
        print(f"[INFO] Demo to IsaacLab DOF mapping: {self.demo_to_isaaclab_dof_mapping.tolist()}")

        # Reverse mapping: IsaacLab order -> demo order
        self.isaaclab_to_demo_dof_mapping = torch.zeros_like(self.demo_to_isaaclab_dof_mapping)
        for demo_idx in range(len(self.demo_to_isaaclab_dof_mapping)):
            isaaclab_idx = self.demo_to_isaaclab_dof_mapping[demo_idx].item()
            self.isaaclab_to_demo_dof_mapping[isaaclab_idx] = demo_idx
        print(f"[INFO] IsaacLab to Demo DOF mapping: {self.isaaclab_to_demo_dof_mapping.tolist()}")

        # Actual counts
        self._actual_num_bodies = len(actual_body_names)
        self._actual_num_joints = len(actual_joint_names)

        # Find wrist body index
        wrist_body_name = self.dexhand.to_dex("wrist")[0]
        if wrist_body_name in self.body_name_to_idx:
            self.wrist_body_idx = self.body_name_to_idx[wrist_body_name]
        else:
            print(f"[WARNING] Wrist body '{wrist_body_name}' not found, falling back to index 0")
            self.wrist_body_idx = 0
        print(f"[INFO] Wrist body index: {self.wrist_body_idx} (name: {wrist_body_name})")

        # Build body reorder mapping: demo body order (excl wrist) -> IsaacLab body indices
        wrist_name = self.dexhand.to_dex("wrist")[0]
        self.demo_body_to_isaaclab_indices = []
        for body_name in self.dexhand.body_names:
            if body_name == wrist_name:
                continue  # Skip wrist
            if body_name in self.body_name_to_idx:
                self.demo_body_to_isaaclab_indices.append(self.body_name_to_idx[body_name])
            else:
                print(f"[WARNING] Body '{body_name}' not found in IsaacLab bodies!")
                self.demo_body_to_isaaclab_indices.append(0)
        self.demo_body_to_isaaclab_indices = torch.tensor(
            self.demo_body_to_isaaclab_indices, device=self.device, dtype=torch.long
        )
        print(f"[INFO] Demo body to IsaacLab indices (excl wrist): {self.demo_body_to_isaaclab_indices.tolist()}")

        # Verify joint count
        if self._actual_num_joints != self.dexhand.n_dofs:
            print(f"[WARNING] Joint count mismatch! IsaacLab={self._actual_num_joints}, expected={self.dexhand.n_dofs}")

    def _init_buffers(self):
        """Initialize tensor buffers."""
        self._compute_body_indices()

        num_joints = self._actual_num_joints if hasattr(self, '_actual_num_joints') else self.dexhand.n_dofs

        # Read joint limits from URDF (in IsaacLab order)
        urdf_path = self.dexhand.urdf_path
        tree = ET.parse(urdf_path)
        urdf_root = tree.getroot()
        urdf_joint_limits = {}
        for joint_elem in urdf_root.findall('.//joint'):
            jname = joint_elem.get('name')
            limit_elem = joint_elem.find('limit')
            if limit_elem is not None and joint_elem.get('type') in ('revolute', 'prismatic', 'continuous'):
                urdf_joint_limits[jname] = (
                    float(limit_elem.get('lower', '-1.5708')),
                    float(limit_elem.get('upper', '1.5708')),
                )

        actual_joint_names = self.hand.joint_names
        lower_limits = []
        upper_limits = []
        for jname in actual_joint_names:
            if jname in urdf_joint_limits:
                lo, hi = urdf_joint_limits[jname]
                lower_limits.append(lo)
                upper_limits.append(hi)
            else:
                print(f"[WARNING] Joint '{jname}' not found in URDF limits, using [-pi/2, pi/2]")
                lower_limits.append(-1.5708)
                upper_limits.append(1.5708)

        self.dexhand_dof_lower_limits = torch.tensor(lower_limits, device=self.device, dtype=torch.float32)
        self.dexhand_dof_upper_limits = torch.tensor(upper_limits, device=self.device, dtype=torch.float32)

        # Read DOF speed limits from URDF (velocity attribute)
        speed_limits = []
        for jname in actual_joint_names:
            found = False
            for joint_elem in urdf_root.findall('.//joint'):
                if joint_elem.get('name') == jname:
                    limit_elem = joint_elem.find('limit')
                    if limit_elem is not None:
                        vel = float(limit_elem.get('velocity', '100.0'))
                        speed_limits.append(vel)
                        found = True
                    break
            if not found:
                speed_limits.append(100.0)
        self.dexhand_dof_speed_limits = torch.tensor(speed_limits, device=self.device, dtype=torch.float32)

        print(f"[INFO] Joint limits (IsaacLab order):")
        print(f"[INFO]   Lower: {self.dexhand_dof_lower_limits.tolist()}")
        print(f"[INFO]   Upper: {self.dexhand_dof_upper_limits.tolist()}")
        print(f"[INFO]   Speed: {self.dexhand_dof_speed_limits.tolist()}")

        # Also build limits in demo order for noisy reset clamping
        self.dexhand_dof_lower_limits_demo = self.dexhand_dof_lower_limits[self.demo_to_isaaclab_dof_mapping]
        self.dexhand_dof_upper_limits_demo = self.dexhand_dof_upper_limits[self.demo_to_isaaclab_dof_mapping]

        # Action-related buffers
        self.prev_targets = torch.zeros(
            (self.num_envs, num_joints), dtype=torch.float, device=self.device
        )
        self.curr_targets = torch.zeros(
            (self.num_envs, num_joints), dtype=torch.float, device=self.device
        )

        # Force application buffers
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

        # Curriculum learning step counter
        self.global_step_count = 0

    def _load_demo_data(self):
        """Load demonstration data for imitation learning."""
        # Compute MuJoCo to Gym coordinate transform
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

        demo_data_list = [segment_data(i) for i in tqdm(range(self.num_envs), desc="Loading imitator demo data")]
        self.demo_data = self._pack_data(demo_data_list)

    def _pack_data(self, data: List[Dict]) -> Dict[str, torch.Tensor]:
        """Pack list of data dicts into batched tensors."""
        packed_data = {}
        # Use wrist_pos for seq_len (imitator has no object requirement)
        packed_data["seq_len"] = torch.tensor(
            [len(d["wrist_pos"]) for d in data], device=self.device
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
        """Set up the simulation scene with robot hand and table (no object)."""
        dexhand_urdf_path = self.dexhand.urdf_path

        # Create robot hand articulation
        self.robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=UrdfFileCfg(
                asset_path=dexhand_urdf_path,
                fix_base=False,
                merge_fixed_joints=False,
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
                    max_depenetration_velocity=1000.0,
                    linear_damping=20.0,
                    angular_damping=20.0,
                    max_linear_velocity=50.0,
                    max_angular_velocity=100.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,   # Reduced from 0.005 for tighter contact (original thickness=0.001)
                    rest_offset=0.0,
                    torsional_patch_radius=0.01,      # Enable torsional friction (original: torsion_friction=0.01)
                    min_torsional_patch_radius=0.005,  # Minimum torsional contact patch
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=self.dexhand.self_collision,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.4, 0.0, 0.4 + 0.015 + ROBOT_HEIGHT),
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

        self.hand = Articulation(self.robot_cfg)

        # Create table with friction=0.1 (matching original ManipTrans, not global 4.0)
        table_cfg = sim_utils.CuboidCfg(
            size=(1.0, 1.6, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.1,
                dynamic_friction=0.1,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        )
        table_cfg.func("/World/envs/env_.*/Table", table_cfg, translation=(-0.1, 0.0, 0.4))

        # NO object spawned (imitator = hand only)

        # Ground plane and lights
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step.

        Computes DOF targets and wrist forces/torques once per policy step,
        matching original ManipTrans where pre_physics_step runs once with dt=1/60.
        _apply_action() then just writes these pre-computed values to the sim.
        """
        self.actions = actions.clone()

        root_control_dim = 9 if self.use_pid_control else 6

        # DOF position targets from action (in demo order)
        dof_pos_demo_order = self.actions[:, root_control_dim:root_control_dim + self.dexhand.n_dofs]
        dof_pos_demo_order = torch.clamp(dof_pos_demo_order, -1, 1)

        # Reorder from demo order to IsaacLab order
        indices = self.demo_to_isaaclab_dof_mapping.unsqueeze(0).expand(dof_pos_demo_order.shape[0], -1)
        dof_pos = torch.zeros_like(dof_pos_demo_order)
        dof_pos.scatter_(1, indices, dof_pos_demo_order)

        # Scale to joint limits (limits are in IsaacLab order)
        self.curr_targets = scale(
            dof_pos,
            self.dexhand_dof_lower_limits,
            self.dexhand_dof_upper_limits,
        )

        # Apply moving average (once per policy step, matching original)
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

        self.prev_targets[:] = self.curr_targets[:]

        # Compute wrist forces/torques (single policy, no residual)
        # Use step_dt (= physics_dt * decimation = 1/60) to match original dt=1/60
        if self.use_pid_control:
            self._apply_pid_control()
        else:
            self._apply_force_control()

    def _apply_action(self) -> None:
        """Write pre-computed targets and forces to the simulator.

        Called decimation times (2x) per policy step. All computation is done
        in _pre_physics_step to match original once-per-step behavior.
        """
        # Set joint position targets
        self.hand.set_joint_position_target(self.curr_targets)

        # Apply external forces/torques to wrist
        self.hand.set_external_force_and_torque(
            forces=self.apply_forces,
            torques=self.apply_torques,
            body_ids=None,
            env_ids=None,
            is_global=True,
        )

    def _apply_pid_control(self):
        """Apply PID-based wrist control (single policy)."""
        # Use step_dt (= physics_dt * decimation = 1/60) to match original dt=1/60
        dt = self.step_dt
        position_error = self.actions[:, :3]
        self.pos_error_integral += position_error * dt
        self.pos_error_integral = torch.clamp(self.pos_error_integral, -1, 1)
        pos_derivative = (position_error - self.prev_pos_error) / dt
        force = (
            self.Kp_pos * position_error
            + self.Ki_pos * self.pos_error_integral
            + self.Kd_pos * pos_derivative
        )
        self.prev_pos_error = position_error

        rotation_error = self.actions[:, 3:9]
        rotation_error = rot6d_to_aa(rotation_error)
        self.rot_error_integral += rotation_error * dt
        self.rot_error_integral = torch.clamp(self.rot_error_integral, -1, 1)
        rot_derivative = (rotation_error - self.prev_rot_error) / dt
        torque = (
            self.Kp_rot * rotation_error
            + self.Ki_rot * self.rot_error_integral
            + self.Kd_rot * rot_derivative
        )
        self.prev_rot_error = rotation_error

        self.apply_forces[:, self.wrist_body_idx, :] = (
            self.act_moving_average * force
            + (1.0 - self.act_moving_average) * self.apply_forces[:, self.wrist_body_idx, :]
        )
        self.apply_torques[:, self.wrist_body_idx, :] = (
            self.act_moving_average * torque
            + (1.0 - self.act_moving_average) * self.apply_torques[:, self.wrist_body_idx, :]
        )

    def _apply_force_control(self):
        """Apply direct force control to wrist (single policy)."""
        # Use step_dt (= physics_dt * decimation = 1/60) to match original dt=1/60
        dt = self.step_dt
        force = self.actions[:, 0:3] * dt * self.translation_scale * 500
        torque = self.actions[:, 3:6] * dt * self.orientation_scale * 200

        self.apply_forces[:, self.wrist_body_idx, :] = (
            self.act_moving_average * force
            + (1.0 - self.act_moving_average) * self.apply_forces[:, self.wrist_body_idx, :]
        )
        self.apply_torques[:, self.wrist_body_idx, :] = (
            self.act_moving_average * torque
            + (1.0 - self.act_moving_average) * self.apply_torques[:, self.wrist_body_idx, :]
        )

    def _get_observations(self) -> dict:
        """Compute observations: proprioception + privileged + target."""
        self._compute_intermediate_values()

        # Proprioception: q, cos(q), sin(q), base_state (position zeroed out)
        zeroed_root_state = torch.cat([
            torch.zeros_like(self.hand_root_state[:, :3]),  # Zero out position
            self.hand_root_state[:, 3:],  # Keep quat + vel + ang_vel
        ], dim=-1)

        prop_obs = torch.cat([
            self.hand_dof_pos,
            torch.cos(self.hand_dof_pos),
            torch.sin(self.hand_dof_pos),
            zeroed_root_state,
        ], dim=-1)

        # Privileged: joint velocities only (no object data for imitator)
        priv_obs = self.hand_dof_vel

        # Target observations from demo data (no object targets)
        target_obs = self._compute_target_observations()

        # Combine
        obs = torch.cat([prop_obs, priv_obs, target_obs], dim=-1)

        # NaN safety
        if torch.isnan(obs).any():
            nan_count = torch.isnan(obs).sum().item()
            print(f"[WARNING] NaN in imitator observations! Count: {nan_count}")
            obs = torch.nan_to_num(obs, nan=0.0)

        return {"policy": obs}

    def _compute_intermediate_values(self):
        """Compute intermediate state values (hand only, no object)."""
        # Hand DOF state - reorder from IsaacLab to demo order
        hand_dof_pos_isaaclab = self.hand.data.joint_pos
        hand_dof_vel_isaaclab = self.hand.data.joint_vel

        if torch.isnan(hand_dof_pos_isaaclab).any() or torch.isnan(hand_dof_vel_isaaclab).any():
            print(f"[WARNING] NaN in hand DOF data!")
            hand_dof_pos_isaaclab = torch.nan_to_num(hand_dof_pos_isaaclab, nan=0.0)
            hand_dof_vel_isaaclab = torch.nan_to_num(hand_dof_vel_isaaclab, nan=0.0)

        # Reorder to demo order
        self.hand_dof_pos = hand_dof_pos_isaaclab[:, self.demo_to_isaaclab_dof_mapping]
        self.hand_dof_vel = hand_dof_vel_isaaclab[:, self.demo_to_isaaclab_dof_mapping]

        # Root state
        root_pos = self.hand.data.root_pos_w - self.scene.env_origins
        root_quat = self.hand.data.root_quat_w
        root_lin_vel = self.hand.data.root_lin_vel_w
        root_ang_vel = self.hand.data.root_ang_vel_w

        if torch.isnan(root_pos).any() or (root_pos.abs() > 10.0).any():
            nan_count = torch.isnan(root_pos).sum().item()
            far_count = (root_pos.abs() > 10.0).sum().item()
            print(f"[WARNING] Physics instability! NaN: {nan_count}, Far: {far_count}")
            root_pos = torch.nan_to_num(root_pos, nan=0.0)
            root_lin_vel = torch.nan_to_num(root_lin_vel, nan=0.0)
            root_ang_vel = torch.nan_to_num(root_ang_vel, nan=0.0)

        self.hand_root_state = torch.cat([
            root_pos, root_quat, root_lin_vel, root_ang_vel,
        ], dim=-1)

        # All body positions in demo order (excluding wrist) for reward computation
        all_body_pos_isaaclab = self.hand.data.body_pos_w - self.scene.env_origins.unsqueeze(1)
        self.hand_body_pos = all_body_pos_isaaclab[:, self.demo_body_to_isaaclab_indices]

        # Body velocities in demo order
        try:
            all_body_vel = self.hand.data.body_vel_w
        except AttributeError:
            all_body_vel = self.hand.data.body_link_vel_w
        self.hand_body_vel = all_body_vel[:, self.demo_body_to_isaaclab_indices, :3]

    def _compute_target_observations(self) -> torch.Tensor:
        """Compute target observations from demo data (no object targets)."""
        nE = self.num_envs
        nF = self.obs_future_length

        cur_idx = self.episode_length_buf + 1
        cur_idx = torch.clamp(cur_idx, torch.zeros_like(self.demo_data["seq_len"]), self.demo_data["seq_len"] - 1)

        cur_idx = torch.stack([cur_idx + t for t in range(nF)], dim=-1)
        cur_idx = torch.clamp(cur_idx, max=self.demo_data["seq_len"].unsqueeze(-1) - 1)

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
        cur_wrist_rot_wxyz = self.hand_root_state[:, 3:7]  # wxyz from IsaacLab
        wrist_quat_wxyz = aa_to_quat(target_wrist_rot.reshape(nE * nF, -1))  # wxyz from pytorch3d
        # Native wxyz for IsaacLab-trained imitator (no xyzw conversion)
        wrist_quat = wrist_quat_wxyz
        delta_wrist_quat = quat_mul(
            cur_wrist_rot_wxyz[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(wrist_quat),
        ).reshape(nE, -1)
        wrist_quat = wrist_quat.reshape(nE, -1)

        target_wrist_ang_vel = indicing(self.demo_data["wrist_angular_velocity"], cur_idx)
        cur_wrist_ang_vel = self.hand_root_state[:, 10:13]
        wrist_ang_vel = target_wrist_ang_vel.reshape(nE, -1)
        delta_wrist_ang_vel = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        # Joint targets (body positions excluding wrist, in demo order)
        target_joints_pos = indicing(self.demo_data["mano_joints"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joints_pos = self.hand_body_pos  # (nE, n_bodies-1, 3) in demo order
        delta_joints_pos = (target_joints_pos - cur_joints_pos[:, None]).reshape(nE, -1)

        target_joints_vel = indicing(self.demo_data["mano_joints_velocity"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joints_vel = self.hand_body_vel
        joints_vel = target_joints_vel.reshape(nE, -1)
        delta_joints_vel = (target_joints_vel - cur_joints_vel[:, None]).reshape(nE, -1)

        # Concatenate (no object targets, no BPS, no tips_distance)
        target_obs = torch.cat([
            delta_wrist_pos,
            wrist_vel,
            delta_wrist_vel,
            wrist_quat,
            delta_wrist_quat,
            wrist_ang_vel,
            delta_wrist_ang_vel,
            delta_joints_pos,
            joints_vel,
            delta_joints_vel,
        ], dim=-1)

        return target_obs

    def _get_rewards(self) -> torch.Tensor:
        """Compute imitation rewards (14 terms, no object/contact rewards)."""
        cur_idx = self.episode_length_buf
        env_arange = torch.arange(self.num_envs, device=self.device)

        # Current states
        current_eef_pos = self.hand_root_state[:, :3]
        current_eef_quat = self.hand_root_state[:, 3:7]
        current_eef_vel = self.hand_root_state[:, 7:10]
        current_eef_ang_vel = self.hand_root_state[:, 10:13]

        # Target states from demo
        target_eef_pos = self.demo_data["wrist_pos"][env_arange, cur_idx]
        target_eef_quat = aa_to_quat(self.demo_data["wrist_rot"][env_arange, cur_idx])
        target_eef_vel = self.demo_data["wrist_velocity"][env_arange, cur_idx]
        target_eef_ang_vel = self.demo_data["wrist_angular_velocity"][env_arange, cur_idx]

        # --- End effector rewards ---
        diff_eef_pos_dist = torch.norm(target_eef_pos - current_eef_pos, dim=-1)
        reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)

        diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
        diff_eef_rot_angle = 2.0 * torch.acos(torch.clamp(torch.abs(diff_eef_rot[:, 0]), max=1.0))
        reward_eef_rot = torch.exp(-1 * diff_eef_rot_angle.abs())

        diff_eef_vel = target_eef_vel - current_eef_vel
        reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))

        diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel
        reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))

        # --- Joint position rewards ---
        joints_pos = self.hand_body_pos  # (nE, n_bodies-1, 3) in demo order
        target_joints_pos = self.demo_data["mano_joints"][env_arange, cur_idx].reshape(
            self.num_envs, -1, 3
        )
        diff_joints_pos = target_joints_pos - joints_pos
        diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)  # (nE, n_bodies-1)

        # Per-group joint distances using weight_idx
        weight_idx = self.dexhand.weight_idx
        diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in weight_idx["thumb_tip"]]].mean(dim=-1)
        diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in weight_idx["index_tip"]]].mean(dim=-1)
        diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in weight_idx["middle_tip"]]].mean(dim=-1)
        diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in weight_idx["ring_tip"]]].mean(dim=-1)
        diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in weight_idx["pinky_tip"]]].mean(dim=-1)
        diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in weight_idx["level_1_joints"]]].mean(dim=-1)
        diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in weight_idx["level_2_joints"]]].mean(dim=-1)

        reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
        reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
        reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
        reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
        reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
        reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
        reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

        # --- Joint velocity reward ---
        joints_vel = self.hand_body_vel
        target_joints_vel = self.demo_data["mano_joints_velocity"][env_arange, cur_idx].reshape(
            self.num_envs, -1, 3
        )
        diff_joints_vel = target_joints_vel - joints_vel
        reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

        # --- Power penalties ---
        applied_torque = self.hand.data.applied_torque
        joint_vel_isaaclab = self.hand.data.joint_vel
        power = torch.abs(applied_torque * joint_vel_isaaclab).sum(dim=-1)
        reward_power = torch.exp(-10 * power)

        wrist_force = self.apply_forces[:, self.wrist_body_idx, :]
        wrist_torque = self.apply_torques[:, self.wrist_body_idx, :]
        wrist_power = torch.abs(torch.sum(wrist_force * current_eef_vel, dim=-1))
        wrist_power += torch.abs(torch.sum(wrist_torque * current_eef_ang_vel, dim=-1))
        reward_wrist_power = torch.exp(-2 * wrist_power)

        # --- Cache for termination check ---
        self._reward_cache = {
            "diff_thumb_tip_pos_dist": diff_thumb_tip_pos_dist,
            "diff_index_tip_pos_dist": diff_index_tip_pos_dist,
            "diff_middle_tip_pos_dist": diff_middle_tip_pos_dist,
            "diff_ring_tip_pos_dist": diff_ring_tip_pos_dist,
            "diff_pinky_tip_pos_dist": diff_pinky_tip_pos_dist,
            "diff_level_1_pos_dist": diff_level_1_pos_dist,
            "diff_level_2_pos_dist": diff_level_2_pos_dist,
            "current_eef_vel": current_eef_vel,
            "current_eef_ang_vel": current_eef_ang_vel,
            "joints_vel": joints_vel,
        }

        # --- Total reward (14 terms, matching original imitator weights) ---
        reward = (
            0.1  * reward_eef_pos
            + 0.6  * reward_eef_rot
            + 0.9  * reward_thumb_tip_pos
            + 0.8  * reward_index_tip_pos
            + 0.75 * reward_middle_tip_pos
            + 0.6  * reward_pinky_tip_pos
            + 0.6  * reward_ring_tip_pos
            + 0.5  * reward_level_1_pos
            + 0.3  * reward_level_2_pos
            + 0.1  * reward_eef_vel
            + 0.05 * reward_eef_ang_vel
            + 0.1  * reward_joints_vel
            + 0.5  * reward_power
            + 0.5  * reward_wrist_power
        )

        self.total_rew_buf += reward

        # Log reward components
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["reward_eef_pos"] = reward_eef_pos.mean()
        self.extras["log"]["reward_eef_rot"] = reward_eef_rot.mean()
        self.extras["log"]["reward_joints_pos"] = (
            reward_thumb_tip_pos + reward_index_tip_pos + reward_middle_tip_pos
            + reward_pinky_tip_pos + reward_ring_tip_pos
            + reward_level_1_pos + reward_level_2_pos
        ).mean()
        self.extras["log"]["reward_eef_vel"] = reward_eef_vel.mean()
        self.extras["log"]["reward_joints_vel"] = reward_joints_vel.mean()
        self.extras["log"]["reward_power"] = reward_power.mean()
        self.extras["log"]["reward_wrist_power"] = reward_wrist_power.mean()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions (imitator: no object checks, grace=20)."""
        self._compute_intermediate_values()
        self.global_step_count += 1
        self.running_progress_buf += 1

        max_length = torch.clip(self.demo_data["seq_len"], 0, self.max_episode_length)

        # Compute tighten curriculum scale factor
        if self.training:
            last_step = self.global_step_count
            if self.tighten_method == "None":
                scale_factor = 1.0
            elif self.tighten_method == "const":
                scale_factor = self.tighten_factor
            elif self.tighten_method == "linear_decay":
                scale_factor = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
            elif self.tighten_method == "exp_decay":
                scale_factor = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (
                    1 - self.tighten_factor
                ) + self.tighten_factor
            elif self.tighten_method == "cos":
                scale_factor = self.tighten_factor + np.abs(
                    -1 * (1 - self.tighten_factor) * np.cos(last_step / self.tighten_steps * np.pi)
                ) * (2 ** (-1 * last_step / self.tighten_steps))
            else:
                scale_factor = 1.0
        else:
            scale_factor = 1.0

        # Use cached values from _get_rewards
        rc = self._reward_cache if hasattr(self, '_reward_cache') else {}
        diff_thumb_tip_pos_dist = rc.get("diff_thumb_tip_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_index_tip_pos_dist = rc.get("diff_index_tip_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_middle_tip_pos_dist = rc.get("diff_middle_tip_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_ring_tip_pos_dist = rc.get("diff_ring_tip_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_pinky_tip_pos_dist = rc.get("diff_pinky_tip_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_level_1_pos_dist = rc.get("diff_level_1_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_level_2_pos_dist = rc.get("diff_level_2_pos_dist", torch.zeros(self.num_envs, device=self.device))
        current_eef_vel = rc.get("current_eef_vel", self.hand_root_state[:, 7:10])
        current_eef_ang_vel = rc.get("current_eef_ang_vel", self.hand_root_state[:, 10:13])
        joints_vel = rc.get("joints_vel", self.hand_body_vel)

        # Sanity check: physics instability (no object velocity checks)
        current_dof_vel = self.hand.data.joint_vel
        error_buf = (
            (torch.norm(current_eef_vel, dim=-1) > 100)
            | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
            | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
            | (torch.abs(current_dof_vel).mean(-1) > 200)
        )
        self.error_buf = error_buf

        s = scale_factor
        # Tracking error termination (no object checks, grace period = 20)
        failed_execute = (
            (
                (diff_thumb_tip_pos_dist > 0.04 / 0.7 * s)
                | (diff_index_tip_pos_dist > 0.045 / 0.7 * s)
                | (diff_middle_tip_pos_dist > 0.05 / 0.7 * s)
                | (diff_pinky_tip_pos_dist > 0.06 / 0.7 * s)
                | (diff_ring_tip_pos_dist > 0.06 / 0.7 * s)
                | (diff_level_1_pos_dist > 0.07 / 0.7 * s)
                | (diff_level_2_pos_dist > 0.08 / 0.7 * s)
            )
            & (self.running_progress_buf >= 20)  # Grace period 20 (residual uses 8)
        ) | error_buf

        # Success: reached end of trajectory
        succeeded = (
            self.episode_length_buf + 1 + 3 >= max_length
        ) & ~failed_execute

        terminated = failed_execute
        time_out = succeeded

        self.success_buf = succeeded
        self.failure_buf = failed_execute

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset environments with noisy default DOF pose and demo wrist state."""
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

        if self.use_pid_control:
            self.prev_pos_error[env_ids] = 0
            self.prev_rot_error[env_ids] = 0
            self.pos_error_integral[env_ids] = 0
            self.rot_error_integral[env_ids] = 0

        # Convert env_ids to tensor
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.long()
        else:
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Determine initial sequence index
        if self.random_state_init:
            seq_idx = torch.floor(
                self.demo_data["seq_len"][env_ids_tensor].float()
                * 0.99
                * torch.rand(len(env_ids_tensor), device=self.device)
            ).long()
        else:
            seq_idx = torch.zeros(len(env_ids_tensor), device=self.device, dtype=torch.long)

        # --- Reset hand DOF state with noisy default pose ---
        # Imitator uses noisy default pose (NOT IK-solved opt_dof_pos)
        # This encourages robustness during training
        n_envs_reset = len(env_ids_tensor)

        # Noise on default DOF positions (in demo order)
        noise_dof_pos = (
            torch.randn(n_envs_reset, self.dexhand.n_dofs, device=self.device)
            * ((self.dexhand_dof_upper_limits_demo - self.dexhand_dof_lower_limits_demo) / 8).unsqueeze(0)
        )
        dof_pos_demo = torch.clamp(
            self.dexhand_default_dof_pos.unsqueeze(0).expand(n_envs_reset, -1) + noise_dof_pos,
            self.dexhand_dof_lower_limits_demo.unsqueeze(0),
            self.dexhand_dof_upper_limits_demo.unsqueeze(0),
        )

        # Small random DOF velocities (in demo order), clamped to speed limits
        dof_vel_demo = torch.randn(n_envs_reset, self.dexhand.n_dofs, device=self.device) * 0.1
        # Speed limits in demo order for clamping
        speed_limits_demo = self.dexhand_dof_speed_limits[self.demo_to_isaaclab_dof_mapping]
        dof_vel_demo = torch.clamp(
            dof_vel_demo,
            -speed_limits_demo.unsqueeze(0),
            speed_limits_demo.unsqueeze(0),
        )

        # Reorder from demo order to IsaacLab order
        indices = self.demo_to_isaaclab_dof_mapping.unsqueeze(0).expand(n_envs_reset, -1)
        dof_pos = torch.zeros_like(dof_pos_demo)
        dof_vel = torch.zeros_like(dof_vel_demo)
        dof_pos.scatter_(1, indices, dof_pos_demo)
        dof_vel.scatter_(1, indices, dof_vel_demo)

        dof_pos = saturate(dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)

        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # Set initial position targets to match DOF state (prevents PD controller fighting)
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)

        # --- Reset wrist state from demo data + noise ---
        wrist_pos = self.demo_data["wrist_pos"][env_ids_tensor, seq_idx]
        wrist_pos = wrist_pos + torch.randn_like(wrist_pos) * 0.01

        wrist_rot_aa = self.demo_data["wrist_rot"][env_ids_tensor, seq_idx]
        wrist_rot_mat = aa_to_rotmat(wrist_rot_aa)
        # Add rotation noise (random axis, ~10 deg magnitude)
        noise_axis = torch.rand(n_envs_reset, 3, device=self.device)
        noise_axis = noise_axis / torch.norm(noise_axis, dim=-1, keepdim=True)
        noise_angle = torch.randn(n_envs_reset, 1, device=self.device) * (np.pi / 18)
        noise_rot = aa_to_rotmat(noise_axis * noise_angle)
        wrist_rot_mat = noise_rot @ wrist_rot_mat
        wrist_rot_quat = rotmat_to_quat(wrist_rot_mat)  # wxyz format (native IsaacLab)

        wrist_vel = self.demo_data["wrist_velocity"][env_ids_tensor, seq_idx]
        wrist_vel = wrist_vel + torch.randn_like(wrist_vel) * 0.01
        wrist_ang_vel = self.demo_data["wrist_angular_velocity"][env_ids_tensor, seq_idx]
        wrist_ang_vel = wrist_ang_vel + torch.randn_like(wrist_ang_vel) * 0.01

        root_state = torch.cat([
            wrist_pos + self.scene.env_origins[env_ids],
            wrist_rot_quat,
            wrist_vel,
            wrist_ang_vel,
        ], dim=-1)
        self.hand.write_root_state_to_sim(root_state, env_ids=env_ids)

        # Update episode length buffer to match demo sequence position
        self.episode_length_buf[env_ids] = seq_idx

        self._compute_intermediate_values()


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Scale tensor from [-1, 1] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower
