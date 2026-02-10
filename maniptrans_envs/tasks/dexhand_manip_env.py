"""
DexHand Manipulation Environment for IsaacLab.

Migrated from ManipTrans IsaacGym implementation to IsaacLab/IsaacSim 5.1.0.
"""

from __future__ import annotations

import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
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
# IsaacLab uses wxyz format; IsaacGym-trained imitators expect xyzw format.
def wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from wxyz (IsaacLab) to xyzw (IsaacGym) format."""
    return quat[..., [1, 2, 3, 0]]


def quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication for xyzw format (IsaacGym convention)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([x, y, z, w], dim=-1)


def quat_conjugate_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for xyzw format."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


# Configuration constants (must match original ManipTrans)
ROBOT_HEIGHT = 0.00214874


class ImitatorPolicy(torch.nn.Module):
    """Lightweight inference-only wrapper for loading frozen imitator RL Games checkpoints.

    Reconstructs just the actor MLP + input normalization from an RL Games
    continuous_a2c_logstd checkpoint. No critic, no sigma — only deterministic
    mu output for base action computation.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_units=(512, 256, 128)):
        super().__init__()
        # Running mean/std for input normalization (matches RL Games running_mean_std)
        self.running_mean = torch.nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.running_var = torch.nn.Parameter(torch.ones(obs_dim), requires_grad=False)

        # Actor MLP (matches RL Games a2c_network.actor_mlp)
        layers = []
        in_dim = obs_dim
        for h in hidden_units:
            layers.extend([torch.nn.Linear(in_dim, h), torch.nn.ELU()])
            in_dim = h
        self.actor_mlp = torch.nn.Sequential(*layers)
        self.mu = torch.nn.Linear(in_dim, action_dim)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic forward pass: normalize → MLP → mu."""
        obs = (obs - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        obs = torch.clamp(obs, -5.0, 5.0)
        x = self.actor_mlp(obs)
        return self.mu(x)

    @classmethod
    def from_rl_games_checkpoint(cls, path: str, obs_dim: int, action_dim: int,
                                  hidden_units=(512, 256, 128), device="cuda"):
        """Load from an RL Games checkpoint file.

        Maps RL Games state dict keys to our simplified model:
          a2c_network.actor_mlp.{i}.weight/bias → actor_mlp.{i}.weight/bias
          a2c_network.mu.weight/bias → mu.weight/bias
          running_mean_std.running_mean → running_mean
          running_mean_std.running_var → running_var
        """
        model = cls(obs_dim, action_dim, hidden_units)
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["model"]

        new_state = {}
        for k, v in state.items():
            if k.startswith("a2c_network.actor_mlp."):
                new_key = k.replace("a2c_network.", "")
                new_state[new_key] = v
            elif k.startswith("a2c_network.mu."):
                new_key = k.replace("a2c_network.", "")
                new_state[new_key] = v
            elif k == "running_mean_std.running_mean":
                new_state["running_mean"] = v
            elif k == "running_mean_std.running_var":
                new_state["running_var"] = v
            # Skip: sigma, value, critic, count

        model.load_state_dict(new_state, strict=False)
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        print(f"[INFO] Loaded imitator policy from {path}")
        print(f"[INFO]   obs_dim={obs_dim}, action_dim={action_dim}, units={list(hidden_units)}")
        return model


@configclass
class DexHandManipEnvCfg(DirectRLEnvCfg):
    """Configuration for the DexHand manipulation environment."""

    # Environment settings
    decimation = 2
    episode_length_s = 20.0  # Will be overridden by max_episode_length

    # Action/Observation space (default values for Inspire hand with 12 DOFs)
    # Will be recalculated in __init__ based on actual hand configuration
    # PPO outputs residual actions only; base actions come from frozen imitator
    action_space = 18  # 6 wrist + 12 dofs (residual only) for inspire
    observation_space = 794  # Computed dynamically in __init__ based on dexhand configuration
    state_space = 0

    # Simulation
    # Original ManipTrans: dt=1/60, substeps=2 → effective physics dt = 1/120
    # IsaacLab equivalent: dt=1/120, decimation=2 → policy runs at 1/60, physics at 1/120
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics_material=RigidBodyMaterialCfg(
            static_friction=4.0,
            dynamic_friction=4.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,        # Match original ManipTrans
            friction_correlation_distance=0.0005,   # Match original ManipTrans
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
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
    use_quat_rot: bool = False  # Original ManipTrans default: false
    action_scale: float = 1.0
    act_moving_average: float = 0.4
    translation_scale: float = 1.0
    orientation_scale: float = 0.1
    use_pid_control: bool = False
    use_isaacgym_imitator: bool = True  # True: obs quaternions in xyzw (IsaacGym); False: wxyz (IsaacLab)

    # Frozen imitator checkpoints (for residual learning)
    # When provided, the env loads a frozen imitator to produce base actions.
    # PPO then only outputs the residual correction (action_dim = single hand action).
    # When None, base_action = 0 and PPO learns from scratch (effectively no residual).
    rh_base_checkpoint: str | None = None  # Right hand imitator checkpoint
    lh_base_checkpoint: str | None = None  # Left hand imitator checkpoint

    # Training settings
    training: bool = True
    obs_future_length: int = 1  # Match original ManipTrans (obsFutureLength: 1)
    rollout_state_init: bool = False
    random_state_init: bool = True

    # Tightening curriculum
    tighten_method: str = "exp_decay"  # Match original ManipTrans (tightenMethod: "exp_decay")
    tighten_factor: float = 0.7
    tighten_steps: int = 3200

    # Rollout settings (optional)
    rollout_len: int | None = None
    rollout_begin: int | None = None

    # Max episode length
    max_episode_length: int = 1200

    # Domain Randomization / Curriculum (matching original ManipTrans ResDexHand.yaml)
    enable_gravity_curriculum: bool = True   # Ramp gravity from 0 → -9.81
    gravity_curriculum_steps: int = 1920     # Steps to reach full gravity
    enable_friction_curriculum: bool = True  # Ramp object friction from 3× → 1×
    friction_curriculum_steps: int = 1920    # Steps to reach normal friction
    friction_curriculum_init_scale: float = 3.0  # Initial friction multiplier
    dr_frequency: int = 32                   # How often to update DR (in env steps)


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
        # Action space: PPO outputs residual action only.
        # Base action comes from frozen imitator (always force control: 6 + n_dofs).
        # Residual action dims depend on control mode:
        #   Force control (default): 6 (wrist force/torque) + n_dofs
        #   PID control: 9 (pos_error(3) + rot6d(6)) + n_dofs
        #   Quat rot (deprecated): 1 + 6 + n_dofs
        if cfg.use_pid_control:
            residual_root_dims = 9  # pos_error(3) + rot6d(6)
        elif use_quat_rot:
            residual_root_dims = 1 + 6  # deprecated
        else:
            residual_root_dims = 6  # force(3) + torque(3)
        residual_action_dim = residual_root_dims + n_dofs
        self._residual_root_dims = residual_root_dims
        cfg.num_actions = residual_action_dim  # Residual only (not doubled)
        cfg.action_space = cfg.num_actions

        # Target observation dimension
        n_contact_bodies = len(self.dexhand.contact_body_names)  # 5 for most hands
        n_all_bodies = self.dexhand.n_bodies  # All bodies including wrist (for obj_to_joints)
        n_joint_bodies = self.dexhand.n_bodies - 1  # Bodies excluding wrist for joint tracking
        target_obs_dim = (
            128  # BPS features
            + n_all_bodies  # obj_to_joints: distance from obj to ALL bodies (not just fingertips)
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
                + n_contact_bodies  # gt_tips_distance (per frame)
            )
            * cfg.obs_future_length
        )

        # Proprioception dimension: q, cos_q, sin_q, base_state (pos zeroed)
        prop_obs_dim = n_dofs * 3 + 13  # q, cos(q), sin(q), base_state

        # Privileged observations (matching original ManipTrans Dict obs "privileged" key)
        # dq(n_dofs) + manip_obj_pos_rel(3) + manip_obj_quat(4) + manip_obj_vel(3)
        # + manip_obj_ang_vel(3) + tip_force_with_mag(n_contact*4) + manip_obj_com_rel(3) + manip_obj_weight(1)
        priv_obs_dim = n_dofs + 3 + 4 + 3 + 3 + n_contact_bodies * 4 + 3 + 1

        cfg.num_observations = prop_obs_dim + priv_obs_dim + target_obs_dim
        cfg.observation_space = cfg.num_observations  # Sync with num_observations for RL framework
        cfg.num_states = 0  # No asymmetric states for now

        self._priv_obs_dim = priv_obs_dim

        # Compute imitator observation dimension (for loading frozen imitator)
        n_joint_bodies_imit = self.dexhand.n_bodies - 1
        imit_target_per_frame = (
            3 + 3 + 3 + 4 + 4 + 3 + 3  # wrist: delta_pos, vel, delta_vel, quat, delta_quat, ang_vel, delta_ang_vel
            + n_joint_bodies_imit * 3    # delta_joints_pos
            + n_joint_bodies_imit * 3    # joints_vel
            + n_joint_bodies_imit * 3    # delta_joints_vel
        )
        self._imitator_obs_dim = (n_dofs * 3 + 13) + n_dofs + imit_target_per_frame  # prop + priv + target (future=1)
        self._imitator_action_dim = 6 + n_dofs  # imitator always uses force control: 6 wrist + n_dofs

        print(f"[INFO] Manip obs dims: prop={prop_obs_dim}, priv={priv_obs_dim}, "
              f"target={target_obs_dim} (bps=128, obj2joints={n_all_bodies}, "
              f"per_frame*{cfg.obs_future_length}), total={cfg.num_observations}")
        print(f"[INFO] Manip action dim: {cfg.num_actions} (residual only, "
              f"root({residual_root_dims})+{n_dofs}dofs)")
        ckpt_side = cfg.rh_base_checkpoint if cfg.side == "right" else cfg.lh_base_checkpoint
        print(f"[INFO] Imitator checkpoint: {ckpt_side or 'None (base_action=0, no residual benefit)'}")
        print(f"[INFO] Imitator obs_dim={self._imitator_obs_dim}, action_dim={self._imitator_action_dim}")

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

        # Load frozen imitator policy for residual learning
        ckpt_path = cfg.rh_base_checkpoint if cfg.side == "right" else cfg.lh_base_checkpoint
        if ckpt_path is not None and os.path.exists(ckpt_path):
            self.imitator_policy = ImitatorPolicy.from_rl_games_checkpoint(
                path=ckpt_path,
                obs_dim=self._imitator_obs_dim,
                action_dim=self._imitator_action_dim,
                hidden_units=(512, 256, 128),
                device=str(self.device),
            )
            self._has_imitator = True
        else:
            self.imitator_policy = None
            self._has_imitator = False
            if ckpt_path is not None:
                print(f"[WARNING] Imitator checkpoint not found: {ckpt_path}")
            print("[INFO] No imitator loaded — base_action will be zeros (no residual benefit)")

        # Pre-allocate base action buffer
        self._base_action_buf = torch.zeros(
            self.num_envs, self._imitator_action_dim, device=self.device
        )

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

        # Create DOF index mapping: demo data order -> IsaacLab order
        # Demo data uses dexhand.dof_names order (depth-first)
        # IsaacLab may use different order (breadth-first)
        self.joint_name_to_idx = {}
        for i, name in enumerate(actual_joint_names):
            self.joint_name_to_idx[name] = i

        # Build mapping from demo data DOF order to IsaacLab DOF order
        self.demo_to_isaaclab_dof_mapping = []
        for demo_dof_name in self.dexhand.dof_names:
            if demo_dof_name in self.joint_name_to_idx:
                self.demo_to_isaaclab_dof_mapping.append(self.joint_name_to_idx[demo_dof_name])
            else:
                print(f"[WARNING] DOF '{demo_dof_name}' not found in IsaacLab joints!")
                print(f"[WARNING] Available joints: {actual_joint_names}")
                self.demo_to_isaaclab_dof_mapping.append(0)  # Fallback

        self.demo_to_isaaclab_dof_mapping = torch.tensor(
            self.demo_to_isaaclab_dof_mapping, device=self.device, dtype=torch.long
        )
        print(f"[INFO] Demo to IsaacLab DOF mapping: {self.demo_to_isaaclab_dof_mapping.tolist()}")

        # Also build reverse mapping: IsaacLab order -> demo data order
        self.isaaclab_to_demo_dof_mapping = torch.zeros_like(self.demo_to_isaaclab_dof_mapping)
        for demo_idx in range(len(self.demo_to_isaaclab_dof_mapping)):
            isaaclab_idx = self.demo_to_isaaclab_dof_mapping[demo_idx].item()
            self.isaaclab_to_demo_dof_mapping[isaaclab_idx] = demo_idx
        print(f"[INFO] IsaacLab to Demo DOF mapping: {self.isaaclab_to_demo_dof_mapping.tolist()}")

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

        # Update the actual number of bodies and joints
        self._actual_num_bodies = len(actual_body_names)
        self._actual_num_joints = len(actual_joint_names)

        # Find wrist body index by name (instead of assuming index 0)
        wrist_body_name = self.dexhand.to_dex("wrist")[0]
        if wrist_body_name in self.body_name_to_idx:
            self.wrist_body_idx = self.body_name_to_idx[wrist_body_name]
        else:
            print(f"[WARNING] Wrist body '{wrist_body_name}' not found in articulation, falling back to index 0")
            self.wrist_body_idx = 0
        print(f"[INFO] Wrist body index: {self.wrist_body_idx} (name: {wrist_body_name})")

        # Build body reorder mapping: dexhand body order (excluding wrist) -> IsaacLab body indices
        # This is needed to compare FK body positions with demo mano_joints data
        wrist_name = self.dexhand.to_dex("wrist")[0]
        self.demo_body_to_isaaclab_indices = []
        for body_name in self.dexhand.body_names:
            if body_name == wrist_name:
                continue  # Skip wrist, mano_joints excludes it
            if body_name in self.body_name_to_idx:
                self.demo_body_to_isaaclab_indices.append(self.body_name_to_idx[body_name])
            else:
                print(f"[WARNING] Body '{body_name}' not found in IsaacLab bodies!")
                self.demo_body_to_isaaclab_indices.append(0)
        self.demo_body_to_isaaclab_indices = torch.tensor(
            self.demo_body_to_isaaclab_indices, device=self.device, dtype=torch.long
        )
        print(f"[INFO] Demo body to IsaacLab indices (excl wrist): {self.demo_body_to_isaaclab_indices.tolist()}")

        # Contact body indices in IsaacLab order (for fingertip contact force)
        self.contact_body_indices = []
        for name in self.dexhand.contact_body_names:
            if name in self.body_name_to_idx:
                self.contact_body_indices.append(self.body_name_to_idx[name])
            else:
                print(f"[WARNING] Contact body '{name}' not found in articulation!")
        print(f"[INFO] Contact body indices: {self.contact_body_indices}")

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

        # Read joint limits from URDF (in IsaacLab order, which may differ from dexhand.dof_names order)
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

        # Build limits in IsaacLab joint order (actual_joint_names)
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

        # Curriculum learning step counter
        self.global_step_count = 0

        # Best rollout tracking (for curriculum)
        self.best_rollout_len = 0
        self.best_rollout_begin = 0

        # Domain randomization buffers
        self._dr_initialized = False  # Will be initialized on first step (after physics views ready)
        self.gravity_scale = 0.0 if (self.training and self.cfg.enable_gravity_curriculum) else 1.0
        self.current_friction_scale = self.cfg.friction_curriculum_init_scale if (self.training and self.cfg.enable_friction_curriculum) else 1.0
        # Object gravity compensation force buffer (num_envs, 1, 3) for RigidObject
        self.object_gravity_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.object_gravity_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.manip_obj_mass = torch.zeros(self.num_envs, device=self.device)  # filled on first step
        self.manip_obj_com = torch.zeros(self.num_envs, 3, device=self.device)  # object center of mass offset

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
                    max_depenetration_velocity=1000.0,  # Match original ManipTrans global setting
                    linear_damping=20.0,   # Prevent flying (same as original)
                    angular_damping=20.0,  # Prevent flying (same as original)
                    max_linear_velocity=50.0,
                    max_angular_velocity=100.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,   # Match original ManipTrans
                    rest_offset=0.0,        # Match original ManipTrans
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=self.dexhand.self_collision,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,  # Match original (0 = friction not solved)
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
            self.object_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object",
                spawn=UsdFileCfg(
                    usd_path=usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=1000.0,  # Match original ManipTrans
                        max_linear_velocity=50.0,           # Match original: prevent object flying
                        max_angular_velocity=100.0,         # Match original: prevent object spinning
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.005,   # Match original ManipTrans
                        rest_offset=0.0,        # Match original ManipTrans
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        density=200.0,  # Match original: average density of 3D-printed models
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
                        max_depenetration_velocity=1000.0,
                        max_linear_velocity=50.0,
                        max_angular_velocity=100.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        density=200.0,
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

    def _compute_imitator_obs(self) -> torch.Tensor:
        """Compute observations in imitator format (for frozen base policy).

        Returns flat tensor matching the imitator env's observation structure:
          [prop_obs(3*n_dofs+13), priv_obs(n_dofs), target_obs(23+(n_bodies-1)*9)]
        Uses wxyz quaternions (IsaacLab native, matching IsaacLab-trained imitator).
        """
        nE = self.num_envs

        # --- Proprioception (same as manip env) ---
        zeroed_root_state = torch.cat([
            torch.zeros_like(self.hand_root_state[:, :3]),
            self.hand_root_state[:, 3:],
        ], dim=-1)
        prop_obs = torch.cat([
            self.hand_dof_pos,
            torch.cos(self.hand_dof_pos),
            torch.sin(self.hand_dof_pos),
            zeroed_root_state,
        ], dim=-1)

        # --- Privileged (joint velocities only, no object data) ---
        priv_obs = self.hand_dof_vel

        # --- Target observations (hand only, future_length=1) ---
        cur_idx = self.episode_length_buf + 1
        cur_idx = torch.clamp(cur_idx, torch.zeros_like(self.demo_data["seq_len"]),
                              self.demo_data["seq_len"] - 1)
        # future_length=1: just single frame
        cur_idx_2d = cur_idx.unsqueeze(-1)  # (nE, 1)
        cur_idx_2d = torch.clamp(cur_idx_2d, max=self.demo_data["seq_len"].unsqueeze(-1) - 1)

        def indicing(data, idx):
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)

        # Wrist target
        target_wrist_pos = indicing(self.demo_data["wrist_pos"], cur_idx_2d)
        cur_wrist_pos = self.hand_root_state[:, :3]
        delta_wrist_pos = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1)

        target_wrist_vel = indicing(self.demo_data["wrist_velocity"], cur_idx_2d)
        cur_wrist_vel = self.hand_root_state[:, 7:10]
        wrist_vel = target_wrist_vel.reshape(nE, -1)
        delta_wrist_vel = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1)

        target_wrist_rot = indicing(self.demo_data["wrist_rot"], cur_idx_2d)
        cur_wrist_rot_wxyz = self.hand_root_state[:, 3:7]
        wrist_quat_wxyz = aa_to_quat(target_wrist_rot.reshape(nE, -1))  # wxyz
        wrist_quat = wrist_quat_wxyz
        delta_wrist_quat = quat_mul(
            cur_wrist_rot_wxyz, quat_conjugate(wrist_quat)
        ).reshape(nE, -1)
        wrist_quat = wrist_quat.reshape(nE, -1)

        target_wrist_ang_vel = indicing(self.demo_data["wrist_angular_velocity"], cur_idx_2d)
        cur_wrist_ang_vel = self.hand_root_state[:, 10:13]
        wrist_ang_vel = target_wrist_ang_vel.reshape(nE, -1)
        delta_wrist_ang_vel = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        # Joint targets (excluding wrist)
        target_joints_pos = indicing(self.demo_data["mano_joints"], cur_idx_2d).reshape(nE, 1, -1, 3)
        cur_joints_pos = self.hand_body_pos  # (nE, n_bodies-1, 3)
        delta_joints_pos = (target_joints_pos - cur_joints_pos[:, None]).reshape(nE, -1)

        target_joints_vel = indicing(self.demo_data["mano_joints_velocity"], cur_idx_2d).reshape(nE, 1, -1, 3)
        cur_joints_vel = self.hand_body_vel
        joints_vel = target_joints_vel.reshape(nE, -1)
        delta_joints_vel = (target_joints_vel - cur_joints_vel[:, None]).reshape(nE, -1)

        target_obs = torch.cat([
            delta_wrist_pos, wrist_vel, delta_wrist_vel,
            wrist_quat, delta_wrist_quat,
            wrist_ang_vel, delta_wrist_ang_vel,
            delta_joints_pos, joints_vel, delta_joints_vel,
        ], dim=-1)

        return torch.cat([prop_obs, priv_obs, target_obs], dim=-1)

    @torch.no_grad()
    def _get_base_action(self) -> torch.Tensor:
        """Get base action from frozen imitator policy.

        Returns tensor of shape (num_envs, imitator_action_dim).
        If no imitator is loaded, returns zeros.
        """
        if self._has_imitator:
            imit_obs = self._compute_imitator_obs()
            imit_obs = torch.nan_to_num(imit_obs, nan=0.0)
            self._base_action_buf = self.imitator_policy(imit_obs)
        else:
            self._base_action_buf.zero_()
        return self._base_action_buf

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step.

        Computes DOF targets and wrist forces/torques once per policy step,
        matching original ManipTrans where pre_physics_step runs once with dt=1/60.
        _apply_action() then just writes these pre-computed values to the sim.

        Actions from PPO are the residual correction only. Base actions come from
        the frozen imitator policy. Combined: total_action = base + residual.
        """
        self.actions = actions.clone()

        # Get base action from frozen imitator (or zeros if no imitator)
        # Base imitator always uses force control: [force(3), torque(3), dof(n_dofs)]
        base_action = self._get_base_action()
        # PPO output is the residual, scaled by 2 (matching original residual scaling)
        residual_action = self.actions * 2

        # DOF position targets (in demo/policy order): base + residual
        # Base DOFs always start at index 6 (after force+torque)
        # Residual DOFs start at _residual_root_dims (6 for force, 9 for PID)
        res_root = self._residual_root_dims
        dof_pos_demo_order = (
            1.0 * base_action[:, 6:6 + self.dexhand.n_dofs]
            + residual_action[:, res_root:res_root + self.dexhand.n_dofs]
        )
        dof_pos_demo_order = torch.clamp(dof_pos_demo_order, -1, 1)

        # Reorder from demo order to IsaacLab order using scatter
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

        # Compute wrist forces/torques (once per policy step)
        # Use step_dt (= physics_dt * decimation = 1/60) to match original dt=1/60
        if self.use_pid_control:
            self._apply_pid_control(base_action, residual_action)
        else:
            self._apply_force_control(base_action, residual_action)

        # Domain randomization: gravity & friction curriculum (once per policy step)
        self._apply_domain_randomization()

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

        # Apply gravity compensation force on object (implements gravity curriculum)
        if self.cfg.enable_gravity_curriculum and self.training:
            self.object.set_external_force_and_torque(
                forces=self.object_gravity_force,
                torques=self.object_gravity_torque,
                body_ids=None,
                env_ids=None,
                is_global=True,
            )

    def _init_domain_randomization(self):
        """Initialize DR state from physics views (called once after first sim step)."""
        # Get object mass from PhysX
        try:
            masses = self.object.root_physx_view.get_masses()  # (num_envs, 1)
            self.manip_obj_mass = masses.squeeze(-1).to(self.device).clamp(max=0.5)  # cap at 500g like original
            print(f"[INFO] Object mass: {self.manip_obj_mass[0].item():.4f} kg (capped at 0.5)")
        except Exception as e:
            print(f"[WARNING] Could not get object mass: {e}, using 0.1 kg")
            self.manip_obj_mass[:] = 0.1

        # Get object center of mass
        try:
            coms = self.object.root_physx_view.get_coms().to(self.device)  # (num_envs, 7) or (num_envs, 3)
            self.manip_obj_com = coms[:, :3]  # take only xyz position
            print(f"[INFO] Object COM offset: {self.manip_obj_com[0].tolist()}")
        except Exception as e:
            print(f"[WARNING] Could not get object COM: {e}, using zeros")
            self.manip_obj_com[:] = 0.0

        # Set object friction to match original ManipTrans (2.0, not the global 4.0)
        # Original: element.friction = 2.0 (compensates for missing skin deformation friction)
        # The global physics_material (4.0) is for the hand; object needs different value.
        try:
            mat_props = self.object.root_physx_view.get_material_properties()  # (num_envs, num_shapes, 3)
            # mat_props[..., 0] = static_friction, [..1] = dynamic_friction, [..2] = restitution
            # Override to match original ManipTrans object friction = 2.0
            mat_props[:, :, 0] = 2.0  # static friction
            mat_props[:, :, 1] = 2.0  # dynamic friction
            env_indices = torch.arange(self.num_envs, device="cpu")
            self.object.root_physx_view.set_material_properties(mat_props, env_indices)
            # Now store these as "original" for friction curriculum
            self._original_obj_static_friction = mat_props[:, :, 0].clone()
            self._original_obj_dynamic_friction = mat_props[:, :, 1].clone()
            print(f"[INFO] Object friction set to match original ManipTrans: static=2.00, dynamic=2.00")
        except Exception as e:
            print(f"[WARNING] Could not set object material properties: {e}")
            self._original_obj_static_friction = None
            self._original_obj_dynamic_friction = None

        self._dr_initialized = True

    def _apply_domain_randomization(self):
        """Apply gravity curriculum and friction curriculum (matching original ManipTrans DR).

        Gravity curriculum: effective gravity ramps from 0 → -9.81 over gravity_curriculum_steps.
        Implemented via compensating upward force on object (hand has disable_gravity=True).

        Friction curriculum: object friction starts at init_scale× and decays to 1× over friction_curriculum_steps.
        """
        if not self.training:
            return

        if not self._dr_initialized:
            self._init_domain_randomization()

        step = self.global_step_count

        # Only update at DR frequency (matching original frequency: 32)
        if step % self.cfg.dr_frequency != 0 and step > 0:
            return

        # --- Gravity curriculum ---
        if self.cfg.enable_gravity_curriculum:
            # gravity_scale: 0 → 1 over gravity_curriculum_steps
            # Matches original: scaling operation + linear_decay schedule + const_scale init_value=0
            self.gravity_scale = min(step, self.cfg.gravity_curriculum_steps) / max(self.cfg.gravity_curriculum_steps, 1)
            # Compensating upward force: neutralizes (1 - gravity_scale) fraction of gravity
            # When gravity_scale=0: full compensation (object floats), gravity_scale=1: no compensation
            compensation_z = self.manip_obj_mass * 9.81 * (1.0 - self.gravity_scale)
            self.object_gravity_force[:, 0, 2] = compensation_z

        # --- Friction curriculum ---
        if self.cfg.enable_friction_curriculum and self._original_obj_static_friction is not None:
            # friction_scale: init_scale → 1 over friction_curriculum_steps
            # Matches original: scaling + linear_decay + const_scale init_value=3
            # sample = init_value * sched_scaling + 1 * (1 - sched_scaling)
            # sched_scaling = 1 - min(step, N) / N  (linear_decay)
            sched_scaling = 1.0 - min(step, self.cfg.friction_curriculum_steps) / max(self.cfg.friction_curriculum_steps, 1)
            friction_scale = self.cfg.friction_curriculum_init_scale * sched_scaling + 1.0 * (1.0 - sched_scaling)
            self.current_friction_scale = friction_scale

            try:
                mat_props = self.object.root_physx_view.get_material_properties()
                mat_props[:, :, 0] = self._original_obj_static_friction * friction_scale
                mat_props[:, :, 1] = self._original_obj_dynamic_friction * friction_scale
                env_indices = torch.arange(self.num_envs, device="cpu")
                self.object.root_physx_view.set_material_properties(mat_props, env_indices)
            except Exception:
                pass  # Silently skip if API not available

    def _apply_pid_control(self, base_action, residual_action):
        """Apply PID-based wrist control.

        Base imitator provides force-control actions: [force(3), torque(3), ...]
        Residual PPO provides PID actions: [pos_error(3), rot6d(6), ...]
        Combined: final_force = base_force + PID(residual_pos_error)
        """
        dt = self.step_dt

        # Residual PID: position error → PID → force correction
        position_error = residual_action[:, 0:3]
        self.pos_error_integral += position_error * dt
        self.pos_error_integral = torch.clamp(self.pos_error_integral, -1, 1)
        pos_derivative = (position_error - self.prev_pos_error) / dt
        pid_force = (
            self.Kp_pos * position_error
            + self.Ki_pos * self.pos_error_integral
            + self.Kd_pos * pos_derivative
        )
        self.prev_pos_error = position_error
        # Add base imitator force
        force = base_action[:, 0:3] * dt * self.translation_scale * 500 + pid_force

        # Residual PID: rotation error → PID → torque correction
        rotation_error = residual_action[:, 3:9]
        rotation_error = rot6d_to_aa(rotation_error)
        self.rot_error_integral += rotation_error * dt
        self.rot_error_integral = torch.clamp(self.rot_error_integral, -1, 1)
        rot_derivative = (rotation_error - self.prev_rot_error) / dt
        pid_torque = (
            self.Kp_rot * rotation_error
            + self.Ki_rot * self.rot_error_integral
            + self.Kd_rot * rot_derivative
        )
        self.prev_rot_error = rotation_error
        # Add base imitator torque
        torque = base_action[:, 3:6] * dt * self.orientation_scale * 200 + pid_torque

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
        # Use step_dt (= physics_dt * decimation = 1/60) to match original dt=1/60
        dt = self.step_dt
        force = (
            1.0 * (base_action[:, 0:3] * dt * self.translation_scale * 500)
            + (residual_action[:, 0:3] * dt * self.translation_scale * 500)
        )
        torque = (
            1.0 * (base_action[:, 3:6] * dt * self.orientation_scale * 200)
            + (residual_action[:, 3:6] * dt * self.orientation_scale * 200)
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

        # Proprioception observations (zero out base position, matching original)
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

        # Privileged observations (matching original ManipTrans Dict obs "privileged" key)
        # manip_obj_pos relative to hand base position
        manip_obj_pos_rel = self.object_pos - self.hand_root_state[:, :3]
        # manip_obj_quat
        manip_obj_quat = self.object_rot
        # manip_obj_com relative to hand base (world-frame COM)
        try:
            cur_com_pos = (
                quat_to_rotmat(self.object_rot)
                @ self.manip_obj_com.unsqueeze(-1)
            ).squeeze(-1) + self.object_pos
            manip_obj_com_rel = cur_com_pos - self.hand_root_state[:, :3]
        except Exception:
            manip_obj_com_rel = manip_obj_pos_rel  # fallback if COM not available
        # tip_force with magnitude (n_contact_bodies * 4)
        tip_force = self.net_contact_forces[:, self.contact_body_indices, :]  # (nE, 5, 3)
        tip_force_with_mag = torch.cat([
            tip_force,
            torch.norm(tip_force, dim=-1, keepdim=True),
        ], dim=-1).reshape(self.num_envs, -1)  # (nE, 5*4=20)
        # manip_obj_weight
        manip_obj_weight = (self.manip_obj_mass * 9.81 * self.gravity_scale).unsqueeze(-1)

        priv_obs = torch.cat([
            self.hand_dof_vel,                   # dq (n_dofs)
            manip_obj_pos_rel,                   # obj pos relative to hand (3)
            manip_obj_quat,                      # obj quat (4)
            self.object_linvel,                  # obj linear vel (3)
            self.object_angvel,                  # obj angular vel (3)
            tip_force_with_mag,                  # fingertip contact forces (5*4=20)
            manip_obj_com_rel,                   # obj COM relative to hand (3)
            manip_obj_weight,                    # effective obj weight (1)
        ], dim=-1)

        # Target observations (from demo data)
        target_obs = self._compute_target_observations()

        # Combine observations: prop + priv + target (matching original Dict obs layout)
        obs = torch.cat([prop_obs, priv_obs, target_obs], dim=-1)

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

        # Reorder to demo order: we want result[demo_idx] = isaaclab[isaaclab_idx]
        # demo_to_isaaclab_dof_mapping[demo_idx] = isaaclab_idx
        # So we gather using demo_to_isaaclab_dof_mapping
        self.hand_dof_pos = hand_dof_pos_isaaclab[:, self.demo_to_isaaclab_dof_mapping]
        self.hand_dof_vel = hand_dof_vel_isaaclab[:, self.demo_to_isaaclab_dof_mapping]
        root_pos = self.hand.data.root_pos_w - self.scene.env_origins
        root_quat = self.hand.data.root_quat_w
        root_lin_vel = self.hand.data.root_lin_vel_w
        root_ang_vel = self.hand.data.root_ang_vel_w

        # Check for physics instability (hand flying away or NaN)
        if torch.isnan(root_pos).any() or (root_pos.abs() > 10.0).any():
            nan_count = torch.isnan(root_pos).sum().item()
            far_count = (root_pos.abs() > 10.0).sum().item()
            print(f"[WARNING] Physics instability! NaN: {nan_count}, Far: {far_count}")
            root_pos = torch.nan_to_num(root_pos, nan=0.0)
            root_lin_vel = torch.nan_to_num(root_lin_vel, nan=0.0)
            root_ang_vel = torch.nan_to_num(root_ang_vel, nan=0.0)

        self.hand_root_state = torch.cat([
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel,
        ], dim=-1)

        # All body positions in demo order (excluding wrist) for reward computation
        all_body_pos_isaaclab = self.hand.data.body_pos_w - self.scene.env_origins.unsqueeze(1)
        self.hand_body_pos = all_body_pos_isaaclab[:, self.demo_body_to_isaaclab_indices]  # (nE, n_bodies-1, 3)

        # All body positions in demo order (INCLUDING wrist) for obj_to_joints
        # Original uses joints_state which includes ALL body positions (n_bodies)
        all_body_demo_indices = [self.body_name_to_idx[name] for name in self.dexhand.body_names
                                 if name in self.body_name_to_idx]
        all_body_demo_indices = torch.tensor(all_body_demo_indices, device=self.device, dtype=torch.long)
        self.all_hand_body_pos = all_body_pos_isaaclab[:, all_body_demo_indices]  # (nE, n_bodies, 3)

        # Body velocities: try body_vel_w (6D) first, fallback to body_link_vel_w
        try:
            all_body_vel = self.hand.data.body_vel_w  # (num_envs, num_bodies, 6)
        except AttributeError:
            all_body_vel = self.hand.data.body_link_vel_w  # newer IsaacLab versions
        self.hand_body_vel = all_body_vel[:, self.demo_body_to_isaaclab_indices, :3]  # linear vel only

        # Fingertip data
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.fingertip_indices]
        self.fingertip_pos -= self.scene.env_origins.unsqueeze(1)

        # Object data
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        # Contact forces for reward computation
        try:
            net_cf = self.hand.root_physx_view.get_net_contact_forces(dt=self.physics_dt)
            self.net_contact_forces = net_cf.view(self.num_envs, -1, 3)
        except Exception:
            self.net_contact_forces = torch.zeros(
                self.num_envs, self._actual_num_bodies, 3, device=self.device
            )

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
        cur_wrist_rot_wxyz = self.hand_root_state[:, 3:7]  # wxyz from IsaacLab
        wrist_quat_wxyz = aa_to_quat(target_wrist_rot.reshape(nE * nF, -1))  # wxyz from pytorch3d
        if self.cfg.use_isaacgym_imitator:
            # Convert to xyzw for IsaacGym-trained imitator
            cur_wrist_rot = wxyz_to_xyzw(cur_wrist_rot_wxyz)
            wrist_quat = wxyz_to_xyzw(wrist_quat_wxyz)
            delta_wrist_quat = quat_mul_xyzw(
                cur_wrist_rot[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
                quat_conjugate_xyzw(wrist_quat),
            ).reshape(nE, -1)
        else:
            # Keep wxyz for IsaacLab-trained imitator
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
        cur_joints_pos = self.hand_body_pos  # (nE, n_bodies-1, 3) already in demo order
        delta_joints_pos = (target_joints_pos - cur_joints_pos[:, None]).reshape(nE, -1)

        target_joints_vel = indicing(self.demo_data["mano_joints_velocity"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joints_vel = self.hand_body_vel  # (nE, n_bodies-1, 3) already in demo order
        joints_vel = target_joints_vel.reshape(nE, -1)
        delta_joints_vel = (target_joints_vel - cur_joints_vel[:, None]).reshape(nE, -1)

        # Object targets
        target_obj_transf = indicing(self.demo_data["obj_trajectory"], cur_idx)
        target_obj_transf = target_obj_transf.reshape(nE * nF, 4, 4)
        delta_manip_obj_pos = (
            target_obj_transf[:, :3, 3].reshape(nE, nF, -1) - self.object_pos[:, None]
        ).reshape(nE, -1)

        target_obj_vel = indicing(self.demo_data["obj_velocity"], cur_idx)
        manip_obj_vel = target_obj_vel.reshape(nE, -1)
        delta_manip_obj_vel = (target_obj_vel - self.object_linvel[:, None]).reshape(nE, -1)

        manip_obj_quat_wxyz = rotmat_to_quat(target_obj_transf[:, :3, :3])  # wxyz from pytorch3d
        if self.cfg.use_isaacgym_imitator:
            # Convert to xyzw for IsaacGym-trained imitator
            manip_obj_quat = wxyz_to_xyzw(manip_obj_quat_wxyz)
            cur_obj_rot_xyzw = wxyz_to_xyzw(self.object_rot)
            delta_manip_obj_quat = quat_mul_xyzw(
                cur_obj_rot_xyzw[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
                quat_conjugate_xyzw(manip_obj_quat),
            ).reshape(nE, -1)
        else:
            # Keep wxyz for IsaacLab-trained imitator
            manip_obj_quat = manip_obj_quat_wxyz
            delta_manip_obj_quat = quat_mul(
                self.object_rot[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
                quat_conjugate(manip_obj_quat),
            ).reshape(nE, -1)
        manip_obj_quat = manip_obj_quat.reshape(nE, -1)

        target_obj_ang_vel = indicing(self.demo_data["obj_angular_velocity"], cur_idx)
        manip_obj_ang_vel = target_obj_ang_vel.reshape(nE, -1)
        delta_manip_obj_ang_vel = (target_obj_ang_vel - self.object_angvel[:, None]).reshape(nE, -1)

        # Object to joints distance (ALL bodies, not just fingertips, matching original)
        obj_to_joints = torch.norm(
            self.object_pos[:, None] - self.all_hand_body_pos, dim=-1
        ).reshape(nE, -1)

        # Tips distance from demo
        gt_tips_distance = indicing(self.demo_data["tips_distance"], cur_idx).reshape(nE, -1)

        # BPS features
        bps = self.obj_bps

        # Concatenate all target observations
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
            delta_manip_obj_pos,
            manip_obj_vel,
            delta_manip_obj_vel,
            manip_obj_quat,
            delta_manip_obj_quat,
            manip_obj_ang_vel,
            delta_manip_obj_ang_vel,
            obj_to_joints,
            gt_tips_distance,
            bps,
        ], dim=-1)

        return target_obs

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on imitation error (matching original ManipTrans 19-term reward)."""
        cur_idx = self.episode_length_buf
        env_arange = torch.arange(self.num_envs, device=self.device)

        # --- Current states ---
        current_eef_pos = self.hand_root_state[:, :3]
        current_eef_quat = self.hand_root_state[:, 3:7]
        current_eef_vel = self.hand_root_state[:, 7:10]
        current_eef_ang_vel = self.hand_root_state[:, 10:13]
        current_obj_pos = self.object_pos
        current_obj_quat = self.object_rot
        current_obj_vel = self.object_linvel
        current_obj_ang_vel = self.object_angvel

        # --- Target states from demo data ---
        target_eef_pos = self.demo_data["wrist_pos"][env_arange, cur_idx]
        target_eef_quat = aa_to_quat(self.demo_data["wrist_rot"][env_arange, cur_idx])
        target_eef_vel = self.demo_data["wrist_velocity"][env_arange, cur_idx]
        target_eef_ang_vel = self.demo_data["wrist_angular_velocity"][env_arange, cur_idx]

        target_obj_transf = self.demo_data["obj_trajectory"][env_arange, cur_idx]
        target_obj_pos = target_obj_transf[:, :3, 3]
        target_obj_quat = rotmat_to_quat(target_obj_transf[:, :3, :3])
        target_obj_vel = self.demo_data["obj_velocity"][env_arange, cur_idx]
        target_obj_ang_vel = self.demo_data["obj_angular_velocity"][env_arange, cur_idx]

        # --- End effector rewards ---
        diff_eef_pos_dist = torch.norm(target_eef_pos - current_eef_pos, dim=-1)
        reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)

        diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
        # w is at index 0 in IsaacLab's wxyz quaternion convention
        diff_eef_rot_angle = 2.0 * torch.acos(torch.clamp(torch.abs(diff_eef_rot[:, 0]), max=1.0))
        reward_eef_rot = torch.exp(-1 * diff_eef_rot_angle.abs())

        diff_eef_vel = target_eef_vel - current_eef_vel
        reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))

        diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel
        reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))

        # --- Joint position rewards (body positions in demo order, excluding wrist) ---
        joints_pos = self.hand_body_pos  # (nE, n_bodies-1, 3) in demo order
        target_joints_pos = self.demo_data["mano_joints"][env_arange, cur_idx].reshape(
            self.num_envs, -1, 3
        )
        diff_joints_pos = target_joints_pos - joints_pos
        diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)  # (nE, n_bodies-1)

        # Per-group joint distances using weight_idx (indices are in body_names order, -1 for wrist skip)
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
        joints_vel = self.hand_body_vel  # (nE, n_bodies-1, 3) in demo order
        target_joints_vel = self.demo_data["mano_joints_velocity"][env_arange, cur_idx].reshape(
            self.num_envs, -1, 3
        )
        diff_joints_vel = target_joints_vel - joints_vel
        reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

        # --- Object rewards ---
        diff_obj_pos_dist = torch.norm(target_obj_pos - current_obj_pos, dim=-1)
        reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

        diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
        diff_obj_rot_angle = 2.0 * torch.acos(torch.clamp(torch.abs(diff_obj_rot[:, 0]), max=1.0))
        reward_obj_rot = torch.exp(-3 * diff_obj_rot_angle.abs())

        diff_obj_vel = target_obj_vel - current_obj_vel
        reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))

        diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
        reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))

        # --- Power penalties ---
        # Joint power: |applied_torque * joint_vel|
        applied_torque = self.hand.data.applied_torque
        joint_vel_isaaclab = self.hand.data.joint_vel
        power = torch.abs(applied_torque * joint_vel_isaaclab).sum(dim=-1)
        reward_power = torch.exp(-10 * power)

        # Wrist power: |force * lin_vel| + |torque * ang_vel|
        wrist_force = self.apply_forces[:, self.wrist_body_idx, :]
        wrist_torque = self.apply_torques[:, self.wrist_body_idx, :]
        wrist_power = torch.abs(torch.sum(wrist_force * current_eef_vel, dim=-1))
        wrist_power += torch.abs(torch.sum(wrist_torque * current_eef_ang_vel, dim=-1))
        reward_wrist_power = torch.exp(-2 * wrist_power)

        # --- Fingertip contact force reward ---
        finger_tip_distance = self.demo_data["tips_distance"][env_arange, cur_idx]  # (nE, 5)
        # Get contact forces on contact bodies
        tip_force = self.net_contact_forces[:, self.contact_body_indices, :]  # (nE, 5, 3)

        # Update contact history
        self.tips_contact_history = torch.cat([
            self.tips_contact_history[:, 1:],
            (torch.norm(tip_force, dim=-1) > 0)[:, None],
        ], dim=1)

        # Weighted contact force: only encourage contact when demo shows fingertip close to object
        contact_range_lo, contact_range_hi = 0.02, 0.03
        finger_tip_weight = torch.clamp(
            (contact_range_hi - finger_tip_distance) / (contact_range_hi - contact_range_lo), 0, 1
        )
        finger_tip_force_masked = tip_force * finger_tip_weight[:, :, None]
        reward_finger_tip_force = torch.exp(
            -1 * (1 / (torch.norm(finger_tip_force_masked, dim=-1).sum(-1) + 1e-5))
        )

        # --- Store intermediate values for termination check ---
        self._reward_cache = {
            "diff_obj_pos_dist": diff_obj_pos_dist,
            "diff_obj_rot_angle": diff_obj_rot_angle,
            "diff_thumb_tip_pos_dist": diff_thumb_tip_pos_dist,
            "diff_index_tip_pos_dist": diff_index_tip_pos_dist,
            "diff_middle_tip_pos_dist": diff_middle_tip_pos_dist,
            "diff_ring_tip_pos_dist": diff_ring_tip_pos_dist,
            "diff_pinky_tip_pos_dist": diff_pinky_tip_pos_dist,
            "diff_level_1_pos_dist": diff_level_1_pos_dist,
            "diff_level_2_pos_dist": diff_level_2_pos_dist,
            "finger_tip_distance": finger_tip_distance,
            "current_eef_vel": current_eef_vel,
            "current_eef_ang_vel": current_eef_ang_vel,
            "joints_vel": joints_vel,
            "current_obj_vel": current_obj_vel,
            "current_obj_ang_vel": current_obj_ang_vel,
        }

        # --- Total reward (matching original ManipTrans weights) ---
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
            + 5.0  * reward_obj_pos
            + 1.0  * reward_obj_rot
            + 0.1  * reward_eef_vel
            + 0.05 * reward_eef_ang_vel
            + 0.1  * reward_joints_vel
            + 0.1  * reward_obj_vel
            + 0.1  * reward_obj_ang_vel
            + 1.0  * reward_finger_tip_force
            + 0.5  * reward_power
            + 0.5  * reward_wrist_power
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
        self.extras["log"]["reward_joints_pos"] = (
            reward_thumb_tip_pos + reward_index_tip_pos + reward_middle_tip_pos
            + reward_pinky_tip_pos + reward_ring_tip_pos
            + reward_level_1_pos + reward_level_2_pos
        ).mean()
        self.extras["log"]["reward_power"] = reward_power.mean()
        self.extras["log"]["reward_wrist_power"] = reward_wrist_power.mean()
        self.extras["log"]["reward_finger_tip_force"] = reward_finger_tip_force.mean()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions (matching original ManipTrans)."""
        self._compute_intermediate_values()
        self.global_step_count += 1
        self.running_progress_buf += 1

        max_length = torch.clip(self.demo_data["seq_len"], 0, self.max_episode_length)
        if self.rollout_len is not None:
            max_length = torch.clamp(max_length, 0, self.rollout_len + self.rollout_begin + 3 + 1)

        # Compute curriculum scale factor
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

        # Use cached values from _get_rewards (called before _get_dones in DirectRLEnv)
        rc = self._reward_cache if hasattr(self, '_reward_cache') else {}
        diff_obj_pos_dist = rc.get("diff_obj_pos_dist", torch.norm(
            self.demo_data["obj_trajectory"][torch.arange(self.num_envs), self.episode_length_buf, :3, 3]
            - self.object_pos, dim=-1
        ))
        diff_obj_rot_angle = rc.get("diff_obj_rot_angle", torch.zeros(self.num_envs, device=self.device))
        diff_thumb_tip_pos_dist = rc.get("diff_thumb_tip_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_index_tip_pos_dist = rc.get("diff_index_tip_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_middle_tip_pos_dist = rc.get("diff_middle_tip_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_ring_tip_pos_dist = rc.get("diff_ring_tip_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_pinky_tip_pos_dist = rc.get("diff_pinky_tip_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_level_1_pos_dist = rc.get("diff_level_1_pos_dist", torch.zeros(self.num_envs, device=self.device))
        diff_level_2_pos_dist = rc.get("diff_level_2_pos_dist", torch.zeros(self.num_envs, device=self.device))
        finger_tip_distance = rc.get("finger_tip_distance", torch.ones(self.num_envs, 5, device=self.device))
        current_eef_vel = rc.get("current_eef_vel", self.hand_root_state[:, 7:10])
        current_eef_ang_vel = rc.get("current_eef_ang_vel", self.hand_root_state[:, 10:13])
        joints_vel = rc.get("joints_vel", self.hand_body_vel)
        current_obj_vel = rc.get("current_obj_vel", self.object_linvel)
        current_obj_ang_vel = rc.get("current_obj_ang_vel", self.object_angvel)

        # Sanity check: physics instability
        current_dof_vel = self.hand.data.joint_vel
        error_buf = (
            (torch.norm(current_eef_vel, dim=-1) > 100)
            | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
            | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
            | (torch.abs(current_dof_vel).mean(-1) > 200)
            | (torch.norm(current_obj_vel, dim=-1) > 100)
            | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
        )
        self.error_buf = error_buf

        s = scale_factor
        # Detailed failure conditions (matching original ManipTrans)
        failed_execute = (
            (
                (diff_obj_pos_dist > 0.02 / 0.343 * s ** 3)
                | (diff_thumb_tip_pos_dist > 0.04 / 0.7 * s)
                | (diff_index_tip_pos_dist > 0.045 / 0.7 * s)
                | (diff_middle_tip_pos_dist > 0.05 / 0.7 * s)
                | (diff_pinky_tip_pos_dist > 0.06 / 0.7 * s)
                | (diff_ring_tip_pos_dist > 0.06 / 0.7 * s)
                | (diff_level_1_pos_dist > 0.07 / 0.7 * s)
                | (diff_level_2_pos_dist > 0.08 / 0.7 * s)
                | (diff_obj_rot_angle.abs() / np.pi * 180 > 30 / 0.343 * s ** 3)
                | torch.any(
                    (finger_tip_distance < 0.005) & ~self.tips_contact_history.any(1),
                    dim=-1,
                )
            )
            & (self.running_progress_buf >= 8)
        ) | error_buf

        # Success: reached end of trajectory
        succeeded = (
            self.episode_length_buf + 1 + 3 >= max_length
        ) & ~failed_execute

        terminated = failed_execute
        time_out = succeeded

        # Update success/failure buffers
        self.success_buf = succeeded
        self.failure_buf = failed_execute

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

        # Demo data DOF positions are in dexhand.dof_names order (depth-first)
        # We need to reorder them to match IsaacLab's joint order
        dof_pos_demo_order = self.demo_data["opt_dof_pos"][env_ids_tensor, seq_idx]
        dof_vel_demo_order = self.demo_data["opt_dof_velocity"][env_ids_tensor, seq_idx]

        # Reorder DOF values from demo order to IsaacLab order using scatter
        # demo_to_isaaclab_dof_mapping[demo_idx] = isaaclab_idx
        indices = self.demo_to_isaaclab_dof_mapping.unsqueeze(0).expand(dof_pos_demo_order.shape[0], -1)
        dof_pos = torch.zeros_like(dof_pos_demo_order)
        dof_vel = torch.zeros_like(dof_vel_demo_order)
        dof_pos.scatter_(1, indices, dof_pos_demo_order)
        dof_vel.scatter_(1, indices, dof_vel_demo_order)

        dof_pos = saturate(dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)

        # Clamp DOF velocities to speed limits (matching original ManipTrans)
        dof_vel = torch.clamp(
            dof_vel,
            -self.dexhand_dof_speed_limits.unsqueeze(0),
            self.dexhand_dof_speed_limits.unsqueeze(0),
        )

        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # Set initial position targets to match DOF state (prevents PD controller fighting)
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)

        # Reset wrist pose
        opt_wrist_pos = self.demo_data["opt_wrist_pos"][env_ids_tensor, seq_idx]
        # aa_to_quat returns wxyz format, IsaacLab also uses wxyz - no conversion needed
        # Note: The original IsaacGym code converts wxyz->xyzw because IsaacGym uses xyzw.
        # For IsaacLab, we keep wxyz format without any conversion or extra rotation.
        opt_wrist_rot = aa_to_quat(self.demo_data["opt_wrist_rot"][env_ids_tensor, seq_idx])

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
