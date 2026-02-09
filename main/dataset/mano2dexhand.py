"""
Mano to Dexhand Retargeting for IsaacLab.

This script performs retargeting from MANO hand data to dexterous robot hand,
optimizing joint angles to match target fingertip positions.

Migrated from IsaacGym to use pure PyTorch + pytorch_kinematics.
The retargeting optimization doesn't require physics simulation,
just forward kinematics.
"""

import argparse
import math
import os
import pickle
import logging

import numpy as np
import pytorch_kinematics as pk
import torch
from termcolor import cprint

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import (
    aa_to_quat,
    aa_to_rotmat,
    quat_to_rotmat,
    rot6d_to_aa,
    rot6d_to_quat,
    rot6d_to_rotmat,
    rotmat_to_aa,
    rotmat_to_quat,
    rotmat_to_rot6d,
)
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def pack_data(data, dexhand):
    """Pack list of data dicts into batched format."""
    packed_data = {}
    for k in data[0].keys():
        if k == "mano_joints":
            mano_joints = []
            for d in data:
                mano_joints.append(
                    torch.concat(
                        [
                            d[k][dexhand.to_hand(j_name)[0]]
                            for j_name in dexhand.body_names
                            if dexhand.to_hand(j_name)[0] != "wrist"
                        ],
                        dim=-1,
                    )
                )
            packed_data[k] = torch.stack(mano_joints).squeeze()
        elif isinstance(data[0][k], torch.Tensor):
            packed_data[k] = torch.stack([d[k] for d in data]).squeeze()
        elif isinstance(data[0][k], np.ndarray):
            packed_data[k] = np.stack([d[k] for d in data]).squeeze()
        else:
            packed_data[k] = [d[k] for d in data]
    return packed_data


def soft_clamp(x, lower, upper):
    """Soft clamping using sigmoid."""
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class Mano2Dexhand:
    """
    Pure PyTorch implementation of Mano to Dexhand retargeting.

    This version uses only pytorch_kinematics for forward kinematics
    without requiring any simulation environment.
    """

    def __init__(self, args, dexhand, obj_urdf_path=None):
        self.dexhand = dexhand
        self.num_envs = args.num_envs
        self.device = getattr(args, 'device', 'cuda:0')
        self.headless = getattr(args, 'headless', True)

        # Build kinematic chain from URDF
        urdf_path = dexhand.urdf_path
        with open(urdf_path, 'r') as f:
            urdf_content = f.read()
        self.chain = pk.build_chain_from_urdf(urdf_content)
        self.chain = self.chain.to(dtype=torch.float32, device=self.device)

        # Get joint names from chain
        self.joint_names = self.chain.get_joint_parameter_names()

        # Setup joint limits (will be updated if URDF has limits)
        n_dofs = len(self.joint_names)
        self.dexhand_dof_lower_limits = torch.full(
            (n_dofs,), -np.pi, device=self.device, dtype=torch.float32
        )
        self.dexhand_dof_upper_limits = torch.full(
            (n_dofs,), np.pi, device=self.device, dtype=torch.float32
        )

        # Default DOF positions
        default_dof_pos = np.ones(dexhand.n_dofs) * np.pi / 50
        if "inspire" in dexhand.name:
            if dexhand.n_dofs > 9:
                default_dof_pos[8] = 0.8
                default_dof_pos[9] = 0.05
        self.dexhand_default_dof_pos = torch.tensor(
            default_dof_pos, device=self.device, dtype=torch.float32
        )

        # Setup coordinate transform (MuJoCo to Gym frame)
        table_width_offset = 0.2
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        table_pos_z = 0.4
        table_half_height = 0.015
        self._table_surface_z = table_pos_z + table_half_height
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(
            mujoco2gym_transf, device=self.device, dtype=torch.float32
        )

        # Map from dexhand DOF ordering to chain ordering
        # Try to match joint names between dexhand and chain
        self.isaac2chain_order = []
        for joint_name in self.joint_names:
            # Find matching dexhand DOF
            found = False
            for i, dof_name in enumerate(dexhand.dof_names):
                if dof_name in joint_name or joint_name in dof_name:
                    self.isaac2chain_order.append(i)
                    found = True
                    break
            if not found:
                # Default: use index
                self.isaac2chain_order.append(len(self.isaac2chain_order))

        # If ordering doesn't match, use identity
        if len(self.isaac2chain_order) != dexhand.n_dofs:
            self.isaac2chain_order = list(range(min(len(self.joint_names), dexhand.n_dofs)))

    def fitting(self, max_iter, obj_trajectory, target_wrist_pos, target_wrist_rot, target_mano_joints):
        """
        Optimize dexhand pose to match MANO joint positions.

        Args:
            max_iter: Maximum optimization iterations
            obj_trajectory: Object trajectory [N, 4, 4]
            target_wrist_pos: Target wrist positions [N, 3]
            target_wrist_rot: Target wrist rotations (axis-angle) [N, 3]
            target_mano_joints: Target MANO joint positions [N, n_joints, 3]

        Returns:
            Dictionary with optimized parameters
        """
        assert target_mano_joints.shape[0] == self.num_envs

        # Transform targets to gym coordinate frame
        target_wrist_pos = (
            self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T
        ).T + self.mujoco2gym_transf[:3, 3]
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
        target_mano_joints = target_mano_joints.reshape(-1, 3)
        target_mano_joints = (
            self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T
        ).T + self.mujoco2gym_transf[:3, 3]
        target_mano_joints = target_mano_joints.reshape(self.num_envs, -1, 3)

        obj_trajectory = self.mujoco2gym_transf @ obj_trajectory

        # Compute initial wrist offset from object
        middle_pos = (target_mano_joints[:, 3] + target_wrist_pos) / 2
        obj_pos = obj_trajectory[:, :3, 3]
        offset = middle_pos - obj_pos
        offset = offset / (torch.norm(offset, dim=-1, keepdim=True) + 1e-8) * 0.2

        # Initialize optimization variables
        opt_wrist_pos = (target_wrist_pos + offset).clone().detach().requires_grad_(True)
        opt_wrist_rot = rotmat_to_rot6d(target_wrist_rot).clone().detach().requires_grad_(True)
        opt_dof_pos = self.dexhand_default_dof_pos[None].repeat(self.num_envs, 1).clone().detach().requires_grad_(True)

        # Setup optimizer
        optimizer = torch.optim.Adam([
            {"params": [opt_wrist_pos, opt_wrist_rot], "lr": 0.0008},
            {"params": [opt_dof_pos], "lr": 0.0004},
        ])

        # Joint weights for loss computation
        weight = []
        for k in self.dexhand.body_names:
            k = self.dexhand.to_hand(k)[0]
            if "tip" in k:
                if "index" in k:
                    weight.append(20)
                elif "middle" in k:
                    weight.append(10)
                elif "ring" in k:
                    weight.append(7)
                elif "pinky" in k:
                    weight.append(5)
                elif "thumb" in k:
                    weight.append(25)
                else:
                    weight.append(10)
            elif "proximal" in k:
                weight.append(1)
            elif "intermediate" in k:
                weight.append(1)
            else:
                weight.append(1)
        weight = torch.tensor(weight, device=self.device, dtype=torch.float32)

        # Optimization loop
        past_loss = 1e10
        for iter_idx in range(max_iter):
            # Clamp DOF positions to limits
            opt_dof_pos_clamped = torch.clamp(
                opt_dof_pos,
                self.dexhand_dof_lower_limits[:opt_dof_pos.shape[1]],
                self.dexhand_dof_upper_limits[:opt_dof_pos.shape[1]],
            )

            # Forward kinematics
            # Reorder DOFs for the chain
            chain_dof_pos = opt_dof_pos_clamped
            if len(self.isaac2chain_order) > 0:
                chain_dof_pos = opt_dof_pos_clamped[:, :len(self.isaac2chain_order)]

            ret = self.chain.forward_kinematics(chain_dof_pos)

            # Get joint positions from FK
            pk_joints = []
            for body_name in self.dexhand.body_names:
                if body_name in ret:
                    pk_joints.append(ret[body_name].get_matrix()[:, :3, 3])
                else:
                    # Try to find a matching name
                    found = False
                    for fk_name in ret.keys():
                        if body_name in fk_name or fk_name in body_name:
                            pk_joints.append(ret[fk_name].get_matrix()[:, :3, 3])
                            found = True
                            break
                    if not found:
                        # Use zeros as placeholder
                        pk_joints.append(torch.zeros(self.num_envs, 3, device=self.device))

            pk_joints = torch.stack(pk_joints, dim=1)

            # Transform to world frame
            opt_wrist_rotmat = rot6d_to_rotmat(opt_wrist_rot)
            pk_joints_world = (opt_wrist_rotmat @ pk_joints.transpose(-1, -2)).transpose(-1, -2) + opt_wrist_pos[:, None]

            # Compute loss
            target_joints = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)
            loss = torch.mean(torch.norm(pk_joints_world - target_joints, dim=-1) * weight[None])

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging and early stopping
            if (iter_idx + 1) % 100 == 0:
                cprint(f"Iter {iter_idx + 1}: loss = {loss.item():.6f}", "green")
                if iter_idx > 0 and past_loss - loss.item() < 1e-5:
                    cprint("Converged early!", "yellow")
                    break
                past_loss = loss.item()

        # Final clamped values
        opt_dof_pos_clamped = torch.clamp(
            opt_dof_pos,
            self.dexhand_dof_lower_limits[:opt_dof_pos.shape[1]],
            self.dexhand_dof_upper_limits[:opt_dof_pos.shape[1]],
        )

        # Get final joint positions
        chain_dof_pos = opt_dof_pos_clamped
        if len(self.isaac2chain_order) > 0:
            chain_dof_pos = opt_dof_pos_clamped[:, :len(self.isaac2chain_order)]

        ret = self.chain.forward_kinematics(chain_dof_pos)
        pk_joints = []
        for body_name in self.dexhand.body_names:
            if body_name in ret:
                pk_joints.append(ret[body_name].get_matrix()[:, :3, 3])
            else:
                for fk_name in ret.keys():
                    if body_name in fk_name or fk_name in body_name:
                        pk_joints.append(ret[fk_name].get_matrix()[:, :3, 3])
                        break
                else:
                    pk_joints.append(torch.zeros(self.num_envs, 3, device=self.device))

        pk_joints = torch.stack(pk_joints, dim=1)
        opt_wrist_rotmat = rot6d_to_rotmat(opt_wrist_rot)
        final_joints = (opt_wrist_rotmat @ pk_joints.transpose(-1, -2)).transpose(-1, -2) + opt_wrist_pos[:, None]

        # Prepare output
        to_dump = {
            "opt_wrist_pos": opt_wrist_pos.detach().cpu().numpy(),
            "opt_wrist_rot": rot6d_to_aa(opt_wrist_rot).detach().cpu().numpy(),
            "opt_dof_pos": opt_dof_pos_clamped.detach().cpu().numpy(),
            "opt_joints_pos": final_joints.detach().cpu().numpy(),
        }

        return to_dump


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mano to Dexhand Retargeting")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without visualization")
    parser.add_argument("--iter", type=int, default=4000, help="Maximum optimization iterations")
    parser.add_argument("--data_idx", type=str, default="g0", help="Data index to process")
    parser.add_argument("--dexhand", type=str, default="inspire", help="Dexhand type")
    parser.add_argument("--side", type=str, default="right", choices=["left", "right"], help="Hand side")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create dexhand instance
    dexhand = DexHandFactory.create_hand(args.dexhand, args.side)

    def run(args, idx):
        dataset_type = ManipDataFactory.dataset_type(idx)
        demo_d = ManipDataFactory.create_data(
            manipdata_type=dataset_type,
            side=args.side,
            device=args.device,
            mujoco2gym_transf=torch.eye(4, device=args.device),
            dexhand=dexhand,
            verbose=False,
        )

        demo_data = pack_data([demo_d[idx]], dexhand)

        args.num_envs = demo_data["mano_joints"].shape[0]

        # Create retargeting instance
        obj_urdf_path = demo_data.get("obj_urdf_path", [None])[0]
        mano2dexhand = Mano2Dexhand(args, dexhand, obj_urdf_path)

        # Run optimization
        to_dump = mano2dexhand.fitting(
            args.iter,
            demo_data["obj_trajectory"],
            demo_data["wrist_pos"],
            demo_data["wrist_rot"],
            demo_data["mano_joints"].view(args.num_envs, -1, 3),
        )

        # Determine output path based on dataset type
        if dataset_type == "oakink2":
            dump_path = f"data/retargeting/OakInk-v2/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.pkl', f'@{idx[-1]}.pkl')}"
        elif dataset_type == "favor":
            dump_path = f"data/retargeting/favor_pass1/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1]}"
        elif dataset_type == "grabdemo":
            dump_path = f"data/retargeting/grab_demo/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.npy', '.pkl')}"
        elif dataset_type == "oakink2_mirrored":
            dump_path = f"data/retargeting/OakInk-v2-mirrored/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.pkl', f'@{idx[-1]}.pkl')}"
        elif dataset_type == "favor_mirrored":
            dump_path = f"data/retargeting/favor_pass1-mirrored/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1]}"
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "wb") as f:
            pickle.dump(to_dump, f)

        cprint(f"Saved retargeting results to: {dump_path}", "green")

    run(args, args.data_idx)


if __name__ == "__main__":
    main()
