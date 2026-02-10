"""
Training script for ManipTrans with IsaacLab.

This script trains RL policies for dexterous hand manipulation using IsaacLab
and the rl_games library.

Usage (similar to original ManipTrans):
    python main/rl/train.py --dexhand inspire --side right --num_envs 4096 \
        --learning_rate 2e-4 --random_state_init --data_indices g0 \
        --rh_base_checkpoint assets/imitator_rh_inspire.pth \
        --actions_moving_average 0.4 --experiment cross_g0_inspire --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train ManipTrans RL agent with RL-Games.")

# Environment configuration
parser.add_argument("--mode", type=str, default="residual", choices=["imitator", "residual"],
                    help="Training mode: imitator (hand-only tracking) or residual (hand+object manipulation)")
parser.add_argument("--dexhand", type=str, default="inspire", help="Dexhand type (inspire, shadow, allegro, etc.)")
parser.add_argument("--side", type=str, default="right", choices=["left", "right", "RH", "LH"], help="Hand side")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--data_indices", type=str, nargs="+", default=["g0"], help="Data indices for demonstration (e.g., g0 or 20aed@0)")

# Training configuration
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--max_epochs", type=int, default=5000, help="Maximum training epochs")
parser.add_argument("--early_stop_epochs", type=int, default=100, help="Early stop after N epochs without improvement")
parser.add_argument("--actions_moving_average", type=float, default=0.4, help="Actions moving average factor")
parser.add_argument("--random_state_init", action="store_true", default=False, help="Use random state initialization")
parser.add_argument("--use_pid_control", action="store_true", default=False, help="Use PID control mode")

# Imitator checkpoints (for residual learning)
parser.add_argument("--rh_base_checkpoint", type=str, default=None, help="Right hand imitator checkpoint")
parser.add_argument("--lh_base_checkpoint", type=str, default=None, help="Left hand imitator checkpoint")

# Testing/evaluation
parser.add_argument("--test", action="store_true", default=False, help="Test mode instead of training")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint for testing/resuming")

# Experiment naming
parser.add_argument("--experiment", type=str, default=None, help="Experiment name")

# Other settings
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument("--track", action="store_true", default=False, help="Track with Weights and Biases")
parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Normalize side argument
if args_cli.side in ["right", "RH"]:
    args_cli.side = "right"
else:
    args_cli.side = "left"

# Always enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
import time
import yaml
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

# Register ManipTrans environments
from maniptrans_envs.tasks import register_maniptrans_envs
register_maniptrans_envs()


def load_agent_cfg(args) -> dict:
    """Load RL-Games agent configuration."""
    # Select config file based on training mode
    if args.mode == "imitator":
        config_filename = "rl_games_imitator_ppo_cfg.yaml"
    else:
        config_filename = "rl_games_ppo_cfg.yaml"

    config_path = os.path.join(
        os.path.dirname(__file__),
        f"../../maniptrans_envs/tasks/agents/{config_filename}"
    )

    with open(config_path, "r") as f:
        agent_cfg = yaml.safe_load(f)

    # Override with command line arguments
    agent_cfg["params"]["config"]["learning_rate"] = args.learning_rate
    agent_cfg["params"]["config"]["max_epochs"] = args.max_epochs

    # Adjust minibatch size based on num_envs
    horizon_length = agent_cfg["params"]["config"]["horizon_length"]
    batch_size = args.num_envs * horizon_length
    # Ensure minibatch divides batch evenly
    minibatch_size = min(batch_size, 16384)
    while batch_size % minibatch_size != 0 and minibatch_size > 64:
        minibatch_size //= 2
    agent_cfg["params"]["config"]["minibatch_size"] = minibatch_size

    return agent_cfg


def main():
    """Train with RL-Games agent."""
    # Select environment config and task based on mode
    if args_cli.mode == "imitator":
        from maniptrans_envs.tasks.dexhand_imitator_env import (
            DexHandImitatorEnvCfg, DexHandImitatorRHEnvCfg, DexHandImitatorLHEnvCfg
        )
        if args_cli.side == "right":
            env_cfg = DexHandImitatorRHEnvCfg()
            task_name = "ManipTrans-DexHand-Imitator-RH-Direct-v0"
        else:
            env_cfg = DexHandImitatorLHEnvCfg()
            task_name = "ManipTrans-DexHand-Imitator-LH-Direct-v0"
    else:
        from maniptrans_envs.tasks.dexhand_manip_env import (
            DexHandManipEnvCfg, DexHandManipRHEnvCfg, DexHandManipLHEnvCfg
        )
        if args_cli.side == "right":
            env_cfg = DexHandManipRHEnvCfg()
            task_name = "ManipTrans-DexHand-RH-Direct-v0"
        else:
            env_cfg = DexHandManipLHEnvCfg()
            task_name = "ManipTrans-DexHand-LH-Direct-v0"

    # Override configurations from command line
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.dexhand_type = args_cli.dexhand
    env_cfg.side = args_cli.side
    env_cfg.data_indices = args_cli.data_indices
    env_cfg.act_moving_average = args_cli.actions_moving_average
    env_cfg.random_state_init = args_cli.random_state_init
    env_cfg.use_pid_control = args_cli.use_pid_control
    env_cfg.training = not args_cli.test

    # Pass imitator checkpoints to residual env (ignored for imitator mode)
    if args_cli.mode == "residual":
        if args_cli.rh_base_checkpoint:
            env_cfg.rh_base_checkpoint = args_cli.rh_base_checkpoint
        if args_cli.lh_base_checkpoint:
            env_cfg.lh_base_checkpoint = args_cli.lh_base_checkpoint

    # Load agent config
    agent_cfg = load_agent_cfg(args_cli)

    # Randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed

    if args_cli.checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args_cli.checkpoint
        print(f"[INFO]: Loading model checkpoint from: {args_cli.checkpoint}")

    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # Multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # Set the environment seed
    env_cfg.seed = agent_cfg["params"]["seed"]

    # Determine experiment name
    if args_cli.experiment:
        experiment_name = args_cli.experiment
    else:
        data_str = "_".join(args_cli.data_indices)
        experiment_name = f"cross_{data_str}_{args_cli.dexhand}"

    # Specify directory for logging experiments
    config_name = experiment_name
    agent_cfg["params"]["config"]["name"] = config_name

    if args_cli.test:
        log_root_path = os.path.abspath(os.path.join("dumps", config_name))
    else:
        log_root_path = os.path.abspath(os.path.join("runs", config_name))

    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # Specify directory for logging runs
    log_dir = datetime.now().strftime("%m-%d-%H-%M-%S")
    full_experiment_name = f"{config_name}__{log_dir}"
    agent_cfg["params"]["config"]["train_dir"] = os.path.dirname(log_root_path)
    agent_cfg["params"]["config"]["full_experiment_name"] = full_experiment_name

    # Create experiment directory
    experiment_dir = os.path.join(os.path.dirname(log_root_path), full_experiment_name)
    os.makedirs(os.path.join(experiment_dir, "params"), exist_ok=True)

    # Dump the configuration into log-directory
    dump_yaml(os.path.join(experiment_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(experiment_dir, "params", "agent.yaml"), agent_cfg)

    # Print configuration summary
    print(f"[INFO] Configuration:")
    print(f"  Mode: {args_cli.mode}")
    print(f"  Dexhand: {args_cli.dexhand}")
    print(f"  Side: {args_cli.side}")
    print(f"  Task: {task_name}")
    print(f"  Num envs: {args_cli.num_envs}")
    print(f"  Data indices: {args_cli.data_indices}")
    print(f"  Learning rate: {args_cli.learning_rate}")
    print(f"  Actions moving average: {args_cli.actions_moving_average}")
    print(f"  Random state init: {args_cli.random_state_init}")
    print(f"  Test mode: {args_cli.test}")
    if args_cli.mode == "residual":
        print(f"  RH imitator checkpoint: {args_cli.rh_base_checkpoint or 'None'}")
        print(f"  LH imitator checkpoint: {args_cli.lh_base_checkpoint or 'None'}")

    # Read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # Create isaac environment
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(experiment_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # Wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # Register the environment to rl-games registry
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # Set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # Create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # Reset the agent and env
    runner.reset()

    # Track with WandB
    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        wandb_project = config_name if args_cli.wandb_project is None else args_cli.wandb_project

        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=full_experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        if not wandb.run.resumed:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg})

    # Run training or testing
    run_args = {
        "train": not args_cli.test,
        "play": args_cli.test,
        "sigma": train_sigma,
    }
    if args_cli.checkpoint is not None:
        run_args["checkpoint"] = args_cli.checkpoint

    runner.run(run_args)

    print(f"Total time: {round(time.time() - start_time, 2)} seconds")

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
