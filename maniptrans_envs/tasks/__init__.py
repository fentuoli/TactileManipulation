# ManipTrans task environments
# Lazy imports to avoid requiring Isaac Sim for preprocessing scripts

from . import agents

def __getattr__(name):
    if name in ("DexHandManipEnv", "DexHandManipEnvCfg",
                "DexHandManipRHEnvCfg", "DexHandManipLHEnvCfg"):
        from .dexhand_manip_env import (
            DexHandManipEnv, DexHandManipEnvCfg,
            DexHandManipRHEnvCfg, DexHandManipLHEnvCfg
        )
        return locals()[name]
    if name in ("DexHandImitatorEnv", "DexHandImitatorEnvCfg",
                "DexHandImitatorRHEnvCfg", "DexHandImitatorLHEnvCfg"):
        from .dexhand_imitator_env import (
            DexHandImitatorEnv, DexHandImitatorEnvCfg,
            DexHandImitatorRHEnvCfg, DexHandImitatorLHEnvCfg
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DexHandManipEnv",
    "DexHandManipEnvCfg",
    "DexHandManipRHEnvCfg",
    "DexHandManipLHEnvCfg",
    "DexHandImitatorEnv",
    "DexHandImitatorEnvCfg",
    "DexHandImitatorRHEnvCfg",
    "DexHandImitatorLHEnvCfg",
]


def register_maniptrans_envs():
    """Register ManipTrans environments with gymnasium.

    Call this function after Isaac Sim is initialized.
    """
    import gymnasium as gym

    # --- Residual (manipulation) environments ---
    # Right hand environment
    gym.register(
        id="ManipTrans-DexHand-RH-Direct-v0",
        entry_point="maniptrans_envs.tasks.dexhand_manip_env:DexHandManipEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "maniptrans_envs.tasks.dexhand_manip_env:DexHandManipRHEnvCfg",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        },
    )

    # Left hand environment
    gym.register(
        id="ManipTrans-DexHand-LH-Direct-v0",
        entry_point="maniptrans_envs.tasks.dexhand_manip_env:DexHandManipEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "maniptrans_envs.tasks.dexhand_manip_env:DexHandManipLHEnvCfg",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        },
    )

    # --- Imitator (hand-only tracking) environments ---
    # Right hand imitator
    gym.register(
        id="ManipTrans-DexHand-Imitator-RH-Direct-v0",
        entry_point="maniptrans_envs.tasks.dexhand_imitator_env:DexHandImitatorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "maniptrans_envs.tasks.dexhand_imitator_env:DexHandImitatorRHEnvCfg",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_imitator_ppo_cfg.yaml",
        },
    )

    # Left hand imitator
    gym.register(
        id="ManipTrans-DexHand-Imitator-LH-Direct-v0",
        entry_point="maniptrans_envs.tasks.dexhand_imitator_env:DexHandImitatorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "maniptrans_envs.tasks.dexhand_imitator_env:DexHandImitatorLHEnvCfg",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_imitator_ppo_cfg.yaml",
        },
    )
