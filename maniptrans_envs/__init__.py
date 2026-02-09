# ManipTrans environments for IsaacLab
# Copyright (c) 2024 ManipTrans Authors

# Lazy imports to avoid requiring Isaac Sim for preprocessing scripts
# that only need pytorch_kinematics
def __getattr__(name):
    if name in ("DexHandManipEnv", "DexHandManipEnvCfg",
                "DexHandManipRHEnvCfg", "DexHandManipLHEnvCfg"):
        from .tasks.dexhand_manip_env import (
            DexHandManipEnv, DexHandManipEnvCfg,
            DexHandManipRHEnvCfg, DexHandManipLHEnvCfg
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DexHandManipEnv",
    "DexHandManipEnvCfg",
    "DexHandManipRHEnvCfg",
    "DexHandManipLHEnvCfg",
]
