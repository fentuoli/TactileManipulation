"""Feature extraction modules ported from ManipTrans.

Contains SimpleFeatureFusion (dict obs -> fused feature), Identity extractor,
and build_mlp utility. These match the original ManipTrans lib/nn/features/
and lib/nn/mlp.py exactly.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn


def get_activation(activation: str | Callable | None) -> Callable:
    if not activation:
        return nn.Identity
    elif callable(activation):
        return activation
    ACT_LAYER = {
        "tanh": nn.Tanh,
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
        "swish": lambda: nn.SiLU(inplace=True),
        "sigmoid": nn.Sigmoid,
        "elu": lambda: nn.ELU(inplace=True),
        "gelu": nn.GELU,
    }
    activation = activation.lower()
    assert activation in ACT_LAYER, f"Supported activations: {ACT_LAYER.keys()}"
    return ACT_LAYER[activation]


def get_initializer(method: str | Callable, activation: str) -> Callable:
    if isinstance(method, str):
        assert hasattr(
            nn.init, f"{method}_"
        ), f"Initializer nn.init.{method}_ does not exist"
        if method == "orthogonal":
            try:
                gain = nn.init.calculate_gain(activation)
            except ValueError:
                gain = 1.0
            return lambda x: nn.init.orthogonal_(x, gain=gain)
        else:
            return getattr(nn.init, f"{method}_")
    else:
        assert callable(method)
        return method


def build_mlp(
    input_dim,
    *,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: int = None,
    num_layers: int = None,
    activation: str | Callable = "relu",
    weight_init: str | Callable = "orthogonal",
    bias_init="zeros",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    add_input_activation: bool | str | Callable = False,
    add_input_norm: bool = False,
    add_output_activation: bool | str | Callable = False,
    add_output_norm: bool = False,
) -> nn.Sequential:
    assert (hidden_depth is None) != (num_layers is None), (
        "Either hidden_depth or num_layers must be specified, but not both. "
        "num_layers is defined as hidden_depth+1"
    )
    if hidden_depth is not None:
        assert hidden_depth >= 0
    if num_layers is not None:
        assert num_layers >= 1
    act_layer = get_activation(activation)

    weight_init = get_initializer(weight_init, activation)
    bias_init = get_initializer(bias_init, activation)

    if norm_type is not None:
        norm_type = norm_type.lower()

    if not norm_type:
        norm_type = nn.Identity
    elif norm_type == "batchnorm":
        norm_type = nn.BatchNorm1d
    elif norm_type == "layernorm":
        norm_type = nn.LayerNorm
    else:
        raise ValueError(f"Unsupported norm layer: {norm_type}")

    hidden_depth = num_layers - 1 if hidden_depth is None else hidden_depth
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), norm_type(hidden_dim), act_layer()]
        for i in range(hidden_depth - 1):
            mods += [
                nn.Linear(hidden_dim, hidden_dim),
                norm_type(hidden_dim),
                act_layer(),
            ]
        mods.append(nn.Linear(hidden_dim, output_dim))

    if add_input_norm:
        mods = [norm_type(input_dim)] + mods
    if add_input_activation:
        if add_input_activation is not True:
            act_layer = get_activation(add_input_activation)
        mods = [act_layer()] + mods
    if add_output_norm:
        mods.append(norm_type(output_dim))
    if add_output_activation:
        if add_output_activation is not True:
            act_layer = get_activation(add_output_activation)
        mods.append(act_layer())

    for mod in mods:
        if isinstance(mod, nn.Linear):
            weight_init(mod.weight)
            bias_init(mod.bias)

    return nn.Sequential(*mods)


class Identity(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self._output_dim = input_dim

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x):
        return x


class SimpleFeatureFusion(nn.Module):
    """Fuses dict observations by concatenating extractors in sorted key order,
    then passing through a head MLP.

    Original: lib/nn/features/fusion.py
    """

    def __init__(
        self,
        extractors: dict[str, nn.Module],
        hidden_depth: int,
        hidden_dim: int,
        output_dim: int,
        activation,
        add_input_activation: bool,
        add_output_activation: bool,
    ):
        super().__init__()
        self._extractors = nn.ModuleDict(extractors)
        extractors_output_dim = sum(e.output_dim for e in extractors.values())
        self.output_dim = output_dim
        self._head = build_mlp(
            input_dim=extractors_output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            weight_init="orthogonal",
            bias_init="zeros",
            norm_type=None,
            add_input_activation=add_input_activation,
            add_input_norm=False,
            add_output_activation=add_output_activation,
            add_output_norm=False,
        )

        self._obs_groups = None
        self._obs_key_checked = False

    def _check_obs_key_match(self, obs: dict, strict: bool = False):
        if strict:
            assert set(self._extractors.keys()) == set(obs.keys())
        elif set(self._extractors.keys()) != set(obs.keys()):
            print(f"[warning] obs key mismatch: {set(self._extractors.keys())} != {set(obs.keys())}")

    def forward(self, x):
        x = self._group_obs(x)
        if not self._obs_key_checked:
            self._check_obs_key_match(x, strict=False)
            self._obs_key_checked = True
        x = {k: v.forward(x[k]) for k, v in self._extractors.items()}
        # Concatenate in sorted key order (critical: privileged, proprioception, target)
        x = torch.cat([x[k] for k in sorted(x.keys())], dim=-1)
        x = self._head(x)
        return x

    def _group_obs(self, obs):
        obs_keys = obs.keys()
        if self._obs_groups is None:
            obs_groups = {k.split("/")[0] for k in obs_keys}
            self._obs_groups = sorted(list(obs_groups))
        obs_rtn = {}
        for g in self._obs_groups:
            is_subgroup = any(k.startswith(f"{g}/") for k in obs_keys)
            if is_subgroup:
                obs_rtn[g] = {k.split("/", 1)[1]: v for k, v in obs.items() if k.startswith(f"{g}/")}
            else:
                obs_rtn[g] = obs[g]
        return obs_rtn
