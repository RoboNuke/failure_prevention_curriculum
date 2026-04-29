"""Block-parallel SimBa networks for SAC.

Ports the block-parallel primitives from RoboNuke/Continuous_Force_RL/models/block_simba.py
and exposes a Gaussian actor + Q-critic suited to skrl SAC. The Bernoulli/hybrid-control
machinery in the upstream actor has been stripped — pure squashed Gaussian only.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


# -----------------------------
#  Squashed Gaussian utilities
# -----------------------------
def squash_log_prob_correction(u: torch.Tensor) -> torch.Tensor:
    # log(1 - tanh(u)^2) summed over last dim; numerically stable form
    return (2.0 * math.log(2.0) - 2.0 * u - 2.0 * F.softplus(-2.0 * u)).sum(dim=-1)


def safe_atanh(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.atanh(torch.clamp(a, -1.0 + eps, 1.0 - eps))


# -----------------------------
#  Block-parallel primitives
# -----------------------------
class BlockLinear(nn.Module):
    def __init__(self, num_blocks: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_blocks, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(num_blocks, out_features))
        for i in range(num_blocks):
            nn.init.kaiming_normal_(self.weight[i])
            nn.init.zeros_(self.bias[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_blocks, batch, in_features)
        return torch.einsum("nbi,noi->nbo", x, self.weight) + self.bias[:, None, :]


class BlockLayerNorm(nn.Module):
    def __init__(self, num_blocks: int, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_blocks, normalized_shape))
        self.bias = nn.Parameter(torch.zeros(num_blocks, normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) * torch.rsqrt(var + self.eps)
        return out * self.weight[:, None, :] + self.bias[:, None, :]


class BlockMLP(nn.Module):
    def __init__(self, num_blocks: int, in_dim: int, hidden_dim: int, out_dim: int, activation=None):
        super().__init__()
        self.fc1 = BlockLinear(num_blocks, in_dim, hidden_dim)
        self.fc2 = BlockLinear(num_blocks, hidden_dim, out_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(F.relu(self.fc1(x)))
        if self.activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.activation == "tanh":
            out = torch.tanh(out)
        return out


class BlockResidualBlock(nn.Module):
    def __init__(self, num_blocks: int, dim: int):
        super().__init__()
        self.ln = BlockLayerNorm(num_blocks, dim)
        self.fc1 = BlockLinear(num_blocks, dim, 4 * dim)
        self.fc2 = BlockLinear(num_blocks, 4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(self.ln(x))))


# -----------------------------
#  BlockSimBa backbone
# -----------------------------
class BlockSimBa(nn.Module):
    """Block-parallel SimBa: input proj -> N residual blocks -> LN -> output proj."""

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        hidden_dim: int,
        act_dim: int,
        device,
        num_blocks: int = 2,
        use_state_dependent_std: bool = False,
        predict_success: bool = False,
    ):
        super().__init__()
        self.device = device
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.num_blocks = num_blocks
        self.use_state_dependent_std = use_state_dependent_std
        self.std_out_dim = act_dim if use_state_dependent_std else 0
        self.predict_success = predict_success
        self.success_out_dim = 1 if predict_success else 0

        # Output layout per row (along last dim):
        #   [0 : act_dim)                                  -> action mean
        #   [act_dim : act_dim + std_out_dim)              -> per-action log_std (state-dep std)
        #   [-success_out_dim:]                            -> success logit (single channel)
        total_out = act_dim + self.std_out_dim + self.success_out_dim
        self.fc_in = BlockLinear(num_agents, obs_dim, hidden_dim)
        self.resblocks = nn.ModuleList(
            [BlockResidualBlock(num_agents, hidden_dim) for _ in range(num_blocks)]
        )
        self.ln_out = BlockLayerNorm(num_agents, hidden_dim)
        self.fc_out = BlockLinear(num_agents, hidden_dim, total_out)

    def forward(self, obs_flat: torch.Tensor, num_envs: int):
        """Return ``(actions, log_std, success_logit)``.

        ``log_std`` is ``None`` unless ``use_state_dependent_std`` was set; ``success_logit``
        is ``None`` unless ``predict_success`` was set. Both shapes when present are
        ``(num_agents * num_envs, *)`` (last dim = ``std_out_dim`` for log_std, 1 for success).
        """
        obs = obs_flat.view(self.num_agents, num_envs, -1)
        x = self.fc_in(obs)
        for block in self.resblocks:
            x = block(x)
        out = self.fc_out(self.ln_out(x))

        actions = out[..., : self.act_dim]
        if self.std_out_dim > 0:
            log_std = out[..., self.act_dim : self.act_dim + self.std_out_dim].reshape(
                -1, self.std_out_dim
            )
        else:
            log_std = None

        if self.predict_success:
            success_logit = out[..., -self.success_out_dim :].reshape(-1, self.success_out_dim)
        else:
            success_logit = None

        return actions.reshape(-1, actions.shape[-1]), log_std, success_logit


# -----------------------------
#  Squashed-Gaussian actor
# -----------------------------
class BlockSimBaActor(GaussianMixin, Model):
    """SAC policy: squashed-Gaussian over action space, block-parallel across agents.

    Reads `inputs["observations"]` per skrl SAC convention. `act()` returns the
    tanh-squashed action and a log-prob with the SAC Jacobian correction applied.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        act_init_std: float = 0.60653066,
        actor_n: int = 2,
        actor_latent: int = 512,
        last_layer_scale: float = 1.0,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
        use_state_dependent_std: bool = False,
        predict_success: bool = True,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.num_agents = num_agents
        self.use_state_dependent_std = use_state_dependent_std
        self.predict_success = predict_success

        self.actor_mean = BlockSimBa(
            num_agents=num_agents,
            obs_dim=self.num_observations,
            hidden_dim=actor_latent,
            act_dim=self.num_actions,
            device=device,
            num_blocks=actor_n,
            use_state_dependent_std=use_state_dependent_std,
            predict_success=predict_success,
        ).to(device)

        if use_state_dependent_std:
            with torch.no_grad():
                self.actor_mean.fc_out.bias[:, self.num_actions:] = math.log(act_init_std)
                self.actor_mean.fc_out.weight[:, self.num_actions:, :] *= 0.1
            self.actor_logstd = None
        else:
            self.actor_logstd = nn.ParameterList(
                [
                    nn.Parameter(torch.ones(1, self.num_actions) * math.log(act_init_std))
                    for _ in range(num_agents)
                ]
            ).to(device)

        with torch.no_grad():
            self.actor_mean.fc_out.weight[:, : self.num_actions, :] *= last_layer_scale

    def compute(self, inputs, role):
        obs = inputs["observations"]
        num_envs = obs.size(0) // self.num_agents
        action_mean, log_std, success_logit = self.actor_mean(obs, num_envs)

        if not self.use_state_dependent_std:
            batch_size = action_mean.size(0) // self.num_agents
            log_std = torch.cat(
                [p.expand(batch_size, self.num_actions) for p in self.actor_logstd], dim=0
            )
        outputs = {"log_std": log_std}
        if success_logit is not None:
            outputs["success_logit"] = success_logit
        return action_mean, outputs

    def act(self, inputs, *, role: str = ""):
        # Squashed-Gaussian sampling with Jacobian correction; emits the (actions, outputs)
        # 2-tuple that skrl 2.x expects, with log_prob in outputs for SAC's update loop.
        mean_actions, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]

        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
            outputs["log_std"] = log_std

        self._g_distribution = Normal(mean_actions, log_std.exp())

        u = self._g_distribution.rsample()
        actions = torch.tanh(u)

        # On replay, evaluate log_prob at the taken actions; otherwise use our fresh sample.
        taken_actions = inputs.get("taken_actions", None)
        u_for_log_prob = u if taken_actions is None else safe_atanh(taken_actions)

        log_prob = (
            self._g_distribution.log_prob(u_for_log_prob).sum(dim=-1)
            - squash_log_prob_correction(u_for_log_prob)
        )
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_actions
        if "success_logit" in outputs:
            outputs["success_prob"] = torch.sigmoid(outputs["success_logit"])
        return actions, outputs

    def get_entropy(self, *, role: str = ""):
        # Use base Normal entropy as a proxy; the squashed Gaussian has no clean closed form
        # and SAC uses log_prob (not entropy) in the policy gradient.
        if self._g_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().to(self.device)


# -----------------------------
#  Q-critic (state, action -> scalar)
# -----------------------------
class BlockSimBaQCritic(DeterministicMixin, Model):
    """SAC Q-function: concatenates observation and action, returns scalar Q per (o, a).

    skrl SAC calls this via `critic.act({**inputs, "taken_actions": actions})`, where
    `inputs["observations"]` carries the observation and `inputs["taken_actions"]` the action.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        critic_output_init_mean: float = 0.0,
        critic_n: int = 2,
        critic_latent: int = 512,
        clip_actions: bool = False,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.num_agents = num_agents
        self.q_net = BlockSimBa(
            num_agents=num_agents,
            obs_dim=self.num_observations + self.num_actions,
            hidden_dim=critic_latent,
            act_dim=1,
            device=device,
            num_blocks=critic_n,
            use_state_dependent_std=False,
        ).to(device)

        torch.nn.init.constant_(self.q_net.fc_out.bias, critic_output_init_mean)

    def compute(self, inputs, role):
        obs = inputs["observations"]
        actions = inputs["taken_actions"]
        x = torch.cat([obs, actions], dim=-1)
        num_envs = x.size(0) // self.num_agents
        value, _, _ = self.q_net(x, num_envs)  # backbone returns (out, log_std, success_logit)
        return value, {}


# -----------------------------
#  Per-agent save/load slicing helpers
# -----------------------------
def _is_block_tensor(t, num_agents: int) -> bool:
    """A tensor is "block-parallel" if its leading dim equals num_agents."""
    return torch.is_tensor(t) and t.dim() >= 1 and t.shape[0] == num_agents


def _per_agent_paramlist_prefixes(block_module: nn.Module, num_agents: int) -> list[str]:
    """Find dotted prefixes of nn.ParameterList children that have ``num_agents`` entries.

    Used by the state_dict slicer to recognize per-agent ParameterList keys (e.g.
    ``actor_logstd.0``, ``actor_logstd.1`` for N=2) so they can be filtered + renumbered
    when slicing for a single agent.
    """
    prefixes = []
    for mod_name, mod in block_module.named_modules():
        if isinstance(mod, nn.ParameterList) and len(mod) == num_agents:
            prefixes.append(mod_name)  # e.g. "actor_logstd"
    return prefixes


def slice_block_state_dict(block_module: nn.Module, agent_idx: int, num_agents: int) -> dict:
    """Return a state_dict with every dim-0 == num_agents tensor sliced to agent_idx.

    Output tensors have one less leading dim than the source (e.g. ``(N, out, in)`` ->
    ``(out, in)``). Per-agent ``nn.ParameterList`` entries (length == num_agents) keep
    only the ``agent_idx``-th entry, renumbered to ``0`` so the result is shaped like a
    single-agent module's state_dict. Non-block tensors are passed through unchanged.
    """
    pl_prefixes = _per_agent_paramlist_prefixes(block_module, num_agents)
    sliced = {}
    for name, param in block_module.state_dict().items():
        # Handle per-agent ParameterList entries: keep only agent_idx, renumber to 0.
        matched_prefix = None
        for pre in pl_prefixes:
            if name.startswith(pre + "."):
                matched_prefix = pre
                break
        if matched_prefix is not None:
            tail = name[len(matched_prefix) + 1 :]
            head, _, rest = tail.partition(".")
            if not head.isdigit():
                raise ValueError(
                    f"Expected integer index after ParameterList prefix '{matched_prefix}.' "
                    f"in state_dict key '{name}', got '{head}'"
                )
            if int(head) != agent_idx:
                continue  # drop other agents' entries
            new_name = f"{matched_prefix}.0" + (("." + rest) if rest else "")
            sliced[new_name] = param.detach().clone().cpu() if torch.is_tensor(param) else param
            continue

        if _is_block_tensor(param, num_agents):
            sliced[name] = param[agent_idx].detach().clone().cpu()
        else:
            sliced[name] = param.detach().clone().cpu() if torch.is_tensor(param) else param
    return sliced


def assign_block_slice(
    block_module: nn.Module, agent_idx: int, num_agents: int, agent_state_dict: dict
) -> None:
    """Write ``agent_state_dict`` (single-agent shape) into block_module's slot ``agent_idx``.

    Inverts what :func:`slice_block_state_dict` did:
      * For each block-parallel param in block_module, copies the source tensor into
        ``param.data[agent_idx]``.
      * For per-agent ParameterList entries renamed ``prefix.0`` on save, writes them
        back to ``prefix.{agent_idx}`` of the destination.
      * Non-block params are copied wholesale.
    """
    pl_prefixes = _per_agent_paramlist_prefixes(block_module, num_agents)
    block_state = block_module.state_dict()

    # Translate "saved key" -> "destination block key" via the renumber-back step.
    remapped = {}
    for name, val in agent_state_dict.items():
        matched_prefix = None
        for pre in pl_prefixes:
            if name.startswith(pre + "."):
                matched_prefix = pre
                break
        if matched_prefix is not None:
            tail = name[len(matched_prefix) + 1 :]
            head, _, rest = tail.partition(".")
            if head == "0":
                new_name = f"{matched_prefix}.{agent_idx}" + (("." + rest) if rest else "")
                remapped[new_name] = val
                continue
        remapped[name] = val

    # Source must be a subset of dest's keys (other agents' ParameterList entries are
    # legitimately absent from a single-agent slice).
    extra = set(remapped.keys()) - set(block_state.keys())
    if extra:
        raise KeyError(f"Unexpected keys in single-agent state_dict: {sorted(extra)}")

    paramlist_keys = {k for k in block_state.keys()
                      if any(k.startswith(p + ".") for p in pl_prefixes)}

    with torch.no_grad():
        for name, agent_param in remapped.items():
            block_param = block_state[name]
            # ParameterList entries are written wholesale into their dedicated slot
            # (they don't have a leading num_agents dim of their own).
            if name in paramlist_keys:
                if torch.is_tensor(block_param):
                    block_param.copy_(agent_param.to(block_param.device))
                continue
            if _is_block_tensor(block_param, num_agents):
                block_param[agent_idx].copy_(agent_param.to(block_param.device))
            else:
                if torch.is_tensor(block_param):
                    block_param.copy_(agent_param.to(block_param.device))


def slice_optimizer_state(
    opt_state_dict: dict, agent_idx: int, num_agents: int
) -> dict:
    """Slice every dim-0 == num_agents tensor in an optimizer state_dict to agent_idx.

    Non-block tensors (e.g. Adam's scalar ``step``, or ``actor_logstd[i]`` whose
    own leading dim is 1) and non-tensor entries are passed through unchanged.
    ``param_groups`` is preserved verbatim. A sidecar ``_sliced_keys`` set records
    which (param_id, key) pairs were sliced so :func:`merge_optimizer_states` can
    unambiguously restack them later.
    """
    if "state" not in opt_state_dict:
        raise KeyError("Optimizer state_dict missing required key 'state'")
    if "param_groups" not in opt_state_dict:
        raise KeyError("Optimizer state_dict missing required key 'param_groups'")

    out_state = {}
    sliced_keys = set()
    for param_id, param_state in opt_state_dict["state"].items():
        new_state = {}
        for k, v in param_state.items():
            if _is_block_tensor(v, num_agents):
                new_state[k] = v[agent_idx].detach().clone().cpu()
                sliced_keys.add((param_id, k))
            elif torch.is_tensor(v):
                new_state[k] = v.detach().clone().cpu()
            else:
                new_state[k] = v
        out_state[param_id] = new_state
    return {
        "state": out_state,
        "param_groups": opt_state_dict["param_groups"],
        "_sliced_keys": sliced_keys,
    }


def merge_optimizer_states(per_agent_state_dicts: list, num_agents: int) -> dict:
    """Stack per-agent optimizer state_dicts back into a block-shaped state_dict.

    Uses each per-agent dict's ``_sliced_keys`` sidecar (written by
    :func:`slice_optimizer_state`) to know exactly which (param_id, key) pairs were
    sliced on save and therefore must be re-stacked. All other tensor entries are
    taken from agent 0 verbatim.
    """
    if len(per_agent_state_dicts) != num_agents:
        raise ValueError(
            f"Expected {num_agents} per-agent state_dicts, got {len(per_agent_state_dicts)}"
        )
    a0 = per_agent_state_dicts[0]
    if "_sliced_keys" not in a0:
        raise KeyError(
            "Per-agent optimizer state_dict missing '_sliced_keys' sidecar; this is "
            "required to know which entries were block-parallel and must be re-stacked. "
            "Did you produce the slice with slice_optimizer_state()?"
        )
    if "state" not in a0 or "param_groups" not in a0:
        raise KeyError("Per-agent optimizer state_dict missing 'state' or 'param_groups'")

    sliced_keys = a0["_sliced_keys"]
    out_state = {}
    for param_id, param_state in a0["state"].items():
        new_state = {}
        for k, v0 in param_state.items():
            if (param_id, k) in sliced_keys:
                new_state[k] = torch.stack(
                    [per_agent_state_dicts[i]["state"][param_id][k] for i in range(num_agents)],
                    dim=0,
                )
            else:
                new_state[k] = v0
        out_state[param_id] = new_state
    return {"state": out_state, "param_groups": a0["param_groups"]}
