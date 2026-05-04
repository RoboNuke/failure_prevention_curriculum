"""Diagnostic: empirical vs theoretical tanh saturation rate of BlockSimBaActor.

Sweeps `last_layer_scale` (and a couple of `act_init_std` settings), feeds the actor
random N(0, I) observations, samples squashed-Gaussian actions, and reports:

  empirical_sat       fraction of action dims with |a| > threshold (after tanh)
  theory_sat_full     P(|N(mu_emp, s_emp^2)| > atanh(threshold)) with empirical (mu, s) of u
  theory_sat_noise    same but assuming mu=0, s=sigma_init  (the "scale has no effect" baseline)
  theory_sat_init     prediction from init-only analysis: u ~ N(0, 2*scale^2 + sigma_init^2)

The init-only prediction comes from: BlockLayerNorm output has per-dim variance ~= 1, the
final BlockLinear is kaiming_normal (Var(w) = 2/hidden) scaled by last_layer_scale, so
Var(mean_action) ~= 2 * last_layer_scale^2. Adding the noise variance sigma_init^2 gives
the total Var(u). Discrepancies vs. empirical_sat point to network-side surprises.
"""

import argparse
import math

import numpy as np
import torch
from gymnasium import spaces

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from models.block_simba import BlockSimBaActor


def _norm_sf(x: float) -> float:
    """P(Z > x) for Z ~ N(0, 1) using erfc; avoids scipy dependency."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def two_sided_sat(threshold: float, mu: float, sigma: float) -> float:
    """P(|N(mu, sigma^2)| > atanh(threshold)) — fraction of pre-tanh values that saturate."""
    if sigma <= 0:
        return 0.0
    a = math.atanh(threshold)
    upper = _norm_sf((a - mu) / sigma)            # P(u > a)
    lower = _norm_sf((a + mu) / sigma)            # P(u < -a)
    return upper + lower


def _make_actor(obs_dim, act_dim, scale, act_init_std, hidden, blocks, device, num_agents=1):
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    return BlockSimBaActor(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        num_agents=num_agents,
        act_init_std=act_init_std,
        actor_n=blocks,
        actor_latent=hidden,
        last_layer_scale=scale,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        use_state_dependent_std=False,
        predict_success=False,
    ).to(device)


@torch.no_grad()
def sweep(
    scales,
    act_init_stds,
    obs_dim=64,
    act_dim=8,
    batch=8192,
    hidden=512,
    blocks=2,
    thresholds=(0.99, 0.999),
    seed=0,
    device="cuda",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"obs_dim={obs_dim} act_dim={act_dim} hidden={hidden} blocks={blocks} batch={batch}")
    print(f"obs ~ N(0, I);  thresholds={thresholds}\n")

    for sigma_init in act_init_stds:
        print(f"=== act_init_std = {sigma_init}  (log_std_init = {math.log(sigma_init):+.4f}) ===")
        header = (
            f"{'scale':>8} {'mu_mean':>9} {'mu_std':>8} {'u_mean':>8} {'u_std':>8}  "
            + "  ".join(
                f"{'thr=' + f'{t}':>10} {'emp':>7} {'th_full':>8} {'th_noise':>9} {'th_init':>8}"
                for t in thresholds
            )
        )
        print(header)
        for scale in scales:
            torch.manual_seed(seed)  # reseed so observations match across scales
            actor = _make_actor(obs_dim, act_dim, scale, sigma_init, hidden, blocks, device)
            obs = torch.randn(batch, obs_dim, device=device)

            mean_actions, outputs = actor.compute({"observations": obs}, role="policy")
            log_std = outputs["log_std"]
            log_std = log_std.clamp(-20.0, 2.0)
            sigma = log_std.exp()

            # Sample many actions per state for stable saturation estimates.
            eps = torch.randn_like(mean_actions)
            u = mean_actions + sigma * eps
            actions = torch.tanh(u)

            mu_mean = mean_actions.mean().item()
            mu_std = mean_actions.std().item()
            u_mean = u.mean().item()
            u_std = u.std().item()

            row_prefix = f"{scale:>8.3g} {mu_mean:>+9.4f} {mu_std:>8.4f} {u_mean:>+8.4f} {u_std:>8.4f} "
            row = row_prefix
            for t in thresholds:
                emp = (actions.abs() > t).float().mean().item()
                th_full = two_sided_sat(t, mu_mean, u_std)
                th_noise = two_sided_sat(t, 0.0, sigma_init)
                # Init-only prediction: Var(mean) ~= 2 * scale^2, mu ~= 0.
                s_init = math.sqrt(2.0 * scale * scale + sigma_init * sigma_init)
                th_init = two_sided_sat(t, 0.0, s_init)
                row += f"  {'':>10} {emp:>7.4f} {th_full:>8.4f} {th_noise:>9.4f} {th_init:>8.4f}"
            print(row)
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--obs-dim", type=int, default=64)
    p.add_argument("--act-dim", type=int, default=8)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    scales = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    act_init_stds = [0.60653066, 1.0]   # default (log_std=-0.5) and current YAML (log_std=0)

    sweep(
        scales=scales,
        act_init_stds=act_init_stds,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        batch=args.batch,
        hidden=args.hidden,
        blocks=args.blocks,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
