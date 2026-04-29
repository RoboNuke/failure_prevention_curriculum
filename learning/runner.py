"""Train/eval entry point for BlockSimba-SAC on Isaac Lab tasks.

Mostly skrl boilerplate for SAC; expected to be modified as the project grows.
The Isaac Lab `AppLauncher` must boot before any `isaaclab.envs` / `isaaclab_tasks`
imports — that's why those imports live inside `main()` after `app_launcher.app`.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Project root (parent of learning/) — used to anchor the default --config path so
# the runner works regardless of the user's CWD.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "configs", "exp_cfgs", "default.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BlockSimba-SAC trainer/eval")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Isaac Lab gym id (e.g. Isaac-Stack-Cube-Franka-v0). "
             "TODO: robosuite-isaaclab integration.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Envs PER agent. Total Isaac envs = num_envs * num_agents.",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=1,
        help="Block-parallel agents trained simultaneously.",
    )
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Folder path. Multi-agent: parent containing 0/, 1/, ... subdirs. "
             "Single-agent: a folder with checkpoints/ckpt_<step>.pt directly. "
             "Mode is auto-detected.",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Specific step to load (e.g. 1500). If omitted, the latest ckpt found is used.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="Transitions PER AGENT (literature convention: each agent's training budget "
             "is 1M transitions, like single-env SAC). env.step() count = "
             "total_timesteps // num_envs (where num_envs = envs per agent).",
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=1_000_000,
        help="Replay buffer capacity PER AGENT (literature convention: each agent has "
             "its own ~1M-transition buffer). Per-env depth = memory_size // num_envs.",
    )
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Overrides sac_cfg.experiment.experiment_name from --config.")
    parser.add_argument("--logdir", type=str, default=None,
                        help="Overrides sac_cfg.experiment.directory from --config. "
                             "Relative paths are resolved against the project root.")
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG_PATH,
        help=f"Path to a YAML config file. Defaults to {_DEFAULT_CONFIG_PATH}.",
    )
    AppLauncher.add_app_launcher_args(parser)  # adds --headless, --device, --num_envs is NOT added
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Boot Omniverse before any isaaclab.envs imports.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Project root on sys.path so `models.block_simba` and `learning.sac` resolve
    # regardless of CWD.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401  registers Isaac-* gym ids
    from isaaclab_tasks.utils import parse_env_cfg
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    import dataclasses
    from memory.multi_random import MultiRandomMemory
    from models.block_simba import BlockSimBaActor, BlockSimBaQCritic
    from learning.sac import SAC
    from configs.manager import ConfigManager
    from wrappers import available_wrappers, make_wrapper

    # Load all registered configs from a single YAML file (defaults to configs/default.yaml).
    loaded = ConfigManager.load(args.config)
    sac_cfg = loaded["sac_cfg"]
    model_cfg = loaded["model_cfg"]

    # Cross-cutting consistency: the success-prediction head needs an env wrapper
    # that emits ``infos[success_info_key]``. Catch the misconfig before booting Isaac.
    if sac_cfg.predict_success and sac_cfg.success_wrapper is None:
        raise ValueError(
            "predict_success=True but sac_cfg.success_wrapper is null. Set "
            f"success_wrapper to one of {available_wrappers()} (or disable "
            "predict_success)."
        )

    set_seed(args.seed if args.seed >= 0 else None)

    # ---- env ----
    total_envs = args.num_envs * args.num_agents
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=total_envs)
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    if sac_cfg.success_wrapper is None:
        env = wrap_env(env, wrapper="isaaclab")
    else:
        # Subclasses of skrl's IsaacLabWrapper — bypass wrap_env and instantiate directly.
        env = make_wrapper(sac_cfg.success_wrapper, env)

    device = torch.device(args.device)
    n_agents = args.num_agents
    obs_space = env.observation_space
    act_space = env.action_space

    # Observation preprocessor: YAML carries the class as a string name; SAC resolves
    # it via the registry. We inject the runtime kwargs (size, device) here since
    # YAML can't carry a Box space or torch.device object. Skip if user explicitly
    # set kwargs already.
    if sac_cfg.observation_preprocessor is not None:
        if not isinstance(sac_cfg.observation_preprocessor_kwargs, dict):
            sac_cfg.observation_preprocessor_kwargs = {}
        sac_cfg.observation_preprocessor_kwargs.setdefault("size", obs_space)
        sac_cfg.observation_preprocessor_kwargs.setdefault("device", device)

    # ---- models ----
    actor_kwargs = dataclasses.asdict(model_cfg.actor)
    critic_kwargs = dataclasses.asdict(model_cfg.critic)
    policy = BlockSimBaActor(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        num_agents=n_agents,
        predict_success=sac_cfg.predict_success,
        **actor_kwargs,
    )

    def make_q():
        return BlockSimBaQCritic(
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            num_agents=n_agents,
            **critic_kwargs,
        )

    critic_1, critic_2 = make_q(), make_q()
    target_critic_1, target_critic_2 = make_q(), make_q()

    models = {
        "policy": policy,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
    }

    # ---- replay memory (per-agent partitioned sampling) ----
    # `--memory_size` is interpreted as transitions PER AGENT (lit convention: each agent
    # has its own ~1M buffer). Each agent owns env partition [i*epa, (i+1)*epa), so the
    # per-env depth that yields `memory_size` per-agent transitions is:
    #     per_env_depth = memory_size // num_envs   (where num_envs == envs per agent)
    # skrl physically allocates a single (per_env_depth, total_envs, *) tensor, so the
    # physical storage is per_env_depth * total_envs = memory_size * n_agents transitions.
    per_env_memory = max(1, args.memory_size // args.num_envs)
    realized_per_agent = per_env_memory * args.num_envs
    realized_total_storage = per_env_memory * total_envs
    print(
        f"[runner] memory: requested_per_agent={args.memory_size:,}  "
        f"per_env={per_env_memory:,}  realized_per_agent={realized_per_agent:,}  "
        f"physical_storage={realized_total_storage:,} "
        f"(num_envs/agent={args.num_envs}, n_agents={n_agents})"
    )
    if sac_cfg.predict_success:
        # Trajectory-staged memory: transitions live in a per-env staging buffer until
        # the episode ends, at which point they're committed to the main buffer with
        # the success label baked in. SAC never samples in-progress / unlabeled rows.
        from memory.trajectory_buffered import TrajectoryBufferedMemory

        # Discover the env's max episode length (Isaac Lab manager-based envs expose it).
        max_ep_len = int(getattr(env.unwrapped, "max_episode_length", 0))
        if max_ep_len <= 0:
            raise RuntimeError(
                "predict_success=True requires env.unwrapped.max_episode_length to be set. "
                "Either disable success prediction (sac_cfg.predict_success=false) or use an "
                "env that exposes a max episode length."
            )
        print(f"[runner] env reports max_episode_length={max_ep_len}")
        memory = TrajectoryBufferedMemory(
            memory_size=per_env_memory,
            num_envs=env.num_envs,
            num_agents=n_agents,
            max_episode_length=max_ep_len,
            device=device,
            replacement=True,
        )
    else:
        memory = MultiRandomMemory(
            memory_size=per_env_memory,
            num_envs=env.num_envs,
            num_agents=n_agents,
            device=device,
            replacement=True,
        )

    # ---- SAC config (loaded above; apply CLI overrides) ----
    cfg = sac_cfg
    assert cfg.batch_size % n_agents == 0, (
        f"batch_size ({cfg.batch_size}) must be divisible by num_agents ({n_agents})"
    )
    # Each agent samples cfg.batch_size // n_agents transitions from its own slice; the
    # slice must be able to provide that many distinct transitions.
    batch_per_agent = cfg.batch_size // n_agents
    assert realized_per_agent >= batch_per_agent, (
        f"per-agent replay buffer ({realized_per_agent}) < batch_per_agent ({batch_per_agent}); "
        f"increase --memory_size (need at least {batch_per_agent} per agent) or reduce batch_size"
    )
    # CLI > YAML > auto-generated
    exp_name = args.experiment_name or cfg.experiment.experiment_name or f"{args.task}_sac_N{n_agents}"
    cfg.experiment.experiment_name = exp_name
    if args.logdir is not None:
        cfg.experiment.directory = args.logdir
    # Anchor a relative `directory` to the project root so runs always land at
    # <project_root>/<directory>, not wherever the user happened to invoke from.
    if not os.path.isabs(cfg.experiment.directory):
        cfg.experiment.directory = os.path.join(_PROJECT_ROOT, cfg.experiment.directory)

    # ---- agent ----
    agent = SAC(
        models=models,
        memory=memory,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        cfg=cfg,
        num_agents=n_agents,
    )

    # ---- trainer ----
    # `--total_timesteps` is interpreted as transitions PER AGENT (lit convention: each
    # agent's budget is e.g. 1M transitions, matching single-env SAC). Each agent owns
    # `num_envs` envs (envs per agent), so the env.step() count needed to give every
    # agent that many transitions is:
    #     env_steps = total_timesteps // num_envs
    # All agents share the same trainer step counter (block-parallel: one env.step()
    # advances every agent's envs simultaneously), so this is also the global step
    # count. Floor at 1 so degenerate configs always run at least one step.
    env_steps = max(1, args.total_timesteps // args.num_envs)
    realized_per_agent = env_steps * args.num_envs
    realized_total_transitions = env_steps * total_envs
    print(
        f"[runner] timesteps: requested_per_agent={args.total_timesteps:,}  "
        f"env_steps={env_steps:,}  realized_per_agent={realized_per_agent:,}  "
        f"realized_total_transitions={realized_total_transitions:,} "
        f"(num_envs/agent={args.num_envs}, n_agents={n_agents})"
    )
    trainer = SequentialTrainer(
        cfg={"timesteps": env_steps, "headless": args.headless},
        env=env,
        agents=agent,
    )

    # Init the agent before trainer.train() so per-agent dirs exist for the config
    # dump below. SAC.init() is idempotent — trainer.train() will call it again
    # but the second call returns immediately.
    agent.init(trainer_cfg=trainer.cfg)

    # Snapshot the merged-and-CLI-applied configs into each agent's results dir so
    # any run can be reconstructed later without consulting the original YAML.
    for i in range(n_agents):
        cfg_path = os.path.join(agent.experiment_dir, str(i), "config.yaml")
        ConfigManager.dump(loaded, cfg_path)

    # Optional checkpoint load — works for both train (resume) and eval.
    if args.checkpoint is not None:
        agent.load(args.checkpoint, step=args.checkpoint_step)
    elif args.mode == "eval":
        raise ValueError("--checkpoint is required for --mode eval")

    try:
        if args.mode == "train":
            trainer.train()
        else:
            trainer.eval()
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
