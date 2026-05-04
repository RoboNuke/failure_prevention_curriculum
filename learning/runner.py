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
        "--config",
        type=str,
        default=_DEFAULT_CONFIG_PATH,
        help=f"Path to a YAML config file. Provides runner_cfg/sac_cfg/model_cfg. "
             f"Defaults to {_DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Overrides sac_cfg.experiment.experiment_name from --config.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Overrides sac_cfg.experiment.directory from --config. "
             "Relative paths are resolved against the project root.",
    )
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
        help="Specific step to load. If omitted, the latest ckpt found is used.",
    )
    # All of the following used to be required CLI flags. They now default to
    # None and fall back to runner_cfg in YAML when omitted. Still overridable
    # from CLI for one-off experiments.
    parser.add_argument("--task", type=str, default=None,
                        help="Overrides runner_cfg.task. e.g. Isaac-Lift-Cube-Franka-v0.")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Overrides runner_cfg.num_envs (envs PER agent).")
    parser.add_argument("--num_agents", type=int, default=None,
                        help="Overrides runner_cfg.num_agents (block-parallel agents).")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Overrides runner_cfg.total_timesteps in train mode "
                             "(transitions PER AGENT).")
    parser.add_argument("--eval_timesteps", type=int, default=None,
                        help="Overrides runner_cfg.eval_timesteps in eval mode.")
    parser.add_argument("--memory_size", type=int, default=None,
                        help="Overrides runner_cfg.memory_size (replay buffer per agent).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Overrides runner_cfg.seed. -1 means non-deterministic.")
    AppLauncher.add_app_launcher_args(parser)  # adds --headless, --device
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
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    import dataclasses
    from memory.multi_random import MultiRandomMemory
    from models.block_simba import BlockSimBaActor, BlockSimBaQCritic
    from learning.sac import SAC
    from configs.manager import ConfigManager
    from wrappers import (
        available_wrappers,
        default_wrapper_for_task,
        fallback_wrapper_name,
        make_wrapper,
    )

    # Load all registered configs from a single YAML file (defaults to configs/default.yaml).
    loaded = ConfigManager.load(args.config)
    runner_cfg = loaded["runner_cfg"]
    sac_cfg = loaded["sac_cfg"]
    model_cfg = loaded["model_cfg"]

    # CLI > YAML for runner-level fields. Apply overrides to the loaded RunnerCfg so
    # the dump-config-to-disk step at the end records the values actually used.
    if args.task is not None:            runner_cfg.task = args.task
    if args.num_envs is not None:        runner_cfg.num_envs = args.num_envs
    if args.num_agents is not None:      runner_cfg.num_agents = args.num_agents
    if args.total_timesteps is not None: runner_cfg.total_timesteps = args.total_timesteps
    if args.eval_timesteps is not None:  runner_cfg.eval_timesteps = args.eval_timesteps
    if args.memory_size is not None:     runner_cfg.memory_size = args.memory_size
    if args.seed is not None:            runner_cfg.seed = args.seed

    # In eval mode the trainer runs for runner_cfg.eval_timesteps instead of total_timesteps.
    effective_total_timesteps = (
        runner_cfg.eval_timesteps if args.mode == "eval" else runner_cfg.total_timesteps
    )

    # Auto-select a success wrapper for known tasks (e.g. Lift) when the YAML didn't
    # set one. Always informs the user so the auto-application isn't invisible.
    if sac_cfg.success_wrapper is None:
        auto = default_wrapper_for_task(runner_cfg.task)
        if auto is not None:
            print(
                f"[runner] auto-selecting success wrapper '{auto}' for task "
                f"'{runner_cfg.task}' (set sac_cfg.success_wrapper explicitly to override)."
            )
            sac_cfg.success_wrapper = auto

    # Cross-cutting consistency: the success-prediction head needs an env wrapper
    # that emits ``infos[success_info_key]``. Catch the misconfig before booting Isaac.
    if sac_cfg.predict_success and sac_cfg.success_wrapper is None:
        raise ValueError(
            "predict_success=True but sac_cfg.success_wrapper is null. Set "
            f"success_wrapper to one of {available_wrappers()} (or disable "
            "predict_success)."
        )

    # YAML-friendly reward shaping: rewards_shaper_scale (float) -> multiplicative lambda.
    # Mirrors skrl runner's convention. Direct assignment of rewards_shaper still wins
    # if a programmatic caller set it before this point.
    if sac_cfg.rewards_shaper is None and sac_cfg.rewards_shaper_scale is not None:
        scale = float(sac_cfg.rewards_shaper_scale)
        sac_cfg.rewards_shaper = lambda rewards, *args, **kwargs: rewards * scale
    elif sac_cfg.rewards_shaper is not None and sac_cfg.rewards_shaper_scale is not None:
        raise ValueError(
            "Both sac_cfg.rewards_shaper and sac_cfg.rewards_shaper_scale are set; "
            "use exactly one."
        )

    set_seed(runner_cfg.seed if runner_cfg.seed >= 0 else None)

    # ---- env ----
    total_envs = runner_cfg.num_envs * runner_cfg.num_agents
    env_cfg = parse_env_cfg(runner_cfg.task, device=args.device, num_envs=total_envs)
    env = gym.make(runner_cfg.task, cfg=env_cfg, render_mode=None)
    # Always wrap with a subclass of skrl's IsaacLabWrapper. When no task-specific
    # wrapper was selected, fall back to RewardDecompositionWrapper so every
    # manager-based task gets per-env per-term reward logging in per-episode units
    # (matches the units of `Reward / Total reward (mean)`). Direct-API envs
    # without a reward_manager fall through gracefully — the hook is a no-op.
    selected_wrapper = sac_cfg.success_wrapper or fallback_wrapper_name()
    env = make_wrapper(selected_wrapper, env)

    device = torch.device(args.device)
    n_agents = runner_cfg.num_agents
    obs_space = env.observation_space
    act_space = env.action_space

    # Asymmetric actor-critic detection: skrl's IsaacLabWrapper exposes `state_space`
    # (and `state()` returning a tensor) when the underlying env returns a
    # {"policy": ..., "critic": ...} dict obs. When present, the critic is built
    # with the (typically larger) state vector and SAC stores both obs/states in
    # memory. Strict: detection is automatic; presence of state_space ALSO
    # requires env.state() to be non-None at runtime, which SAC enforces on the
    # first record_transition call.
    state_space = getattr(env, "state_space", None)
    if state_space is not None:
        print(
            f"[runner] asymmetric actor-critic detected: obs_dim={obs_space.shape[0]}, "
            f"state_dim={state_space.shape[0]} (critic uses state)"
        )

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

    # Critic input space: state_space when asymmetric, else obs_space.
    critic_input_space = state_space if state_space is not None else obs_space

    def make_q():
        return BlockSimBaQCritic(
            observation_space=critic_input_space,
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
    per_env_memory = max(1, runner_cfg.memory_size // runner_cfg.num_envs)
    realized_per_agent = per_env_memory * runner_cfg.num_envs
    realized_total_storage = per_env_memory * total_envs
    print(
        f"[runner] memory: requested_per_agent={runner_cfg.memory_size:,}  "
        f"per_env={per_env_memory:,}  realized_per_agent={realized_per_agent:,}  "
        f"physical_storage={realized_total_storage:,} "
        f"(num_envs/agent={runner_cfg.num_envs}, n_agents={n_agents})"
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
    # batch_size is interpreted PER AGENT — each agent samples cfg.batch_size
    # transitions from its own env-partition slice. Total memory draw per
    # gradient step is cfg.batch_size * num_agents.
    assert realized_per_agent >= cfg.batch_size, (
        f"per-agent replay buffer ({realized_per_agent}) < batch_size ({cfg.batch_size}); "
        f"increase memory_size (need at least {cfg.batch_size} per agent) or reduce batch_size"
    )
    print(
        f"[runner] batch_size={cfg.batch_size} per agent  "
        f"(memory.sample returns {cfg.batch_size * n_agents:,} total rows / grad step)"
    )
    # CLI > YAML > auto-generated
    exp_name = args.experiment_name or cfg.experiment.experiment_name or f"{runner_cfg.task}_sac_N{n_agents}"
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
        state_space=state_space,
        device=device,
        cfg=cfg,
        num_agents=n_agents,
    )

    # ---- trainer ----
    # `total_timesteps` is interpreted as raw env_steps (env.step() calls). One env_step
    # advances every parallel env by one tick, so the per-agent transition count is
    # `env_steps * num_envs` and total physical transitions written is
    # `env_steps * num_envs * num_agents`. Floor at 1 so degenerate configs always run.
    env_steps = max(1, effective_total_timesteps)
    transitions_per_agent = env_steps * runner_cfg.num_envs
    realized_total_transitions = env_steps * total_envs
    print(
        f"[runner] timesteps ({args.mode}): env_steps={env_steps:,}  "
        f"transitions_per_agent={transitions_per_agent:,}  "
        f"realized_total_transitions={realized_total_transitions:,} "
        f"(num_envs/agent={runner_cfg.num_envs}, n_agents={n_agents})"
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

    # Wrap trainer.train()/eval() so any exception is flushed to stdout BEFORE
    # simulation_app.close() runs — Isaac's shutdown can swallow late stderr and
    # mask exceptions, leaving rc=0 with an empty checkpoint dir as the only
    # symptom. Re-raise so the launcher sees a non-zero exit.
    import traceback
    print(f"[runner] entering trainer.{args.mode}() with timesteps={trainer.cfg.timesteps}",
          flush=True)
    train_exc: BaseException | None = None
    try:
        if args.mode == "train":
            trainer.train()
        else:
            trainer.eval()
        print(f"[runner] trainer.{args.mode}() returned normally", flush=True)
    except BaseException as e:
        train_exc = e
        print(f"[runner] trainer.{args.mode}() raised {type(e).__name__}: {e}",
              flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[runner] env.close() raised: {e!r}", flush=True)
        # If training raised, exit non-zero NOW — Isaac's simulation_app.close()
        # internally calls os._exit(0) on shutdown, which would otherwise mask
        # our exception and report rc=0 to the launcher.
        if train_exc is not None:
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
        try:
            simulation_app.close()
        except Exception as e:
            print(f"[runner] simulation_app.close() raised: {e!r}", flush=True)


if __name__ == "__main__":
    main()
