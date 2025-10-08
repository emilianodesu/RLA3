import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from utils import VizDoomGym, TrainAndLoggingCallback


def make_env(cfg, frame_skip, log_dir, render_mode, seed, cfg_path="./content/VizDoom/scenarios"):
    """
    Returns a thunk to create a single VizDoom environment with proper seeding and monitoring.
    Useful for DummyVecEnv/SubprocVecEnv.
    """
    def _init():
        env = VizDoomGym(cfg=cfg, frame_skip=frame_skip, render_mode=render_mode, cfg_path=cfg_path)
        env.seed(seed) ###### Check
        env = Monitor(env, log_dir)
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent on VizDoom.")

    # Basic experiment setup
    parser.add_argument("--run_name", type=str, default=None, help="Optional name for this run (used for folders).")
    parser.add_argument("--cfg", type=str, default="basic", help="Scenario config (e.g. basic, deadly_corridor).")
    parser.add_argument("--cfg_path", type=str, default="./content/VizDoom/scenarios", help="Path to scenario configs.")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Number of timesteps to train.")
    parser.add_argument("--frame_skip", type=int, default=4, help="Number of frames to skip per action.")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Logging / checkpointing
    parser.add_argument("--check_freq", type=int, default=10000, help="Save checkpoint every N steps.")
    parser.add_argument("--verbose", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    # Build folder names with optional run_name
    run_suffix = f"_{args.run_name}" if args.run_name else ""
    checkpoint_dir = f"./train/{args.cfg}{run_suffix}"
    log_dir = f"./logs/{args.cfg}{run_suffix}"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Set seeds globally
    set_random_seed(args.seed)

    # 1. Create VecEnv
    env_fns = [
        make_env(
            cfg=args.cfg,
            frame_skip=args.frame_skip,
            log_dir=log_dir,
            render_mode=None,
            seed=args.seed + i,
            cfg_path=args.cfg_path
        )
        for i in range(args.n_envs)
    ]

    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # 2. Create PPO model
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=args.verbose,
        tensorboard_log=log_dir,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        clip_range=args.clip_range,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        device=args.device,
        normalize_advantage=True,
        policy_kwargs={
            "ortho_init": True,
            # Example custom architecture, can be modified:
            "net_arch": [512, 512]
        }
    )

    # 3. Callback for checkpointing
    callback = TrainAndLoggingCallback(
        check_freq=args.check_freq,
        save_path=checkpoint_dir,
        save_best_only=False
    )

    # 4. Train the model
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    # 5. Save final model
    model.save(os.path.join(checkpoint_dir, "final_model"))
    print(f"Training finished. Final model saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
