import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from utils import VizDoomGym


def make_env(cfg: str, render_mode=None, frame_skip=4, seed: int = None):
    """Factory function to create a monitored VizDoomGym environment."""
    log_dir = f"./logs/eval_{cfg}/"
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        env = VizDoomGym(cfg=cfg, render_mode=render_mode, frame_skip=frame_skip)
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env, log_dir)
    return _init


def evaluate_trained_model(model_path: str, cfg: str, n_eval_episodes: int, seed: int = None):
    """Load a trained PPO model and evaluate it on the given environment."""
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    eval_env = DummyVecEnv([make_env(cfg, render_mode=None, seed=seed)])

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    print(f"\nEvaluation over {n_eval_episodes} episodes:")
    print(f"   Mean reward = {mean_reward:.2f} ± {std_reward:.2f}\n")

    return model


def record_video(model, cfg: str, video_length: int, seed: int = None):
    """Record a video of the trained model playing the environment."""
    print(f"Recording gameplay for {video_length} steps...")
    video_dir = f"./videos/{cfg}"
    os.makedirs(video_dir, exist_ok=True)

    video_env = DummyVecEnv([make_env(cfg, render_mode="rgb_array", seed=seed)])
    video_env = VecVideoRecorder(
        video_env,
        video_dir,
        record_video_trigger=lambda a_step: a_step == 0,
        video_length=video_length,
        name_prefix=f"ppo-{cfg}-test",
    )

    obs = video_env.reset()
    for step in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = video_env.step(action)
        if done.any():
            obs = video_env.reset()

    video_env.close()
    print(f"Video saved in: {video_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO VizDoom agent.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.zip)")
    parser.add_argument("--cfg", type=str, default="basic", help="Scenario config name (e.g., basic, deadly_corridor)")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--video", action="store_true", help="If set, record a gameplay video")
    parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record in video")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set global seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Run evaluation
    model = evaluate_trained_model(
        model_path=args.model_path,
        cfg=args.cfg,
        n_eval_episodes=args.n_eval_episodes,
        seed=args.seed
    )

    # Optional video recording
    if args.video:
        record_video(
            model=model,
            cfg=args.cfg,
            video_length=args.video_length,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
