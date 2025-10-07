import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import VizDoomGym, TrainAndLoggingCallback


def make_env(cfg: str, log_dir: str, frame_skip: int):
    """Factory function to create a monitored VizDoom environment."""
    def _init():
        env = VizDoomGym(cfg=cfg, render_mode=None, frame_skip=frame_skip)
        env = Monitor(env, log_dir)
        return env
    return _init


def train(
    cfg: str,
    checkpoint_dir: str,
    log_dir: str,
    timesteps: int,
    learning_rate: float,
    gamma: float,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    frame_skip: int,
    verbose: int = 1
):
    """Train a PPO agent on the given VizDoom environment."""
    # 1. Create environment
    env = DummyVecEnv([make_env(cfg, log_dir, frame_skip)])

    # 2. Create PPO model
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=verbose,
        tensorboard_log=log_dir,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
    )

    # 3. Define callback
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=checkpoint_dir)

    # 4. Train model
    model.learn(total_timesteps=timesteps, callback=callback)

    # 5. Save final model
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "final_model")
    model.save(model_path)
    print(f"Training finished. Final model saved to: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent on VizDoom.")
    parser.add_argument("--cfg", type=str, default="basic", help="Scenario config name")
    parser.add_argument("--checkpoint_dir", type=str, default="./train/train_basic", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs/log_basic", help="Directory for Monitor and TensorBoard logs")
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of training timesteps")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run per environment per update")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--n_epochs", type=int, default=4, help="Number of epochs when optimizing the surrogate loss")
    parser.add_argument("--frame_skip", type=int, default=4, help="Frame skip for VizDoom")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        cfg=args.cfg,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        frame_skip=args.frame_skip,
        verbose=args.verbose,
    )
