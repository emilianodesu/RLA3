import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vizdoom import DoomGame
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback


class VizDoomGym(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg_path="./content/VizDoom/scenarios", cfg="basic", frame_skip=None, render_mode="rgb_array"):
        super().__init__()

        self.cfg = cfg
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.game = DoomGame()
        self.game.load_config(os.path.join(cfg_path, f"{cfg}.cfg"))
        self.np_random, _ = seeding.np_random(None)

        # Only show window if human render mode
        self.game.set_window_visible(self.render_mode == "human")
        self.game.init()

        # Define the action space and observation space
        self.observation_space = Box(
            low=0, high=255, shape=(100, 160, 1), dtype=np.uint8
        )
        if cfg == "deadly_corridor":
            self.action_space = Discrete(7)
            self.damage_taken = 0
            self.hit_count = 0
            self.ammo = 52
        else:
            self.action_space = Discrete(3)

    def seed(self, seed=None):
        """Set RNG seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Identity action encoding
        actions = np.identity(self.action_space.n, dtype=np.uint8)

        skip = self.frame_skip if self.frame_skip is not None else 1

        # Take action
        reward = self.game.make_action(actions[action], skip)

        # If episode already finished -> return dummy state immediately
        if self.game.is_episode_finished():
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            reward = 0.0
            info = {}
            return obs, reward, True, False, info

        # Otherwise get state
        state = self.game.get_state().screen_buffer
        obs = self._grayscale(state)
        info = {"variables": self.game.get_state().game_variables}

        # Reward shaping for deadly_corridor
        if self.cfg == "deadly_corridor":
            health, damage_taken, hit_count, ammo = self.game.get_state().game_variables

            reward -= (damage_taken - self.damage_taken) * 10
            reward += (hit_count - self.hit_count) * 200
            reward -= (ammo - self.ammo) * 5

            self.damage_taken = damage_taken
            self.hit_count = hit_count
            self.ammo = ammo

        return obs, reward, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.game.new_episode()
        # Handle case where no state is available after reset
        state = self.game.get_state()
        if state is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = {}
        else:
            obs = self._grayscale(state.screen_buffer)
            info = {"variables": state.game_variables}

        return obs, info

    def render(self):
        if self.render_mode == "human":
            return None
        elif self.render_mode == "rgb_array":
            state = self.game.get_state()
            if state is None:
                return None
            frame = np.moveaxis(state.screen_buffer, 0, -1)
            return frame
        else:
            return None

    def close(self):
        self.game.close()

    def sample_action(self):
        return self.np_random.integers(self.action_space.n)

    @staticmethod
    def _grayscale(observation):
        if observation is None or not isinstance(observation, np.ndarray):
            return np.zeros((100, 160, 1), dtype=np.uint8)

        try:
            # Move channel axis: (C, H, W) -> (H, W, C)
            frame = np.moveaxis(observation, 0, -1)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
            state = np.reshape(resized, (100, 160, 1))
            return state
        except Exception as e:
            print(f"[WARN] _grayscale failed: {e}")
            return np.zeros((100, 160, 1), dtype=np.uint8)


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, save_path, check_freq=10000, save_best_only=False, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.save_best_only = save_best_only
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if self.save_best_only:
                # Compute mean reward over last 100 episodes
                rewards = self.model.ep_info_buffer
                if rewards:
                    mean_reward = np.mean([ep_info['r'] for ep_info in rewards])
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        model_path = os.path.join(self.save_path, "best_model")
                        self.model.save(model_path)
                        if self.verbose > 0:
                            print(f"Best model saved with mean reward {mean_reward:.2f}")
            else:
                model_path = os.path.join(self.save_path, f"checkpoint_{self.n_calls}")
                self.model.save(model_path)
                if self.verbose > 0:
                    print(f"Checkpoint saved: {model_path}")
        return True


def run_vizdoom_demo(cfg_path="./content/VizDoom/scenarios", cfg="basic", episodes=1, steps=20, render_mode="human", frame_skip=4, seed=42):
    print(f"--- Starting VizDoom demo: {cfg}, render_mode={render_mode}, frame_skip={frame_skip} ---")
    np.random.seed(seed)

    # 1. Create environment
    env = VizDoomGym(cfg_path=cfg_path, cfg=cfg, render_mode=render_mode, frame_skip=frame_skip)

    # 2. Check validity
    print("\n[ Gymnasium check_env validation ]")
    check_env(env, warn=True)

    # 3. Run random episodes
    for ep in range(episodes):
        print(f"\n--- Episode {ep + 1} ---")
        obs, info = env.reset(seed=seed + ep)
        done, trunc = False, False
        step = 0
        last_rgb_frame = None

        while not (done or trunc) and step < steps:
            action = env.sample_action()
            obs, reward, done, trunc, info = env.step(action)

            print(f"Step {step:03d} | action={action} | reward={reward:.2f} | info={info}")
            step += 1

        # Plot last observation
        plt.imshow(obs.squeeze(), cmap="gray")
        plt.title(f"Episode {ep + 1} final observation")
        plt.show()

        rgb_frame = env.render()
        if rgb_frame is not None:
            last_rgb_frame = rgb_frame

        # Plot last raw RGB frame if available
        if last_rgb_frame is not None:
            plt.imshow(last_rgb_frame)
            plt.title(f"Episode {ep + 1} last raw RGB frame")
            plt.show()

    env.close()
    print("\n--- Demo finished ---")


if __name__ == "__main__":
    run_vizdoom_demo(cfg="basic", episodes=1, steps=5, render_mode="rgb_array", frame_skip=4)
