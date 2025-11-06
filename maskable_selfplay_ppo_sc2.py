"""
maskable_selfplay_ppo_sc2.py

Headless PPO + PySC2 example for Simple64:
- Continuous action space [x, y] normalized to [0,1]
- CNN Policy (processes feature_screen)
- MaskablePPO automatically handles invalid actions
- Headless: visualize=False, realtime=False
- Uses SubprocVecEnv for parallel envs
"""


from absl import flags
FLAGS = flags.FLAGS
FLAGS(['maskable_selfplay_ppo_sc2.py'])

import copy
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from pysc2.env import sc2_env, environment as sc2_environment
from pysc2.lib import actions as sc2_actions, features
import torch
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers.action_masker import ActionMasker

from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from sb3_contrib.ppo_mask import MaskablePPO

# ---------------- CONFIG ----------------
MAP_NAME = "Simple64"
STEP_MUL = 8
FEATURE_SCREEN_SIZE = 84
TOTAL_TIMESTEPS = 1000000
NUM_ENVS = 2
MODEL_PATH = "maskableselfplay_ppo_sc2.zip"
FROZEN_OPPONENT = "frozen_opponent.zip"
# ---------------------------------------

class SaveEveryNStepsCallback(BaseCallback):
    """
    Callback for saving a model every N steps, ensuring saves are not skipped
    due to timesteps jumping over the exact multiple of save_freq.
    """
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.last_saved = 0  # Track last saved timestep

    def _on_step(self) -> bool:
        # Check if enough timesteps have passed since last save
        if self.num_timesteps - self.last_saved >= self.save_freq:
            save_file = os.path.join(self.save_path, f"maskableselfplay_ppo_sc2.zip")
            self.model.save(save_file)
            self.last_saved = self.num_timesteps
            if self.verbose > 0:
                print(f"Saved model at timestep {self.num_timesteps} to {save_file}")
        return True


class SC2ContinuousCNNEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self):
        super().__init__()

        self._players = [sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Agent(sc2_env.Race.terran)]

        self.sc2_env = sc2_env.SC2Env(
            map_name=MAP_NAME,
            players=self._players,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=FEATURE_SCREEN_SIZE,
                                                      minimap=FEATURE_SCREEN_SIZE),
                use_feature_units=True
            ),
            step_mul=STEP_MUL,
            visualize=False,
            realtime=False,
            game_steps_per_episode=None,
        )

        self.opponent_policy = None
        
        # Observation: HxWxC
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(FEATURE_SCREEN_SIZE, FEATURE_SCREEN_SIZE, 3),
                                            dtype=np.uint8)

        # Track invalid actions + episode stats + valid actions
        self.invalid_action_count = 0
        self.valid_action_count = 0
        self.episode_reward = 0.0

        self.episode_steps = 0

        # All SC2 functions are allowed, tiered restriction will be handled manually
        self.valid_funcs = [f.id for f in sc2_actions.FUNCTIONS]
        self.action_space = spaces.MultiDiscrete([len(self.valid_funcs), 64, 64])

        self._last_obs = None
        self._prev_score = 0.0
        self._prev_kills = 0.0
        self._prev_minerals = 0.0
        self._prev_workers = 0.0
        self._prev_idle_workers = 0.0
        self._prev_supply_despot = 0.0


    def _make_action(self, func_id, x=None, y=None):
        """
        Safely create a PySC2 FunctionCall from func_id and optional x, y.
        Handles argument defaults for queued, control_group, etc.
        """
        func = sc2_actions.FUNCTIONS[func_id]
        args = []

        for arg in func.args:
            if arg.name in ["screen", "screen2", "minimap"]:
                if x is None or y is None:
                    x, y = 32, 32  # default center
                args.append([int(x), int(y)])
            elif arg.name in ["queued"]:
                args.append([0])
            elif arg.name in ["control_group_act", "control_group_id"]:
                args.append([0])
            else:
                args.append([0])

        try:
            return sc2_actions.FunctionCall(func.id, args)
        except Exception as e:
            print(f"Error creating FunctionCall for {func.name}: {e}")
            return sc2_actions.FUNCTIONS.no_op()

    def set_opponent_policy(env, policy):
        env.opponent_policy = policy
        

    def reset(self, seed=None, options=None):
        """
        Reset the environment safely for single-agent or self-play.
        Ensures temp map issues are avoided and last observations are correctly set.
        """
        if seed is not None:
            np.random.seed(seed)
        try:
            # Reset SC2 environment
            obs = self.sc2_env.reset()
        except Exception as e:
            print(f"[reset] SC2 reset failed: {e}. Trying cleanup...")
            # Force cleanup: close the environment, remove temp files
            self.sc2_env.close()
            import glob, os
            temp_maps = glob.glob(os.path.join(os.getenv('TEMP'), 'StarCraft II', 'TempLaunchMap*.SC2Map'))
            for f in temp_maps:
                try:
                    os.remove(f)
                except:
                    pass
            # Recreate environment
            self.sc2_env = sc2_env.SC2Env(
                map_name=MAP_NAME,
                players=self._players,
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(
                        screen=FEATURE_SCREEN_SIZE,
                        minimap=FEATURE_SCREEN_SIZE
                    ),
                    use_feature_units=True,
                ),
                step_mul=STEP_MUL,
                visualize=False,
                realtime=False,
                game_steps_per_episode=None,
            )
            obs = self.sc2_env.reset()

        # Set last observations for both players
        if len(self._players) == 1:
            self._last_obs = obs[0] if isinstance(obs, (list, tuple)) else obs
            self._last_obs2 = None
        elif len(self._players) == 2:
            self._last_obs, self._last_obs2 = obs

        # Reset episode stats
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.valid_action_count = 0
        self.invalid_action_count = 0

        # Reset cumulative metrics
        self._prev_score = 0.0
        self._prev_kills = 0.0
        self._prev_minerals = 0.0
        self._prev_workers = 0.0
        self._prev_idle_workers = 0.0
        self._prev_supply_despot = 0.0

        info = {}

        # Return processed obs for Stable-Baselines3
        return self._process_obs(self._last_obs), info



    def _process_obs(self, ts):
        obs = ts.observation
        layers = []
        for key in ["player_relative", "unit_type", "height_map", "visibility_map"]:
            try:
                layer = np.array(obs["feature_screen"][key])
                if layer.ndim == 2:
                    layers.append(layer)
            except Exception:
                pass
        if len(layers) == 0:
            fs = np.zeros((FEATURE_SCREEN_SIZE, FEATURE_SCREEN_SIZE, 3), dtype=np.uint8)
        else:
            stacked = np.stack(layers, axis=2)
            if stacked.shape[2] < 3:
                repeats = 3 - stacked.shape[2]
                last = np.expand_dims(stacked[:, :, -1], axis=2)
                stacked = np.concatenate([stacked] + [last]*repeats, axis=2)
            fs = stacked[:, :, :3].astype(np.uint8)
        return fs

    def step(self, action):
        # --- Extract player 1 action ---
        func_index, x, y = int(action[0]), float(action[1]), float(action[2])
        func_index = max(0, min(func_index, len(self.valid_funcs)-1))
        func_id = self.valid_funcs[func_index] if func_index < len(self.valid_funcs) else sc2_actions.FUNCTIONS.no_op.id

        available = self._last_obs.observation["available_actions"]

        # --- Build the mask ---
        action_mask = np.zeros(len(self.valid_funcs), dtype=np.float32)
        for idx, fid in enumerate(self.valid_funcs):
            if fid in available:
                action_mask[idx] = 1.0

            # --- Determine player 2 action ---
        if hasattr(self, "opponent_policy") and self.opponent_policy is not None:
            obs2 = self._last_obs2
            action2, _ = self.opponent_policy.predict(obs2, deterministic=True)
            func_id2, x2, y2 = int(action2[0]), float(action2[1]), float(action2[2])
            action2_call = self._make_action(func_id2, x2, y2)
        else:
            action2_call = sc2_actions.FUNCTIONS.no_op()

        # --- Player 1 FunctionCall ---
        action1_call = self._make_action(func_id, x, y)

        # --- Step environment ---
        obs = self.sc2_env.step([action1_call, action2_call])
        obs1, obs2 = obs
        self._last_obs = obs1
        self._last_obs2 = obs2

        # --- Base reward ---
        reward = float(obs1.reward)
        # --- Cumulative metrics ---
        # --- score_cumulative array number values: 
        # --- 0='score', 1='idle_production_time', 2='idle_worker_time', 3='total_value_units',
        # --- 4='total_value_structures', 5='...', 6='collected_vespene', 7='collection_rate_minerals', 8='collection_rate_vespene', 9='spent_minerals', 10='spent_vespene'
        # ---
        # --- Player contains = ['player_id', 'minerals', 'vespene', 'food_used', 'food_cap',
        # --- 'food_army', 'food_workers', 'idle_worker_count', 'army_count', 'warp_gate_count', 'larva_count'] 
        try:
            score_cumulative = obs1.observation["score_cumulative"]
            player_obs = obs1.observation["player"]
            score = score_cumulative[0]
            gathered = score_cumulative[7]
            workers = player_obs[6]
            idle_workers = player_obs[7]
            supply_despot = player_obs[4]
        except (KeyError, AttributeError, TypeError):
            print("KeyError, AttributeError or TypeError")
            score = gathered = workers = idle_workers = supply_despot = 0.0

        #print(f"gathered minerals:",gathered)
        #print(f"score:", score)
        #print("score cumulative:",score_cumulative)
        #print("obs.observation", obs.observation)

   
        # player_obs = obs.observation["player"]
        # print("player : ", player_obs)

        # --- Compute changes ---
        if hasattr(self, "_prev_score"):
            delta_score = score - self._prev_score
            delta_minerals = gathered - self._prev_minerals
            delta_workers = workers - self._prev_workers
            delta_idle_workers = idle_workers - self._prev_idle_workers
            delta_supply_despot = supply_despot - self._prev_supply_despot
        else:
            delta_score = delta_minerals = delta_workers = delta_idle_workers = delta_supply_despot = 0

        #print("delta minerals:", delta_minerals)
        #print("delta score:", delta_score)

        self._prev_score = score
        #self._prev_kills = killed
        self._prev_minerals = gathered
        self._prev_workers = workers
        self._prev_idle_workers = idle_workers
        self._prev_supply_despot = supply_despot

        # --- Normalization ---
        # Cap large jumps to keep PPO stable
        delta_score = np.clip(delta_score, -100, 100) / 100.0
        #delta_kills = np.clip(delta_kills, -50, 50) / 50.0
        delta_minerals = np.clip(delta_minerals, -500, 500) / 500.0
        delta_workers = np.clip(delta_workers, -5, 5) / 5.0
        delta_idle_workers = np.clip(delta_idle_workers, 0, 20) / 20.0  # only penalize positive idle change
        delta_supply_despot = np.clip(delta_supply_despot, -8, 8) / 8.0

        #print("delta minerals after cap:", delta_minerals)
        #print("delta score after cap:", delta_score)
        
        #print("reward : ", 0.5 * reward)
        #print("delta score, score:", 0.3 * delta_score)
        #print("delta score kills:", 0.15 * delta_kills)
        #print("delta score minerals:", 0.05 * delta_minerals)
        #print("delta score villagers:", 0.05 * delta_villagers)
        #print("delta idle Workers: ", -0.2 * delta_idle_workers)

        #print(f"Raw score_minerals: {gathered}")
        # --- Weighted combination ---
        shaped_reward = (
            0.5 * reward
            + 0.3 * delta_score
            #+ 0.15 * delta_kills
            + 0.05 * delta_minerals
            + 0.05 * delta_workers  # reward for training new workers
            + 0.1 * delta_supply_despot
            - 0.2 * delta_idle_workers  # penalize idle workers
        )

        # track episode total
        self.episode_reward += shaped_reward
        
        done = obs1.last()
        info = {"action_mask": action_mask}

        # Increment episode step counter
        self.episode_steps += 1

        # split done into terminated / truncated
        terminated = done
        truncated = False  # or implement a max-steps limit

        if done:
            print(f"Episode finished. Reward/result: {reward}, Valid actions: {self.valid_action_count}, invalid actions {self.invalid_action_count}, episode reward: {self.episode_reward}, Total steps: {self.episode_steps}")
            # Reset episode stats
            self.valid_action_count = 0
            self.episode_steps = 0
            self.episode_reward = 0
            self.invalid_action_count = 0

        return self._process_obs(obs1), shaped_reward, terminated, truncated, info
        

    def close(self):
        self.sc2_env.close()
        
# ---------------- Mask function ----------------
def mask_fn(env):
    info = env._last_obs.observation
    available = info["available_actions"]
    
    # Mask function ID
    func_mask = np.zeros(len(env.valid_funcs), dtype=np.float32)
    for idx, fid in enumerate(env.valid_funcs):
        if fid in available:
            func_mask[idx] = 1.0

    # Mask x and y coordinates (always valid)
    xy_mask = np.ones(64 + 64, dtype=np.float32)

    return np.concatenate([func_mask, xy_mask])




# ---------------- Environment constructor ----------------
def make_env(mask_fn):
    def _init():
        env = SC2ContinuousCNNEnv()
        env = ActionMasker(env, mask_fn)
        return env
    return _init

if __name__ == "__main__":
    #env = DummyVecEnv([make_env])

    # Use SubprocVecEnv to run each env in a separate process
    env_fns = [make_env(mask_fn) for i in range(NUM_ENVS)]  # call make_env
    env = SubprocVecEnv(env_fns)
    env = VecTransposeImage(env) #CNN-friendly format

    print(f"Created {NUM_ENVS} SC2 environments in parallel.")

    save_callback = SaveEveryNStepsCallback(save_freq=10_000, save_path="./")

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = MaskablePPO.load(MODEL_PATH, env)
    else:
        model = MaskablePPO("CnnPolicy", env,
                            learning_rate=0.00005,
                            ent_coef=0.01,
                            n_steps=512,
                            batch_size=64,
                            gamma=0.99,
                            gae_lambda=0.95,
                            max_grad_norm=0.5,
                            clip_range=0.2,
                            verbose=1,
                            device="cuda",
                            tensorboard_log="./selfplay_maskable_ppo_sc2_logs/")

    print("ent_coef:", model.ent_coef)
    print("num steps:", model.n_steps)
    print("Epochs each rollout:", model.n_epochs)
    print("Using device:", model.device)
    print("CUDA available:", torch.cuda.is_available())
    print(f"Training MaskablePPO on {MAP_NAME} for {TOTAL_TIMESTEPS} timesteps...")

    print(f"Training PPO on {MAP_NAME} for {TOTAL_TIMESTEPS} timesteps...")
    
    UPDATE_OPPONENT_EVERY = 5_000
    timesteps_done = 0
    while timesteps_done < TOTAL_TIMESTEPS:
        #env.env_method("set_opponent_policy", policy)

        if os.path.exists(FROZEN_OPPONENT):
            print(f"Loading existing opponent model from {FROZEN_OPPONENT}")
            frozen_opponent = MaskablePPO.load(FROZEN_OPPONENT)
            for env_idx in range(NUM_ENVS):
                env.env_method('set_opponent_policy', frozen_opponent.predict, indices=[env_idx])
        else:
            print(f"Loading opponent model from existing Original model: {MODEL_PATH}")
            frozen_opponent = MaskablePPO.load(MODEL_PATH)
            for env_idx in range(NUM_ENVS):
                env.env_method('set_opponent_policy', frozen_opponent.predict, indices=[env_idx])
                
        # Train for a chunk
        model.learn(total_timesteps=UPDATE_OPPONENT_EVERY, reset_num_timesteps=False, callback=save_callback)
        timesteps_done += UPDATE_OPPONENT_EVERY

        # --- Update frozen opponent in each env ---
        frozen_opponent = copy.deepcopy(model)  # freeze current model
        for env_idx in range(NUM_ENVS):
            env.env_method('set_opponent_policy', frozen_opponent.predict, indices=[env_idx])

        print(f"Updated frozen opponent at {timesteps_done} steps")

        # Save frozen opponent
        frozen_opponent.save(FROZEN_OPPONENT)
        # Optional: save main agent
        model.save(MODEL_PATH)
        print(f"Main agent saved at {timesteps_done} steps")

    env.close()
    print("Self-play training complete.")
