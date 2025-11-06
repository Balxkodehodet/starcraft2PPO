"""
train_ppo_sc2_headless.py

Headless PPO + PySC2 example for MoveToBeacon:
- Continuous action space [x, y] normalized to [0,1]
- CNN Policy (processes feature_screen)
- Headless: visualize=False, realtime=False
- Uses DummyVecEnv for Windows reliability
"""

#import sys
#sys.argv = [sys.argv[0]]  # Prevent Abseil flags issues

from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_ppo_sc2_headless.py'])

from stable_baselines3.common.callbacks import BaseCallback
from pysc2.env import environment as sc2_environment
import os
import numpy as np
import gym
from gym import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions as sc2_actions, features
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

# ----------- Configuration -------------
MAP_NAME = "Simple64" #"MoveToBeacon"
STEP_MUL = 8
FEATURE_SCREEN_SIZE = 84
TOTAL_TIMESTEPS = 1000000
NUM_ENVS = 3
# --------------------------------------

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
            save_file = os.path.join(self.save_path, f"ppo_sc2_headless2.zip")
            self.model.save(save_file)
            self.last_saved = self.num_timesteps
            if self.verbose > 0:
                print(f"Saved model at timestep {self.num_timesteps} to {save_file}")
        return True

class SC2ContinuousCNNEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self):
        super().__init__()

        
        self.sc2_env = sc2_env.SC2Env(
            map_name=MAP_NAME,
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)], #comment out this player 2 bot for single player maps
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

        # Observation: HxWxC numpy array
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(FEATURE_SCREEN_SIZE, FEATURE_SCREEN_SIZE, 3),
            dtype=np.uint8
        )


        # Track invalid actions + episode stats + valid actions
        self.invalid_action_count = 0
        self.valid_action_count = 0
        self.episode_reward = 0.0

        self.episode_steps = 0

        
        self.current_tier = 0
        # Define base tiers individually
        tier0_actions = [
            sc2_actions.FUNCTIONS.no_op.id,
            sc2_actions.FUNCTIONS.select_point.id,
            sc2_actions.FUNCTIONS.select_rect.id,      # select rectangle of units
            sc2_actions.FUNCTIONS.Move_screen.id,
            sc2_actions.FUNCTIONS.Train_SCV_quick.id,
            sc2_actions.FUNCTIONS.Build_SupplyDepot_screen.id,
            sc2_actions.FUNCTIONS.select_idle_worker.id,  # this one helps a lot
        ]

        tier1_actions = [
            sc2_actions.FUNCTIONS.Build_Barracks_screen.id,
            sc2_actions.FUNCTIONS.select_army.id,
            sc2_actions.FUNCTIONS.Train_Marine_quick.id,
        ]

        tier2_actions = [
            
            sc2_actions.FUNCTIONS.Attack_screen.id,
        ]

        # Automatically make tiers cumulative
        self.action_tiers = []
        cumulative = []
        for tier in [tier0_actions, tier1_actions, tier2_actions]:
            cumulative = cumulative + tier  # add new actions
            self.action_tiers.append(cumulative.copy())  # save cumulative tier

        # Tier 3: all functions, NOT cumulative
        tier3_actions = [func.id for func in sc2_actions.FUNCTIONS]
        self.action_tiers.append(tier3_actions)    

        # Use current tier
        self.current_tier = 0
        self.valid_funcs = self.action_tiers[self.current_tier]



        #self.action_space = spaces.MultiDiscrete([len(self.valid_funcs), 64, 64])
        self.action_space = spaces.MultiDiscrete([len(sc2_actions.FUNCTIONS), 64, 64]) #ALL functions are available

        self._last_obs = None
        self.beacon_pos = np.array([31, 31])
        self._prev_score = 0.0
        self._prev_kills = 0.0
        self._prev_minerals = 0.0
        self._prev_villagers = 0.0
        self._prev_idle_workers = 0.0

    def reset(self):
        obs = self.sc2_env.reset()
        self._last_obs = obs[0]
        self.episode_reward = 0.0
        return self._process_obs(obs[0])

    def step(self, action):
        #print("\n🧠 Raw action from PPO:", action)
        # Expect action as array-like: [func_index, x, y]
        func_index, x, y = int(action[0]), float(action[1]), float(action[2])
        # Map index -> actual function id (safe clamp)
        func_index = max(0, min(func_index, len(self.valid_funcs) - 1))

        
        # ✅ Keep old model compatibility — always assume full function range
        # but map chosen index to current tier
        if func_index < len(self.valid_funcs):
            func_id = self.valid_funcs[func_index]
        else:
            # random fallback or just no_op
            func_id = sc2_actions.FUNCTIONS.no_op.id

        penalty = 0.0
        if func_index >= len(self.valid_funcs):
            penalty = -0.05  # discourage choosing actions beyond unlocked tier
  
        
        # Ensure last observation exists
        if not hasattr(self, "_last_obs"):
            self._last_obs = self.sc2_env.reset()[0]

        available = self._last_obs.observation["available_actions"]
        #print("🎯 Available actions in SC2:", available)


        #print(f"👉 func_id selected: {func_id} | Valid funcs tier: {len(self.valid_funcs)}")
        # --- Safe action handling ---
        if func_id not in available:
            func = sc2_actions.FUNCTIONS.no_op
            args = []
            self.invalid_action_count += 1
            
            invalid_action_penalty = -0.005
            valid_action_bonus = 0.0
            func_def = sc2_actions.FUNCTIONS[func_id]
            #print(f"Invalid action {func_id}: {func_def.name} chosen, penalty={invalid_action_penalty:.3f}")
        else:
            func = sc2_actions.FUNCTIONS[func_id]
            #print("Valid action:", func.name)
            args = []
            invalid_action_penalty = 0.0
            self.valid_action_count += 1
            valid_action_bonus = 0.005
            for arg in func.args:
                if arg.name in ["screen", "screen2", "minimap"]:
                    args.append([int(x), int(y)])
                elif arg.name in ["queued"]:
                    args.append([0])
                elif arg.name in ["control_group_act", "control_group_id"]:
                    args.append([0])
                else:
                    args.append([0])

        sc2_action = sc2_actions.FunctionCall(func.id, args)
        obs = self.sc2_env.step([sc2_action])[0]
        self._last_obs = obs

        # --- Base reward ---
        reward = float(obs.reward)
        
        if func_id not in available:
            reward += invalid_action_penalty  # apply adaptive penalty
        else:
            reward += valid_action_bonus
            
        obs_dict = obs.observation

        # Penalize if action chosen outside of tiers..
        reward += penalty
        # --- Cumulative metrics ---
        # --- score_cumulative array number values: 
        # --- 0='score', 1='idle_production_time', 2='idle_worker_time', 3='total_value_units',
        # --- 4='total_value_structures', 5='...', 6='collected_vespene', 7='collection_rate_minerals', 8='collection_rate_vespene', 9='spent_minerals', 10='spent_vespene'
        try:
            score_cumulative = obs.observation["score_cumulative"]
            score = score_cumulative[0]
            gathered = score_cumulative[7]
        except (KeyError, AttributeError, TypeError):
            print("KeyError, AttributeError or TypeError")
            score = gathered = 0.0

        #print(f"gathered minerals:",gathered)
        #print(f"score:", score)
        #print("score cumulative:",score_cumulative)
        #print("obs.observation", obs.observation)

        # --- Compute changes ---
        if hasattr(self, "_prev_score"):
            delta_score = score - self._prev_score
            delta_minerals = gathered - self._prev_minerals
        else:
            delta_score = delta_minerals = 0

        #print("delta minerals:", delta_minerals)
        #print("delta score:", delta_score)

        self._prev_score = score
        #self._prev_kills = killed
        self._prev_minerals = gathered
        #self._prev_villagers = villagers_trained
        #self._prev_idle_workers = idle_workers

        # --- Normalization ---
        # Cap large jumps to keep PPO stable
        delta_score = np.clip(delta_score, -100, 100) / 100.0
        #delta_kills = np.clip(delta_kills, -50, 50) / 50.0
        delta_minerals = np.clip(delta_minerals, -500, 500) / 500.0
        #delta_villagers = np.clip(delta_villagers, -5, 5) / 5.0
        #delta_idle_workers = np.clip(delta_idle_workers, 0, 20) / 20.0  # only penalize positive idle change

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
            #+ 0.05 * delta_villagers  # reward for training new workers
            #- 0.2 * delta_idle_workers  # penalize idle workers
        )

        # track episode total
        self.episode_reward += shaped_reward
        #print("shaped reward:", shaped_reward)
        done = obs.last()
        info = {}

        # Increment episode step counter
        self.episode_steps += 1

        # --- Auto-unlock higher tiers based on valid actions ---
        if done:
            print(f"Episode finished. Valid actions: {self.valid_action_count}, invalid actions {self.invalid_action_count}, episode reward: {self.episode_reward}, Total steps: {self.episode_steps}")

            # Only unlock next tier if enough valid actions were executed
            if self.current_tier == 0:
                MIN_VALID_ACTIONS_TO_UNLOCK = 600
            elif self.current_tier == 1:
                MIN_VALID_ACTIONS_TO_UNLOCK = 700
            elif self.current_tier == 2:
                MIN_VALID_ACTIONS_TO_UNLOCK = 800

            if (self.current_tier < len(self.action_tiers) - 1 and
                self.valid_action_count >= MIN_VALID_ACTIONS_TO_UNLOCK):
                self.current_tier += 1
                self.valid_funcs = self.action_tiers[self.current_tier]
                shaped_reward += 2.0
                print(f"🌟 Expanded to Tier {self.current_tier} — now {len(self.valid_funcs)} valid actions")

            # Reset episode stats
            self.valid_action_count = 0
            self.episode_steps = 0
            self.episode_reward = 0
            self.invalid_action_count = 0


        return self._process_obs(obs), shaped_reward, done, info

    # -------------------- Multiple Envs Setup --------------------
    def make_env(rank):
        return SC2ContinuousCNNEnv()
            


    def _process_obs(self, ts):
        obs = ts.observation
        fs = None

        if "feature_screen" in obs:
            try:
                candidate = np.array(obs["feature_screen"])
                if candidate.ndim == 3 and candidate.shape[0] <= 8:
                    fs = np.moveaxis(candidate, 0, 2)
                elif candidate.ndim == 3 and candidate.shape[2] <= 8:
                    fs = candidate
                elif candidate.ndim == 2:
                    fs = np.expand_dims(candidate, axis=2)
            except Exception:
                fs = None

        if fs is None:
            layers = []
            for key in ["player_relative", "unit_type", "height_map", "visibility_map"]:
                try:
                    layer = np.array(obs["feature_screen"][key])
                    if layer.ndim == 2:
                        layers.append(layer)
                except Exception:
                    try:
                        layer = np.array(obs[key])
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

    def close(self):
        self.sc2_env.close()


# ---------------- PPO Training Setup ----------------
def make_env():
    return SC2ContinuousCNNEnv()


if __name__ == "__main__":
    #env = DummyVecEnv([make_env])

    # Use SubprocVecEnv to run each env in a separate process
    env_fns = [make_env for _ in range(NUM_ENVS)]
    env = SubprocVecEnv(env_fns)

    print(f"Created {NUM_ENVS} SC2 environments in parallel.")

    save_callback = SaveEveryNStepsCallback(save_freq=10_000, save_path="./")


    model_path = "ppo_sc2_headless2.zip"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = PPO.load(model_path, env)
    else:
        model = PPO("CnnPolicy", env,
                    ent_coef=0.02,
                    verbose=1,
                    device="cuda",
                    tensorboard_log="./ppo_sc2_logs/")

    model.lr_schedule = lambda _: 0.00003
        
    model.ent_coef = 0.02

    print(f"updated learning rate: {model.lr_schedule(0):.6f}")
    print(f"Updated entropy co efficient: {model.ent_coef}")
    print(f"N_steps: {model.n_steps}")
    print(f"batch size: {model.batch_size}")
    
    print("Using device:", model.device)
    print("CUDA available:", torch.cuda.is_available())    

    print(f"Training PPO on {MAP_NAME} for {TOTAL_TIMESTEPS} timesteps...")
    print("model ent_coef:", model.ent_coef)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=save_callback)
    model.save(model_path)
    print("model ent_coef:", model.ent_coef)

    print("Training complete. Model saved to", model_path)
    env.close()
