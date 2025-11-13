"""
customCNN_ppo_x10default.py

Headless PPO + PySC2 example for MoveToBeacon:
- Continuous action space [x, y] normalized to [0,1]
- CNN Policy (processes feature_screen)
- MaskablePPO automatically handles invalid actions
- Headless: visualize=False, realtime=False
- Uses SubprocVecEnv for parallel envs
"""


from absl import flags
FLAGS = flags.FLAGS
FLAGS(['customCNN_ppo_x10default.py'])


from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from pysc2.env import sc2_env, environment as sc2_environment
from pysc2.lib import actions as sc2_actions, features
from pysc2.lib import units
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from collections import deque
from stable_baselines3.common.monitor import Monitor
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.ppo_mask import MaskablePPO

# ---------------- CONFIG ----------------
MAP_NAME = "Simple64"
STEP_MUL = 8
FEATURE_SCREEN_SIZE = 84
TOTAL_TIMESTEPS = 1000000
NUM_ENVS = 3
MODEL_PATH = "v3_customCNN_ppo_x10.zip" #"customCNN_ppo_x10.zip"
START_STEPS = 400 #How many timesteps should an episode last for in the beginning?
CHECK_FREQ = 200 #How many episodes should pass before checking reward mean
TARGET_RATE = 0.7 # Average reward each step before upgrading to next curriculum
INC_STEPS = 200 #How many timesteps should each level increase the episode
ATTEMPTNR = 3 # increase this by 1 before each new restart of the model to update the logs
# ---------------------------------------

class CustomDeepCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=2048):
        super(CustomDeepCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Finn ut output-størrelse etter CNN-delen
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        # Større MLP-del
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))



class CurriculumCallback(BaseCallback):
    def __init__(self,env,target_rate=TARGET_RATE, increase_steps=INC_STEPS, check_freq=CHECK_FREQ, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.target_rate = target_rate
        self.increase_steps = increase_steps
        self.check_freq = check_freq
        self.reward_rates = []
    def _on_step(self):
        infos = self.locals["infos"]
        for info in infos:
            final = info.get("final_info") or info.get("terminal_observation")
            if final is not None:
                ep_info = info.get("episode")
                if ep_info is not None:
                    r = ep_info["r"]

                    l = ep_info["l"]

                    reward_rate = r / max(l, 1)
                    print(f"Episode reward rate: {reward_rate}") 
                    self.reward_rates.append(reward_rate)

        # Check curriculum every check_freq episodes
        if len(self.reward_rates) >= self.check_freq:
            mean_rate = np.mean(self.reward_rates[-self.check_freq:])
            print(f"Mean rate over the last {self.check_freq} episodes: {mean_rate}")

            if mean_rate >= self.target_rate:
                #Increase in all envs
                for e in self.env.envs:
                    e.max_episode_steps += self.increase_steps
                    
                print(f"Successfully achieved higher mean rate than target rate")
                if self.verbose:
                    print(f"Curriculum: Increased max steps -> {self.env.envs[0].max_episode_steps}")

            self.reward_rates = []        
            
        return True

class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, start_steps=START_STEPS):
        super().__init__(env)
        self.max_episode_steps=start_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        self.current_step += 1

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Force episode termination if max steps reached
        if self.current_step >= self.max_episode_steps:
            truncated = True
            info["TimeLimit.truncated"] = True
            
        return obs, reward, terminated, truncated, info

class ShapedRewardCallback(BaseCallback):
    def __init__(self, base_log_dir="./customCNN_ppo_x10_logs", verbose=0):
        super(ShapedRewardCallback, self).__init__(verbose)
        self.base_log_dir = base_log_dir
        self.episode_rewards = {}
        self.episode_count = 0
        self.writer = None  # writer will be created dynamically
        self.total_reward = 0.0
        self.last200 = 0

    def _init_writer(self):
        """Create a new folder and writer for the current episode."""
        run_dir = os.path.join(self.base_log_dir, f"{ATTEMPTNR}_run_{self.episode_count:05d}")
        os.makedirs(run_dir, exist_ok=True)
        if self.writer:
            self.writer.close()
        self.writer = SummaryWriter(log_dir=run_dir)
        if self.verbose > 0:
            print(f"[TensorBoard] Started new run: {run_dir}")    

    def _on_step(self):
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        num_envs = len(infos)

        for i in range(num_envs):
            info = infos[i]
            done = dones[i]

            shaped = rewards[i]

            self.episode_rewards[i] = self.episode_rewards.get(i, 0) + shaped

            if done:
                # Start new log folder for this episode
                self._init_writer()
                
                self.writer.add_scalar(
                    "Rewards/ShapedReward_episode",
                    self.episode_rewards[i],
                    self.episode_count,
                )
                self.total_reward += self.episode_rewards[i]
                
                print(f"Episode {self.episode_count} done")
                if self.last200 >= 200:
                    meanreward200 = self.total_reward / max(self.last200, 1)
                    print(f"Total reward across last 200 episode: ", self.total_reward)
                    print(f"Mean single episode reward last 200 episodes: ", meanreward200)
                    self.last200 = 0
                    self.total_reward = 0.0
                    
                    
                self.episode_rewards[i] = 0
                self.episode_count += 1
                self.last200 += 1
                

        return True

    def _on_training_end(self):
        self.writer.close()


                


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
            save_file = os.path.join(self.save_path, MODEL_PATH)
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
                         sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)]

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

        # Observation: HxWxC
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(FEATURE_SCREEN_SIZE, FEATURE_SCREEN_SIZE, 3),
                                            dtype=np.uint8)

        # Track invalid actions + episode stats + valid actions
        self.invalid_action_count = 0
        self.valid_action_count = 0
        self.episode_reward = 0.0

        self.existing_buildings = set()  # initialize empty

        # Command Center flight timer
        self.cc_lift_timer = 0

        # Command center lift attempts
        self.num_lift_attempts = 0

        # Command center condition if it has attempted flying
        self.command_center_flying = False

        # Track episodes
        self.episode = 0 

        # Store last actions so that it can be penalized if it is over 10 actions repeated
        self.recent_actions = deque(maxlen=10)
        self.repeated_action_count = 0


        self.episode_steps = 0

        # All SC2 functions are allowed, tiered restriction will be handled manually
        self.valid_funcs = [f.id for f in sc2_actions.FUNCTIONS]
        self.action_space = spaces.MultiDiscrete([len(self.valid_funcs), 64, 64])

        self._last_obs = None
        self._first_time = 0
        self._prev_score = 0.0
        self._prev_kills = 0.0
        self._prev_minerals = 0.0
        self._prev_workers = 0.0
        self._prev_idle_workers = 0.0
        self._prev_supply_despot = 0.0
        self._prev_vespene = 0.0
        self._prev_army_power = 0.0
        self._prev_idle_production_time = 0.0
        # Store all scores
        self._prev_high_score = 0.0
        # Store the first step as true
        self.first_step = True


    def reset(self, seed=None, options=None):
        """
        Reset the environment safely for single-agent or self-play.
        Ensures temp map issues are avoided and last observations are correctly set.
        """
        if seed is not None:
            np.random.seed(seed)
        try:
            # Reset SC2 environment
            obs = self.sc2_env.reset()  # returns (obs, info)
            # pysc2 kan returnere (obs, info)
            if isinstance(obs, tuple) and hasattr(obs[0], 'observation'):
                self._last_obs = obs[0]
            elif isinstance(obs, list) and hasattr(obs[0], 'observation'):
                self._last_obs = obs[0]
            else:
                self._last_obs = obs  # fallback
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
            # pysc2 kan returnere (obs, info)
            if isinstance(obs, tuple) and hasattr(obs[0], 'observation'):
                self._last_obs = obs[0]
            elif isinstance(obs, list) and hasattr(obs[0], 'observation'):
                self._last_obs = obs[0]
            else:
                self._last_obs = obs  # fallback

        

        # Reset building tracking
        completed_buildings = self.get_building_state_feature_units(obs[0])[1]
        self.existing_buildings = set(unit.tag for unit in completed_buildings)


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
        self._prev_vespene = 0.0
        self._prev_army_power = 0.0
        self._prev_idle_production_time = 0.0
        #self.prev_completed = 0
        # Reset first step
        self.first_step = True

        # Reset Command Center flight timer
        self.cc_lift_timer = 0

        # reset last build progress
        self.last_build_progress = {}
        
        #Reset lift attempts of Command center
        self.num_lift_attempts = 0

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


    def is_command_center_flying(self, obs):
        feature_units = obs.observation.feature_units # Numpy array

        if feature_units is None or len(feature_units) == 0:
            return False
                                        
        for unit in feature_units:
            unit_type = int(unit[0]) # Column 0 : unit Type
            alliance = int(unit[1]) # Column 1: alliance
            x, y = unit[2], unit[3]
            if alliance != 1:
                continue

            #Detect lifting or already airborne CC
            if unit_type  in [units.Terran.CommandCenterFlying, units.Terran.OrbitalCommandFlying]:
                return True
            # Check y>0 as proxy for lifted CC
            if unit_type in [units.Terran.CommandCenter, units.Terran.OrbitalCommand] and y > 0:
                return True
            # Some versions encode 'is_flying' or 'height' info
            if len(unit) > 8 and unit[8] > 0:  # e.g. height or flying flag
                print("height or flying flag")
                return True
            
        return False

    def get_building_state_feature_units(self, obs):
        under_construction = []
        completed = []
        feature_units = obs.observation.feature_units

        # Safety check
        if feature_units is None or len(feature_units) == 0:
            return under_construction, completed

        for unit in feature_units:
            unit_type = int(unit.unit_type)
            alliance = int(unit.alliance)


            if alliance != 1:
                continue
            if not self.is_building_type(unit_type):
                continue

            # Prefer build_progress if available
            build_progress = getattr(unit, "build_progress", None)
            if build_progress is not None:
                if build_progress < 100:
                    under_construction.append(unit)
                else:
                    completed.append(unit)

        return under_construction, completed


    def is_building_type(self, unit_type: int) -> bool:
        # Simplified Terran set; you can expand this
        terran_buildings = {
            # Base
            units.Terran.CommandCenter,
            units.Terran.OrbitalCommand,
            units.Terran.PlanetaryFortress,

            # Production
            units.Terran.Barracks,
            units.Terran.Factory,
            units.Terran.Starport,

            # Addons
            units.Terran.BarracksTechLab,
            units.Terran.BarracksReactor,
            units.Terran.FactoryTechLab,
            units.Terran.FactoryReactor,
            units.Terran.StarportTechLab,
            units.Terran.StarportReactor,

            # Resource
            units.Terran.Refinery,
            units.Terran.SupplyDepot,
            units.Terran.SupplyDepotLowered,

            # Tech
            units.Terran.EngineeringBay,
            units.Terran.Armory,
            units.Terran.FusionCore,
            units.Terran.GhostAcademy,

            # Defense
            units.Terran.Bunker,
            units.Terran.MissileTurret,
            units.Terran.SensorTower,


        }
        return unit_type in terran_buildings

    
    def step(self, action):
        func_index, x, y = int(action[0]), float(action[1]), float(action[2])
        func_index = max(0, min(func_index, len(self.valid_funcs) - 1))
        func_id = self.valid_funcs[func_index]

        # --- Build a safe action mask ---
        try:
            available = self._last_obs.observation.get("available_actions", [])
            if not isinstance(available, (list, np.ndarray)):
                available = []

        except Exception as e:
            print(f"⚠️ Exception with available: {e}")
        
                

        # --- Safe action execution ---
        if func_id not in available:
            func = sc2_actions.FUNCTIONS.no_op
            args = []
            self.invalid_action_count += 1
            func_def = sc2_actions.FUNCTIONS[func_id]
            #print(f"Invalid action {func_id}: {func_def.name} chosen")
        else:
            func = sc2_actions.FUNCTIONS[func_id]
            #print("Valid action:", func.name)
            self.valid_action_count += 1
            args = []
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

        # --- Detect repeated actions ---
        action_signature = (func.id, tuple(map(tuple, args)))  # hashable
        self.recent_actions.append(action_signature)

        # Count how many times the last action has been repeated
        if len(self.recent_actions) >= 2 and all(a == self.recent_actions[-1] for a in list(self.recent_actions)[-10:]):
            self.repeated_action_count += 1
            repeat_penalty = -0.05 * self.repeated_action_count  # scale penalty with frequency
        else:
            self.repeated_action_count = 0
            repeat_penalty = 0.0


        if repeat_penalty < 0:
            print(f"⚠️ Over 10 Repeated actions detected ({self.repeated_action_count+10}x) | Penalty: {repeat_penalty:.3f}")

        
        obs = self.sc2_env.step([sc2_action])[0]
        self._last_obs = obs[0] if isinstance(obs, list) else obs

        reward = float(obs.reward)

       
        cc_flying = self.is_command_center_flying(obs)

        # --- Track fligh duration --- #
        if cc_flying:
            self.cc_lift_timer += 1
            #print(f"Agent is currently flying Command center for {self.cc_lift_timer} steps")
        else:
            self.cc_lift_timer = 0 # Reset when landed
     
            
        # --- Cumulative metrics ---
        # --- score_cumulative array number values: 
        # --- 0='score', 1='idle_production_time', 2='idle_worker_time', 3='total_value_units',
        # --- 4='total_value_structures', 5='...', 6='collected_vespene', 7='collection_rate_minerals', 8='collection_rate_vespene', 9='spent_minerals', 10='spent_vespene'
        # ---
        # --- Player contains = ['player_id', 'minerals', 'vespene', 'food_used', 'food_cap',
        # --- 'food_army', 'food_workers', 'idle_worker_count', 'army_count', 'warp_gate_count', 'larva_count'] 
        try:
            score_cumulative = obs.observation["score_cumulative"]
            player_obs = obs.observation["player"]
            idle_production_time = score_cumulative[1]
            score = score_cumulative[0]
            gathered = score_cumulative[7]
            vespene = score_cumulative[8]
            workers = player_obs[6]
            idle_workers = player_obs[7]
            supply_despot = player_obs[4]
            army_power = player_obs[5]
        except (KeyError, AttributeError, TypeError):
            print("KeyError, AttributeError or TypeError")
            score = gathered = vespene = army_power = workers = idle_workers = supply_despot =idle_production_time = 0.0

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
            delta_vespene = vespene - self._prev_vespene
            delta_army_power = army_power - self._prev_army_power
            delta_idle_production_time = idle_production_time - self._prev_idle_production_time
        else:
            delta_score = delta_minerals = delta_vespene = delta_army_power = delta_workers = delta_idle_workers = delta_supply_despot = delta_idle_production_time = 0

        #print("delta minerals:", delta_minerals)
        #print("delta score:", delta_score)
        
        first_time_reward = 0
        #Reward the first time it builds a supply despot or if episode reward is above 20
        if self._first_time == 2:
            if delta_supply_despot > 0 or delta_workers > 0:
                self._first_time = 3
                first_time_reward = 2.0
                print(f"Congratulations! You did something good for the first time! Rewarded {first_time_reward} extra points")

        # Set first time +1 ONLY if it is the first time it steps to avoid rewards being given for free because prevs are 0
        if self._first_time == 0 or self._first_time == 1 :
            self._first_time += 1
        
        
        self._prev_score = score
        #self._prev_kills = killed
        self._prev_minerals = gathered
        self._prev_workers = workers
        self._prev_idle_workers = idle_workers
        self._prev_supply_despot = supply_despot
        self._prev_vespene = vespene
        self._prev_army_power = army_power
        self._prev_idle_production_time = idle_production_time

        # --- Normalization ---
        # Cap large jumps to keep PPO stable
        delta_score = np.clip(delta_score, -100, 100) / 100.0 # General Score
        #delta_kills = np.clip(delta_kills, -50, 50) / 50.0 # Kills (not yet found)
        delta_minerals = np.clip(delta_minerals, -500, 500) / 500.0 # Minerals gathered
        delta_workers = np.clip(delta_workers, -5, 5) / 5.0 # Power of workers 
        delta_idle_workers = np.clip(delta_idle_workers, 0, 20) / 20.0  # only penalize positive idle change
        delta_supply_despot = np.clip(delta_supply_despot, -8, 8) / 8.0 # Supply despots nr
        delta_vespene = np.clip(delta_vespene, -500, 500) / 500.0 # Vespene gathered
        delta_army_power = np.clip(delta_army_power, -5, 5) / 5.0 # Army power
        delta_idle_production_time = np.clip(delta_idle_production_time, -1, 1) / 1.0 #Idle production time

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
            + 0.15 * delta_army_power #reward for having supply used for army units  
            + 0.05 * delta_vespene
            + 0.05 * delta_minerals
            + 0.05 * delta_workers  # reward for training new workers
            + 0.1 * delta_supply_despot
            - 0.05 * delta_idle_production_time 
            - 0.2 * delta_idle_workers  # penalize idle workers
        )
        # Add penalty from repeated actions
        shaped_reward += repeat_penalty
        # Add first time reward
        shaped_reward += first_time_reward

        # Construction rewards
        under_construction, completed = self.get_building_state_feature_units(obs)

        if len(under_construction) > 0:
            print(f"{len(under_construction)} Terran buildings are currently being built.")

        if self.first_step:
            self.existing_buildings = set(unit.tag for unit in completed)
            self.first_step = False

        for unit in completed:
            if unit.tag not in self.existing_buildings:
                # Get the unit type ID
                unit_id = unit.unit_type


                # Reverse lookup in units.Terran
                unit_name = None
                for name, val in units.Terran.__dict__.items():
                    # Only look at ints (unit IDs)
                    if not isinstance(val, int):
                        continue

                    if int(val) == int(unit_id):
                        unit_name = name
                        break
                
                if unit_name is None:
                    unit_name = f"Unknown({unit_id})"

                print(f"{unit_name} was just built finished")    
                shaped_reward += 10  # reward for finishing a new building
                print("rewarded: 10 points for completing a new building")
                self.existing_buildings.add(unit.tag)    


        # Store last progress in the env
        if not hasattr(self, "last_build_progress"):
            self.last_build_progress = {}   
        # Loop through all buildings under construction and give rewards and prints
        for b in under_construction:
            uid = b.tag  # unique tag for each unit
            last_prog = self.last_build_progress.get(uid, 0)
            delta_prog = b.build_progress - last_prog
            self.last_build_progress[uid] = b.build_progress
            
            shaped_reward += 0.1 * delta_prog  # reward only for new progress
            #print(f"delta_prog is: {delta_prog}, reward for building progress is:", 0.1 * delta_prog)
            
            # Get the unit type ID
            unit_id = b.unit_type


            # Reverse lookup in units.Terran
            unit_name = None
            for name, val in units.Terran.__dict__.items():
                # Only look at ints (unit IDs)
                if not isinstance(val, int):
                    continue

                if int(val) == int(unit_id):
                    unit_name = name
                    break
            
            if unit_name is None:
                unit_name = f"Unknown({unit_id})"
                
            #print(f"⛏️ {unit_name} ({b.unit_type}) at ({b.x}, {b.y}) is {b.build_progress}% done")  

        # --- Penalize if flying too long --- #
        FLY_TRESHOLD = 15
        if self.cc_lift_timer > FLY_TRESHOLD:
            #print(f" Agent flown Command Center for too long {self.cc_lift_timer} long.. Treshold is {FLY_TRESHOLD}")
            #print(f"Penalized for :", 1.0 * (self.cc_lift_timer - FLY_TRESHOLD) / 50.0, " points") 
            shaped_reward -= 0.5 * (self.cc_lift_timer - FLY_TRESHOLD) / 50.0 #Scale penalty

        was_flying = self.command_center_flying
        is_flying_now = self.is_command_center_flying(obs)

        # detect lift-off transition
        if not was_flying and is_flying_now:
            self.num_lift_attempts += 1
            if self.num_lift_attempts > 2:
                shaped_reward -= 0.5  # large penalty per lift-off attempt
                #print(f"New attempt of flying detected, this is {self.num_lift_attempts} times attempted, and gets penalized after trying for more than 2 times")

        # update state
        self.command_center_flying = is_flying_now    

        
        # track episode total
        self.episode_reward += shaped_reward
        
        done = obs.last()
        info = {}

        # Increment episode step counter
        self.episode_steps += 1

        # split done into terminated / truncated
        terminated = done
        truncated = False  # or implement a max-steps limit

        if done or truncated or terminated:

            if self.episode_reward > 20.0:
                print(f"Congratulations you got rewarded an extra 5.0 bonus points for reaching 20.0 reward points")
                shaped_reward += 5.0 # Use to reward if the episode reward is above 20.0 (Above the average)
                self.episode_reward += 5.0
            
            # highest score
            if self.episode_reward > self._prev_high_score:
                top1_score = self.episode_reward
                self._prev_high_score = top1_score
                print(f"Congratulations... You reached a new high score: {top1_score}")

            print(f"Episode finished. Reward/result: {reward}, Valid actions: {self.valid_action_count}, invalid actions {self.invalid_action_count}, episode reward: {self.episode_reward}, Total steps: {self.episode_steps}")
            # Reset episode stats
            self.valid_action_count = 0
            self.episode_steps = 0
            self.episode_reward = 0
            self.invalid_action_count = 0
    

        
        return self._process_obs(obs), shaped_reward, terminated, truncated, info
        

    def close(self):
        self.sc2_env.close()


# -------- Masking function ---------------#

def mask_fn(env):
    obs = getattr(env, "_last_obs", None)

    if obs is None:
        print("⚠️ mask_fn: _last_obs is None, using full mask")
        # fallback: alle funksjoner + xy-posisjoner tillatt
        return np.ones(len(env.valid_funcs) + 128, dtype=np.float32)

    # Sjekk at obs har 'observation'
    if not hasattr(obs, "observation"):
        print(f"⚠️ mask_fn: _last_obs has no 'observation'. Type: {type(obs)}")
        print(f"_last_obs content: {obs}")
        return np.ones(len(env.valid_funcs) + 128, dtype=np.float32)

    # Hent tilgjengelige funksjoner
    try:
        available = obs.observation.get("available_actions", [])
    except Exception as e:
        print(f"⚠️ mask_fn: failed to access available_actions: {e}")
        available = []

    # Lag funksjons-mask
    func_mask = np.zeros(len(env.valid_funcs), dtype=np.float32)
    for idx, fid in enumerate(env.valid_funcs):
        if fid in available:
            func_mask[idx] = 1.0

    # xy-mask (alltid tillatt)
    xy_mask = np.ones(128, dtype=np.float32)
    full_mask = np.concatenate([func_mask, xy_mask])

    # fallback hvis ingen funksjoner er gyldige
    if not np.any(func_mask):
        print("⚠️ mask_fn: no valid functions, fallback to full mask")
        full_mask[:] = 1.0

    return full_mask








# ---------------- Environment constructor ----------------
def make_env(mask_fn):
    def _init():
        env = SC2ContinuousCNNEnv()
        env = ActionMasker(env, mask_fn)
        #env = CurriculumWrapper(env)
        env = Monitor(env)
        return env
    return _init

# ------------------ count params -----------------------
def count_all_params(model):
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.policy.parameters() if not p.requires_grad)
    return trainable, non_trainable

if __name__ == "__main__":
    env_fns = [make_env(mask_fn) for i in range(NUM_ENVS)]  # call make_env
    env = SubprocVecEnv(env_fns)

    # ✅ Viktig: reset før trening starter
    obs = env.reset()

    print(f"Created {NUM_ENVS} SC2 environments in parallel.")

    save_callback = SaveEveryNStepsCallback(save_freq=10_000, save_path="./")

    shaped_callback = ShapedRewardCallback()

    curr_callback = CurriculumCallback(env, target_rate=TARGET_RATE, increase_steps=INC_STEPS)
    callback = CallbackList([save_callback, shaped_callback])

    policy_kwargs = dict(
    features_extractor_class=CustomDeepCNN,
    features_extractor_kwargs=dict(features_dim=2048),
    net_arch=dict(pi=[2048, 1024, 512], vf=[2048, 1024, 512]),)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = MaskablePPO.load(MODEL_PATH, env)
    else:
        print("Found no model. Creating new model.")
        model = MaskablePPO("CnnPolicy", env,
                            policy_kwargs=policy_kwargs,
                            learning_rate=0.00015,
                            ent_coef=0.01,
                            n_steps=1024,
                            batch_size=256,
                            gamma=0.99,
                            gae_lambda=0.95,
                            max_grad_norm=0.5,
                            clip_range=0.2,
                            verbose=1,
                            device="cuda",
                            tensorboard_log="./customCNN_ppo_x10_logs/")

        
    trainable, non_trainable = count_all_params(model)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")
    print(f"Total parameters: {trainable + non_trainable:,}")
    #model.batch_size = 4096
    #model.n_steps = 256
    #model.ent_coef=0.01
    #model.learning_rate=0.0001  <<---- does not work
    #model.lr_schedule = lambda _: 0.0001 <---- this works
    print("Batch size: ", model.batch_size)
    print("Learning rate: ", model.learning_rate)
    print("ent_coef:", model.ent_coef)
    print("num steps:", model.n_steps)
    print("Epochs each rollout:", model.n_epochs)
    print("Using device:", model.device)
    print("CUDA available:", th.cuda.is_available())
    print(f"Training MaskablePPO on {MAP_NAME} for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save(MODEL_PATH)
    print("Training complete. Model saved to", MODEL_PATH)
    env.close()
