"""
PySC2 4.0.0 headless example using RandomAgent
Runs MoveToBeacon map in headless mode on Windows.
"""

from absl import flags
FLAGS = flags.FLAGS
FLAGS(['random_agent.py'])

from pysc2.env import sc2_env
from pysc2.agents import random_agent
from pysc2.lib import actions

MAP_NAME = "MoveToBeacon"
FEATURE_SCREEN_SIZE = 84
STEP_MUL = 8

def main():
    # Create the SC2 environment
    env = sc2_env.SC2Env(
        map_name=MAP_NAME,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(
                screen=FEATURE_SCREEN_SIZE,
                minimap=FEATURE_SCREEN_SIZE
            ),
            use_feature_units=True
        ),
        step_mul=STEP_MUL,
        visualize=False,  # Headless
        realtime=False
    )

    agent = random_agent.RandomAgent()  # Prebuilt random agent

    try:
        obs = env.reset()
        done = False
        episode = 0

        while episode < 5:  # Run 5 episodes
            step_count = 0
            obs = env.reset()
            done = False
            print(f"Starting episode {episode + 1}")

            while not done:
                action = agent.step(obs[0])
                obs = env.step([action])
                done = obs[0].step_type == sc2_env.StepType.LAST
                step_count += 1

            print(f"Episode {episode + 1} finished in {step_count} steps")
            episode += 1

    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
