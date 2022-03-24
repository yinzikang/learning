from dm_control.locomotion import soccer
import numpy as np

team_size = 2
num_walkers = 2 * team_size
env = soccer.load(team_size=team_size, time_limit=45, walker_type=soccer.WalkerType.BOXHEAD)
# assert len(env.action_spec()) == num_walkers
# assert len(env.observation_spec()) == num_walkers
# # Reset and initialize the environment.
# timestep = env.reset()
# # Generates a random action according to the ‘action_spec‘.
# random_actions = [spec.generate_value() for spec in env.action_spec()]
# timestep = env.step(random_actions)
# # Check that timestep respects multi-agent action and observation convention.
# assert len(timestep.observation) == num_walkers
# assert len(timestep.reward) == num_walkers