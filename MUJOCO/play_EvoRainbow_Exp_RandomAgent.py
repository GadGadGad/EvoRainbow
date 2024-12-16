import numpy as np
import os
import random
import torch
import gym
import argparse
from EvoRainbow_Exp_core import mod_utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) ' +
                    '(Swimmer-v2) (Hopper-v2)', required=True, type=str)
parser.add_argument('-seed', help='Random seed to be used', type=int, default=7)
parser.add_argument('-render', help='Render gym episodes', action='store_true')
args = parser.parse_args()

# Define a RandomAgent class
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        # Return a random action
        return self.action_space.sample()

def evaluate(agent, env, trials=1, render=False):
    results = []
    for trial in range(trials):
        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = agent.select_action(np.array(state))

            # Simulate one step in environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state

        results.append(total_reward)

    print("Average Reward:", np.mean(results))

if __name__ == "__main__":
    # Create the gym environment and normalize actions
    env = utils.NormalizedActions(gym.make(args.env))

    # Seed everything
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create a RandomAgent
    agent = RandomAgent(env.action_space)

    # Evaluate the random agent
    evaluate(agent, env, render=args.render)
