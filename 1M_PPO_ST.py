""" 
    Evaluate in the target the best model trained in the source environment for 1M
"""

import gym
from env.custom_hopper import *

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import wandb

def main():

    target_env = gym.make('CustomHopper-target-v0')
    target_env = Monitor(target_env)

    #the model trained in the source environment, with the best set of hp found on the source->source transfer
    model = PPO.load("./models/best_model_ppo_ss")

    # evaluate on target environment every 50 episode
    mean_reward_1, std_reward_1 = evaluate_policy(model, target_env, n_eval_episodes=50, deterministic=True)
    mean_reward_2, std_reward_2 = evaluate_policy(model, target_env, n_eval_episodes=50, deterministic=True)
    mean_reward_3, std_reward_3 = evaluate_policy(model, target_env, n_eval_episodes=50, deterministic=True)

    mean_mean_reward = (mean_reward_1 + mean_reward_2 + mean_reward_3)/3
    mean_std_reward = (std_reward_1 + std_reward_2 + std_reward_3)/3

    print("Mean reward: ", mean_mean_reward) 
    print("Mean std: ", mean_std_reward) 

    """ 1M
        Mean reward:  813.4313615333332 # This is the UDR Lower Bound
        Mean std:  125.013457543662685 
    """

if __name__ == '__main__':
    main()