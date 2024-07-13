"""
    Train a model in the target environment with the same hyperparameters found in the source -> source configuration.
    Evaluate the model on the target environment every 50 episode
"""

import gym
from env.custom_hopper import *

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def main():
    target_env = gym.make('CustomHopper-target-v0')
    target_env = Monitor(target_env)

    # best hyperparameter found in the source --> source configuration trained on 350K timesteps
    best_hp = {
        'lr': 0.0012541462101916157,
        'gamma': 0.9885202328222382,
        'batch_size': 123,
        'n_epochs': 11,
        'gl': 0.9472233613269306,
        'nsteps': 3936}
    
    # train and evaluate three model with different seeds [0, 14, 42]
    tt_model_1 = PPO("MlpPolicy", target_env, learning_rate=best_hp['lr'], n_steps=best_hp['nsteps'], gamma=best_hp['gamma'], batch_size=best_hp['batch_size'], n_epochs=best_hp['n_epochs'], gae_lambda=best_hp['gl'], seed=0, verbose=2)
    tt_model_1.learn(total_timesteps = 1_000_000)
    mean_reward_1, std_reward_1 = evaluate_policy(tt_model_1, target_env, n_eval_episodes=50, deterministic=True)

    tt_model_2 = PPO("MlpPolicy", target_env, learning_rate=best_hp['lr'], n_steps=best_hp['nsteps'], gamma=best_hp['gamma'], batch_size=best_hp['batch_size'], n_epochs=best_hp['n_epochs'], gae_lambda=best_hp['gl'], seed=14, verbose=2)
    tt_model_2.learn(total_timesteps=1_000_000)
    mean_reward_2, std_reward_2 = evaluate_policy(tt_model_2, target_env, n_eval_episodes=50, deterministic=True)

    tt_model_3 = PPO("MlpPolicy", target_env, learning_rate=best_hp['lr'], n_steps=best_hp['nsteps'], gamma=best_hp['gamma'], batch_size=best_hp['batch_size'], n_epochs=best_hp['n_epochs'], gae_lambda=best_hp['gl'], seed=42, verbose=2)
    tt_model_3.learn(total_timesteps=1_000_000)
    mean_reward_3, std_reward_3 = evaluate_policy(tt_model_3, target_env, n_eval_episodes=50, deterministic=True)
    
    mean_mean_reward = (mean_reward_1 + mean_reward_2 + mean_reward_3)/3
    mean_std_reward = (std_reward_1 + std_reward_2 + std_reward_3)/3

    print("Mean reward: ", mean_mean_reward)
    print("Mean std: ", mean_std_reward)

    """ 
        1M
        Mean reward:  1175.9516913466666     # This is UDR Upper Bound
        Mean std:  201.58074850355388 
    """

    #save best model wrt the mean_reward
    models = [(tt_model_1, mean_reward_1), (tt_model_2, mean_reward_2), (tt_model_3, mean_reward_3)]
    best_model, best_mean_reward = max(models, key=lambda item: item[1])
    best_model.save("./models/best_model_ppo_tt")

if __name__ == '__main__':
    main()