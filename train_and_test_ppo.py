"""Finding the best set of hyperparameters in the source_env->source_env configuration, using a wandb random sweep for 30 runs.


The model with the best hyperparameters is trained for 350k timesteps (three seeds) and saved as "best_model_ppo_ss", 
because it is trained and tested on the source environment. 

"""

import gym
from env.custom_hopper import *

import pprint
from functools import partial

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def main():

    source_env = gym.make('CustomHopper-source-v0')
    source_env = Monitor(source_env)
        #No need for the target env since the hp are found just in the source environment, the only one we actually know about
    #target_env = gym.make('CustomHopper-target-v0')
    #target_env = Monitor(target_env)

    
    best_parameters = {
        'best_mean_reward' : 0,
        'best_std_reward' : 0,
        'lr' : 0,
        'n_steps' : 0,
        'gamma' : 0,
        'batch_size' : 0,
        'n_epochs' : 0,
        'gl' : 0
    }

    #https://arxiv.org/pdf/1707.06347

    sweep_config = {
        "method" : "random",

        #goal not necessary if method is not bayesian, but still useful
        "metric" : {
            "name" : "mean_mean_reward",
            "goal" : 'maximize'
        },
        "parameters" : {
            "learning_rate" : {"min" : 0.00003, "max" : 0.003}, #default 0.0003
            #"n_steps" : {"min": 1024, "max": 3072}, #default 2048 - In order not to have truncated batches, 
            # batch_size must be a factor of n_steps * n_envs. Originally, since n_envs =1, n_steps = batch_size*32
            "gamma" : {"min" : 0.9, "max" : 0.9999}, #default 0.99
            "batch_size" : {"min" : 32, "max" : 128}, #default 64
            "n_epochs" : {"min": 5, "max": 20}, #default 10
            "gae_lambda" : {"min": 0.85, "max": 0.999} #default 0.95
        }
    }


    sweep_id_ss = wandb.sweep(sweep_config, project="ppo_sweep_ss__5")

    def train_and_evaluate(train_env, test_env) :
        with wandb.init(config=sweep_config):
            
            config = wandb.config

            lr = config.learning_rate
            gamma = config.gamma
            bs = round(config.batch_size)
            nsteps = bs * 32
            nepochs = round(config.n_epochs)
            gl = config.gae_lambda

            print("learning_rate: ", lr)
            print("nsteps: ", nsteps)
            print("gamma: ", gamma)
            print("batch size: ", bs)
            print("n_epochs: ", nepochs)
            print("gae_lambda: ", gl)

            #train on the desired environment (source or target)
            model_1 = PPO("MlpPolicy", train_env, learning_rate= lr, n_steps=nsteps, gamma=gamma, batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=0, verbose=2)
            model_1.learn(total_timesteps=350000)
            #evaluate on the desired environment (source or target)
            mean_reward_1, std_reward_1 = evaluate_policy(model_1, test_env, n_eval_episodes=50, deterministic=True)

            model_2 = PPO("MlpPolicy", train_env, learning_rate=lr, n_steps=nsteps, gamma=gamma, batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=14, verbose=2)
            model_2.learn(total_timesteps=350000)
            mean_reward_2, std_reward_2 = evaluate_policy(model_2, test_env, n_eval_episodes=50, deterministic=True)

            model_3 = PPO("MlpPolicy", train_env, learning_rate=lr, n_steps=nsteps, gamma=gamma, batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=42, verbose=2)
            model_3.learn(total_timesteps=350000)
            mean_reward_3, std_reward_3 = evaluate_policy(model_3, test_env, n_eval_episodes=50, deterministic=True)

            print(f"Mean reward 1: {mean_reward_1}, Mean reward 2: {mean_reward_2}, Mean reward 3: {mean_reward_3}")

            mean_mean_reward = (mean_reward_1 + mean_reward_2 + mean_reward_3)/3
            mean_std_reward = (std_reward_1 + std_reward_2 + std_reward_3)/3

            #ss
            if mean_mean_reward > best_parameters['best_mean_reward']:
                best_parameters['best_mean_reward'] = mean_mean_reward
                best_parameters['best_std_reward'] = mean_std_reward

                models = [(model_1, mean_reward_1), (model_2, mean_reward_2), (model_3, mean_reward_3)]
                    #Find the model with the maximum mean reward to save
                best_model, best_mean_reward = max(models, key=lambda item: item[1])
                best_model.save("./models/best_model_ppo_ss_350")

                best_parameters['lr'] = lr
                best_parameters['nsteps'] = nsteps
                best_parameters['gamma'] = gamma
                best_parameters['batch_size'] = bs
                best_parameters['n_epochs'] = nepochs
                best_parameters['gl'] = gl

            wandb.log({"mean_mean_reward": mean_mean_reward})

    #This is necessary to pass arguments to the function in wandb.agent for a sweep,
    #because wandb.agent does not accept functions with arguments
    p_train_and_evaluate = partial(train_and_evaluate, source_env, source_env)

    #source->source
    wandb.agent(sweep_id_ss, p_train_and_evaluate, count=30) #SWEEP
    print("Best mean reward: ", best_parameters['best_mean_reward'])
    print("Best std reward: ", best_parameters['best_std_reward'])
    print("Best hyperparameters for the model: ", best_parameters)

    """ Best mean reward:  1406.35123122
    Best std reward:  84.98911874395765
    Best hyperparameters for the model:  {'best_mean_reward': 1406.35123122, 'best_std_reward': 84.98911874395765,
     'lr': 0.0012541462101916157, 'n_steps': 0, 'gamma': 0.9885202328222382,
      'batch_size': 123, 'n_epochs': 11, 'gl': 0.9472233613269306, 'nsteps': 3936} """

if __name__ == '__main__':
    main()