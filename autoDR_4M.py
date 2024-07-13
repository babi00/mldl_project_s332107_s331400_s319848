"""Testing what happens by training the model for 4M timesteps instead of 10M, since at 4M the mean reward starts dropping.
In the end, 4M was better than 10M, possible motivations are described in the paper."""

import gym
from env.custom_hopper import *

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import matplotlib.pyplot as plt

import wandb

import numpy as np
import random

from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback

from timeit import default_timer as timer


class AutoDRCallback(BaseCallback):

    def __init__(self, delta, threshold_high, seed_number, verbose: int = 0):
        super().__init__(verbose)

        self.original_masses = None

        #never used if using "closed distribution" approach
        self.boundary_sampling_probabilty = 0.5

        self.buffer_size = 30 
        self.threshold_H = threshold_high
        self.threshold_L = threshold_high / 2
        
        self.initial_range = 0.5 #initially, use 50% as the initial range for the masses distributions

        #0: thigh, 1: leg, 2: foot
        self.buffers = {
            0 : {'low' : [], 'high' : []},
            1 : {'low': [], 'high' : []},
            2 : {'low' : [], 'high' : []}
        }
        self.bounds = {}
        self.deltas = {}
        self.delta = delta

        #ONLY USED FOR WANDB LOGGING
        self.seed_number = seed_number
        #self.num_episodes = 0
        self.tot_reward = []

        
    def _on_training_start(self) -> None: # before the first rollout
        print("--Starting the rollout--")

        self.original_masses = self.training_env.get_attr("env")[0].get_original_parameters()[1:] #Returns everything except "world", here I exclude "torso"

        #50% RANGE APPROACH
        #comment out if using "closed distribution" approach
        self.bounds = {
            0 : [self.original_masses[0] - self.original_masses[0]*self.initial_range, self.original_masses[0] + self.original_masses[0]*self.initial_range ],
            1 : [self.original_masses[1] - self.original_masses[1]*self.initial_range, self.original_masses[1] + self.original_masses[1]*self.initial_range ],
            2 : [self.original_masses[2] - self.original_masses[2]*self.initial_range, self.original_masses[2] + self.original_masses[2]*self.initial_range ]
        }

        #CLOSED DISTRIBUTION APPROACH
        # set initial bounds at 0 - comment out if using "50% range" approach
        """ for i in range(3):
            starting_distance = (self.deltas[i] - 1e-10)/2
            bounds = [self.original_masses[i]-starting_distance, self.original_masses[i]+starting_distance]
            self.bounds[i] = bounds """

        self.deltas = {
            0: self.original_masses[0] * self.delta,
            1: self.original_masses[1] * self.delta,
            2: self.original_masses[2] * self.delta
        }

        print('Dynamics parameters:', self.training_env.get_attr("env")[0].get_parameters())

        # Initialize wandb logging
        wandb.init(project="auto_dr_testing_4M_only_model_0", config={ #CHANGE NAME OF THE PROJECT, auto_dr_testing_4M_only_model_0 IS ONLY FOR MODEL 0
            "delta": self.deltas,
            "threshold_H": self.threshold_H,
            "threshold_L": self.threshold_L,
            "initial_range": self.initial_range
        })
        wandb.run.name = "AutoDR_Run_" + str(f"delta_{self.delta}_threshold_{self.threshold_H}_seed_{self.seed_number}")
        wandb.run.save() 

    def _on_step(self) -> bool: # after each env.step()
        if any(self.locals['dones']): #array of size 1, False if episode not done, True if last step of the episode
            #self.num_episodes += 1 #keping count of number of episodes for wandb logging of mean reward
            self.tot_reward.append(self.locals['infos'][0]['episode']['r']) #keeping track of total reward for mean reward

            self.training_env.get_attr("env")[0].reset()  # Reset the environment before modification

            if np.random.uniform(0,1) < self.boundary_sampling_probabilty: #at the end of each episode have the same prob to launch AutoDR or keep training
                self.autoDR()
            else:
                self.training_env.get_attr("env")[0].sample_parameters_autoDR(self.bounds) #execute normal Domain Randomization

            # Log the bounds and entropy at the end of each episode
            bounds_log = {f"bounds_param_{i}": self.bounds[i] for i in self.bounds}
            entropy = self.compute_entropy()
            #CHECK SELF.LOCALS IF CURIOUS TO WHY EPISODE REWARDS ARE LOGGED LIKE THAT
            wandb.log({"entropy": entropy, "step": self.num_timesteps, "mean_reward" : np.mean(self.tot_reward), "std reward" : np.std(self.tot_reward)})
            for i in range(len(self.bounds)):
                wandb.log({f"bounds_param_{i}[0]": self.bounds[i][0], f"bounds_param_{i}[1]": self.bounds[i][1], "step": self.num_timesteps})
        
        return True

    def autoDR(self):
        param_id = np.random.randint(0, len(self.bounds)) #select a random parameter to fix, the others are sampled from their uniform distribution
        
        if np.random.uniform(0,1) < 0.5: #select lower bound to fix
            fixed_bound = self.bounds[param_id][0]
            buffer_name = 'low' 
        else: #select upper bound to fix
            fixed_bound = self.bounds[param_id][1]
            buffer_name = 'high'

        #new method implemented in the custom_hopper.py to sample parameters according to AutoDR rules, fixing one bound and sampling the others
        self.training_env.get_attr("env")[0].sample_parameters_autoDR(self.bounds, param_id, fixed_bound) #execute domain randomization with one fixed bound

        #evaluate policy with the current environment mass on the source environment for one episode
        episode_reward, _ = evaluate_policy(self.model, self.training_env.get_attr("env")[0], n_eval_episodes=1, deterministic=True)

        self.buffers[param_id][buffer_name].append(episode_reward) #append the episode reward to the buffer associated to the chosen parameter and bound (low or high)

        #if the buffer is full (>= chosen size) find the average of the performances and empty the buffer associated to the selected parameters and bound 
        if len(self.buffers[param_id][buffer_name]) >= self.buffer_size :
            avg_reward = np.mean(self.buffers[param_id][buffer_name])
            self.buffers[param_id][buffer_name].clear()

            if avg_reward >= self.threshold_H :
                self.increase_entropy(param_id)
            elif avg_reward <= self.threshold_L :
                self.decrease_entropy(param_id)
            #else don't do anything and keep on

    def increase_entropy(self, param_id):
        bounds_distance = self.bounds[param_id][1] - self.bounds[param_id][0]

        if self.bounds[param_id][0] - self.deltas[param_id]/2 > 0: #masses cannot become negative: if the step makes the bounds < 0, do nothing
            self.bounds[param_id][0] -=  (self.deltas[param_id])/2
            self.bounds[param_id][1] +=  (self.deltas[param_id])/2   

    def decrease_entropy(self, param_id):
        bounds_distance = self.bounds[param_id][1] - self.bounds[param_id][0]

        if bounds_distance > self.deltas[param_id]: #don't decrease the distance of the bounds if it is lower than the step size or they might cross (low becomes higher than high)
            self.bounds[param_id][0] +=  (self.deltas[param_id])/2
            self.bounds[param_id][1] -=  (self.deltas[param_id])/2   

    def compute_entropy(self):
        tot_entropy = 0
        #print("computing entropy")
        #print("BOUNDS: ", self.bounds)
        for i in self.bounds:
            tot_entropy += np.log(self.bounds[i][1] - self.bounds[i][0])

        return tot_entropy / len(self.bounds)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print("--Training has ended--")
        wandb.finish()

def main():
    """ TO TUNE
        delta: percentage of the original mass to increase or decrease the distribution bounds [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        threshold_high: upper threshold to decide if increase or decrease entropy (lower=high/2) [800, 1000, 1200]
     """
    source_env = gym.make('CustomHopper-source-v0')
    source_env = Monitor(source_env)
    target_env = gym.make('CustomHopper-target-v0')
    target_env = Monitor(target_env)

    total_timesteps = 4_000_000

    # Hyperparameters to tune
    #delta_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    delta_values = [0.02]
    threshold_high_values = [1000]

    #PPO hyperparameters found previously
    lr = 0.0012541462101916157
    gamma = 0.9885202328222382
    bs = 123
    nsteps = 3936
    nepochs = 10
    gl = 0.95

    best = {
        'mean_reward' : 0,
        'mean_std' : 0,
        'delta' : 0,
        'threshold_high' : 0
    }


    for delta in delta_values:
        for threshold_high in threshold_high_values:

            start = timer()

            model_1 = PPO("MlpPolicy", source_env, learning_rate= lr, n_steps=nsteps, gamma=gamma, batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=0, verbose=2)
            model_1.learn(total_timesteps=total_timesteps, callback=AutoDRCallback(verbose=2, delta=delta, threshold_high=threshold_high, seed_number = 0))
            model_1.save('./models/model_1_autoDR_4M')
            #model_1_autoDR_4M saved to check jumping 5/07/2024
            mean_reward_source_1, std_reward_source_1 = evaluate_policy(model_1, source_env, n_eval_episodes=50, deterministic=True)
            mean_reward_target_1, std_reward_target_1 = evaluate_policy(model_1, target_env, n_eval_episodes=50, deterministic=True)

            end = timer()


           
            model_2 = PPO("MlpPolicy", source_env, learning_rate= lr, n_steps=nsteps, gamma=gamma, batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=14, verbose=2)
            model_2.learn(total_timesteps=total_timesteps, callback=AutoDRCallback(verbose=2, delta=delta, threshold_high=threshold_high, seed_number = 14))
            model_2.save('./models/model_2_autoDR')
            mean_reward_source_2, std_reward_source_2 = evaluate_policy(model_2, source_env, n_eval_episodes=50, deterministic=True)
            mean_reward_target_2, std_reward_target_2 = evaluate_policy(model_2, target_env, n_eval_episodes=50, deterministic=True)



            run_name_general = f"AutoDR_Run_delta_{delta}_threshold_{threshold_high}"
            run_general = wandb.init(project="auto_dr_testing_4M_0", name=run_name_general, config={
                "delta": delta,
                "threshold_high": threshold_high,
                "learning_rate": lr,
                "n_steps": nsteps,
                "gamma": gamma,
                "batch_size": bs,
                "n_epochs": nepochs,
                "gae_lambda": gl,
                "total_timesteps": total_timesteps
            })

            model_3 = PPO("MlpPolicy", source_env, learning_rate= lr, n_steps=nsteps, gamma=gamma, batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=42, verbose=2, tensorboard_log=f"runs/{run_general.id}")
            model_3.learn(total_timesteps=total_timesteps, callback=AutoDRCallback(verbose=2, delta=delta, threshold_high=threshold_high, seed_number = 42))
            model_3.save('./models/model_3_autoDR')
            mean_reward_source_3, std_reward_source_3 = evaluate_policy(model_3, source_env, n_eval_episodes=50, deterministic=True)
            mean_reward_target_3, std_reward_target_3 = evaluate_policy(model_3, target_env, n_eval_episodes=50, deterministic=True)



            mean_reward_source = (mean_reward_source_1 + mean_reward_source_2 + mean_reward_source_3)/3
            std_reward_source = (std_reward_source_1 + std_reward_source_2 + std_reward_source_3)/3

            mean_reward_target = (mean_reward_target_1 + mean_reward_target_2 + mean_reward_target_3)/3
            std_reward_target = (std_reward_target_1 + std_reward_target_2 + std_reward_target_3)/3


            print("Mean reward (source)", mean_reward_source)
            print("Std reward (source)", std_reward_source)
            print("Mean reward (target)", mean_reward_target)
            print("Std reward (target)", std_reward_target)
 
            print(f"[Seed 0] Total training time for {total_timesteps} steps: {end - start:.3f} s")

            #print(f"Delta: {delta}, Threshold High: {threshold_high}, Mean Reward: {mean_reward_source}, Std Reward: {std_reward_source}")

            run_general.finish()
          

"""     BEST VALUES:  {'mean_reward': 1653.3039766799996, 'mean_std': 29.89373272772578, 'delta': 0.02, 'threshold_high': 1000} """
""" Mean reward (source) 716.84646092
Std reward (source) 211.6724280838583
Mean reward (target) 720.9195125266668
Std reward (target) 97.00263264665898
[Seed 0] Total training time for 10_000_000 steps: 9789.432 s """

""" Mean reward (source) 629.4014864199999
Std reward (source) 299.3423684274263
Mean reward (target) 700.4489773933334
Std reward (target) 172.66492893849622
[Seed 0] Total training time for 10000000 steps: 9842.367 s """

""" Mean reward (source) 908.6256327133333
Std reward (source) 241.59758273229338
Mean reward (target) 862.64664716
Std reward (target) 224.3330673533599
[Seed 0] Total training time for 4000000 steps: 3651.223 s """





if __name__ == '__main__':
    main()


#4/07/2024 changed the folder for saving models(typo...)