"""
This file is used for tuning of hyperparameters (2M timesteps). 
They were tuned separately for each one of us (each component of the group tested 6 configurations)

"""


import gym
from env.custom_hopper import *

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import matplotlib.pyplot as plt

import wandb

import numpy as np
import random

from timeit import default_timer as timer

from stable_baselines3.common.callbacks import BaseCallback


class AutoDRCallback(BaseCallback):

    def __init__(self, verbose: int = 0, delta=0.02, threshold_high=800):
        super().__init__(verbose)

        self.original_masses = None

        self.boundary_sampling_probabilty = 0.5

        self.buffer_size = 30
        self.threshold_H = threshold_high
        self.threshold_L = threshold_high / 2
        
        #note that the tuning has been done only on the 50% range distribution, not on the "closed range" distribution.
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

        
    def _on_training_start(self) -> None: # before the first rollout
        print("--Starting the rollout--")

        self.original_masses = self.training_env.get_attr("env")[0].get_original_parameters()[1:] #Returns everything except "world", here I exclude "torso"

        self.bounds = {
            0 : [self.original_masses[0] - self.original_masses[0]*self.initial_range, self.original_masses[0] + self.original_masses[0]*self.initial_range ],
            1 : [self.original_masses[1] - self.original_masses[1]*self.initial_range, self.original_masses[1] + self.original_masses[1]*self.initial_range ],
            2 : [self.original_masses[2] - self.original_masses[2]*self.initial_range, self.original_masses[2] + self.original_masses[2]*self.initial_range ]
        }

        self.deltas = {
            0: self.original_masses[0] * self.delta,
            1: self.original_masses[1] * self.delta,
            2: self.original_masses[2] * self.delta
        }

        print('Dynamics parameters:', self.training_env.get_attr("env")[0].get_parameters())

        # Initialize wandb logging
        wandb.init(project="auto_dr_hyperparameter_tuning", config={
            "delta": self.deltas,
            "threshold_H": self.threshold_H,
            "threshold_L": self.threshold_L,
            "initial_range": self.initial_range
        })
        wandb.run.name = "AutoDR_Run_" + str(f"AutoDR_Run_delta_{self.delta}_threshold_{self.threshold_H}")
        wandb.run.save() 

    def _on_step(self) -> bool: # after each env.step()
        if any(self.locals['dones']): #array of size 1, False if episode not done, True if last step of the episode
            self.training_env.get_attr("env")[0].reset()  # Reset the environment before modification

            if np.random.uniform(0,1) < self.boundary_sampling_probabilty: #at the end of each episode have the same prob to launch AutoDR or keep training
                self.autoDR()
            else:
                self.training_env.get_attr("env")[0].sample_parameters_autoDR(self.bounds) #execute normal Domain Randomization

            # Log the bounds and entropy at the end of each episode
            bounds_log = {f"bounds_param_{i}": self.bounds[i] for i in self.bounds}
            entropy = self.compute_entropy()
            wandb.log({"entropy": entropy, "step": self.num_timesteps})
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

    start = timer()
    """ 
    TO TUNE
    delta: percentage of the original mass to increase or decrease the distribution bounds [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    threshold_high: upper threshold to decide if increase or decrease entropy (lower=high/2) [800, 1000, 1200]
     """
    source_env = gym.make('CustomHopper-source-v0')
    source_env = Monitor(source_env)
    target_env = gym.make('CustomHopper-target-v0')
    target_env = Monitor(target_env)

    total_timesteps = 2_000_000

    # Hyperparameters to tune - This were tuned separately and then results where compared, in order to find the best hp
    delta_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    #threshold_high_values = [800, 1000, 1200]
    threshold_high_values = [1000]


    #PPO hyperparameters found previously
    lr = 0.0012541462101916157
    gamma = 0.9885202328222382
    bs = 123
    nsteps = 3936
    nepochs = 10
    gl = 0.95

    # Store results
    results = []

    for delta in delta_values:
        for threshold_high in threshold_high_values:

            #only one seed (seed 0) for tuning. Using the same seed makes tuning significant
            model = PPO("MlpPolicy", source_env, learning_rate= lr, n_steps=nsteps, gamma=gamma, batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=0, verbose=2)
            run_name = f"AutoDR_Run_delta_{delta}_threshold_{threshold_high}"
            wandb.init(project="auto_dr_hyperparameter_tuning", name=run_name, config={
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

            model.learn(total_timesteps=total_timesteps, callback=AutoDRCallback(verbose=2, delta=delta, threshold_high=threshold_high))
            wandb.finish()
            end = timer()
            print(f"Total training time for {total_timesteps} steps: {end - start:.3f} s")

            mean_reward_target, std_reward_target = evaluate_policy(model, source_env, n_eval_episodes=50, deterministic=True)
            print(f"Delta: {delta}, Threshold High: {threshold_high}, Mean Reward: {mean_reward_target}, Std Reward: {std_reward_target}")

            results.append({
                "delta": delta,
                "threshold_high": threshold_high,
                "mean_reward": mean_reward_target,
                "std_reward": std_reward_target
            })

    print("\nFinal Results:")
    for result in results:
        print(f"Delta: {result['delta']}, Threshold High: {result['threshold_high']}, Mean Reward: {result['mean_reward']}, Std Reward: {result['std_reward']}")


if __name__ == '__main__':
    main()


""" Total training time for 2000000 steps: 10956.262 s
Delta: 0.2, Threshold High: 1000, Mean Reward: 710.0277384599999, Std Reward: 5.088458054424435

Final Results:
Delta: 0.005, Threshold High: 1000, Mean Reward: 1397.4480519800002, Std Reward: 276.8104567217324
Delta: 0.01, Threshold High: 1000, Mean Reward: 786.2596197799999, Std Reward: 486.70646547950076
Delta: 0.02, Threshold High: 1000, Mean Reward: 1430.3300085199999, Std Reward: 248.688812181205
Delta: 0.05, Threshold High: 1000, Mean Reward: 1098.6040210999997, Std Reward: 197.52024997225277
Delta: 0.1, Threshold High: 1000, Mean Reward: 535.56103384, Std Reward: 27.10573297592735
Delta: 0.2, Threshold High: 1000, Mean Reward: 710.0277384599999, Std Reward: 5.088458054424435 """
