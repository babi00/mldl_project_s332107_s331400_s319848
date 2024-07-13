from stable_baselines3.common.callbacks import BaseCallback

import gym
from env.custom_hopper import *

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import matplotlib.pyplot as plt

import wandb

import numpy as np

class SampleParametersCallback(BaseCallback):

    def __init__(self, verbose: int = 0, dist=0):
        super().__init__(verbose)

        self.dist = dist

    def _on_training_start(self) -> None: # before the first rollout

        print("--Starting the rollout--")
        masses_randomization(self.training_env.get_attr("env")[0], self.dist)  # Apply randomization
        print("Sampling parameters")
        print('Dynamics parameters:', self.training_env.get_attr("env")[0].get_parameters())

        self.globals['episodes'] = 0


    def _on_step(self) -> bool: # after each env.step()
    
        if any(self.locals['dones']): #array of size 1, False if episode not done, True if last step of the episode
            self.globals['episodes'] += 1
            #print("\n\n LOCALS BEFORE EPISODE END: ", self.locals['dones'])
            #print("\n\n")
            self.training_env.reset()  # Reset the environment before modification
            masses_randomization(self.training_env.get_attr("env")[0], self.dist)  # Apply randomization
            print('Dynamics parameters:', self.training_env.get_attr("env")[0].get_parameters()) # check randomization

        return True


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print("--Training has ended--")
        print("Total number of episodes: ", self.globals['episodes'])
        #print("Total number of timesteps: ", self.globals['tot_steps'])


def masses_randomization(env, dist):
    #defined a new function in the custom_hopper that samples parameters from the uniform distributions of the masses.
    env.sample_parameters(dist)

    return


def plot_results(rewards, std_devs, source):
    runs = ["5%", "20%", "50%", "75%", "100%", "300%", "600%"]
    lower_bound = 813 #Mean reward for PPO in the source->target configuration
    upper_bound = 1175 #Mean reward for PPO in the target->target configuration

    fig, ax = plt.subplots()

    if source:
        color = 'blue'
    else:
        color = 'orange'

    ax.bar(runs, rewards, color=color, yerr=std_devs, capsize=5, label='Reward')

    ax.axhline(y=lower_bound, color='black', linestyle='dashed', linewidth=2, label='Lower Bound')
    ax.axhline(y=upper_bound, color='black', linestyle='dashdot', linewidth=2, label='Upper Bound')

    # Add labels and title
    ax.set_xlabel('Runs')
    ax.set_ylabel('Mean reward')

    ax.legend()
    if source:
        ax.set_title('Mean rewards for different ranges of distribution (source)')
        plt.savefig("./visuals/domain_randomization_source.pdf", bbox_inches='tight', format='pdf')
    else:
        ax.set_title('Mean rewards for different distributions (target)')
        plt.savefig("./visuals/domain_randomization_target.pdf", bbox_inches='tight', format='pdf')



def main():
    source_env = gym.make('CustomHopper-source-v0')
    source_env = Monitor(source_env)
    target_env = gym.make('CustomHopper-target-v0')
    target_env = Monitor(target_env)

    #substitute with the best hyperparameters found in the PPO tuning
    """Best hyperparameters for the model:  {'best_mean_reward': 1406.35123122, 'best_std_reward': 84.98911874395765,
     'lr': 0.0012541462101916157, 'n_steps': 0, 'gamma': 0.9885202328222382,
      'batch_size': 123, 'n_epochs': 11, 'gl': 0.9472233613269306, 'nsteps': 3936} """
    lr = 0.0012541462101916157
    gamma = 0.9885202328222382
    bs = 123
    nsteps = 3936
    nepochs = 10
    gl = 0.95
    
    total_timesteps = 1_000_000

    source_rewards = []
    source_std_devs = []
    target_rewards = []
    target_std_devs = []

    config = {"policy_type" : "MlpPolicy", "total_timesteps" : total_timesteps, "env_id" : source_env}

    for distance in [0.05, 0.2, 0.5, 0.75, 1, 3, 6]:   

        run = wandb.init(project='1M_DR_try0', config = config, sync_tensorboard = True, name = f'1M_DR_{distance*100}')

        original_parameters = source_env.original_masses
        print(f"Distance: {distance*100}%")
        print("Original masses [thigh, leg, foot]: ", original_parameters[1:]) #exclude torso
        print("Min values: ", (original_parameters * (1-distance))[1:])
        print("Max values: ", (original_parameters * (1+distance))[1:])

        #source->source
        model_1 = PPO("MlpPolicy", source_env, learning_rate= lr, n_steps=nsteps, gamma=gamma, batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=0, verbose=2)
        model_1.learn(total_timesteps = total_timesteps, callback=SampleParametersCallback(dist=distance))
        #evaluate on the desired environment (source or target)
        mean_reward_source_1, std_reward_source_1 = evaluate_policy(model_1, source_env, n_eval_episodes=50, deterministic=True)
        mean_reward_target_1, std_reward_target_1 = evaluate_policy(model_1, target_env, n_eval_episodes=50, deterministic=True)

        #Train again, with a different seed (same as the one used to evaluate upper and lower bound)
        
        model_2 = PPO("MlpPolicy", source_env, learning_rate= lr, n_steps=nsteps, gamma=gamma, batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=14, verbose=2)
        model_2.learn(total_timesteps = total_timesteps, callback=SampleParametersCallback(dist=distance))
        mean_reward_source_2, std_reward_source_2 = evaluate_policy(model_2, source_env, n_eval_episodes=50, deterministic=True)
        mean_reward_target_2, std_reward_target_2 = evaluate_policy(model_2, target_env, n_eval_episodes=50, deterministic=True)

        #Train again, with a different seed (same as the one used to evaluate upper and lower bound)

        model_3 = PPO("MlpPolicy", source_env, learning_rate= lr, n_steps=nsteps, gamma=gamma, batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=42, verbose=2, tensorboard_log=f'runs/{run.id}')
        model_3.learn(total_timesteps = total_timesteps, callback=[SampleParametersCallback(dist=distance), WandbCallback(verbose=2)])
        mean_reward_source_3, std_reward_source_3 = evaluate_policy(model_3, source_env, n_eval_episodes=50, deterministic=True)
        mean_reward_target_3, std_reward_target_3 = evaluate_policy(model_3, target_env, n_eval_episodes=50, deterministic=True)

        run.finish()

        #Now we can compute a more statistically meaningful mean reward and standard deviation
        mean_mean_reward_source = (mean_reward_source_1 + mean_reward_source_2 + mean_reward_source_3)/3
        mean_std_reward_source = (std_reward_source_1 + std_reward_source_2 + std_reward_source_3)/3 
        source_rewards.append(mean_mean_reward_source)
        source_std_devs.append(mean_std_reward_source)

        mean_mean_reward_target = (mean_reward_target_1 + mean_reward_target_2 + mean_reward_target_3)/3
        mean_std_reward_target = (std_reward_target_1 + std_reward_target_2 + std_reward_target_3)/3 
        target_rewards.append(mean_mean_reward_target)
        target_std_devs.append(mean_std_reward_target)
        

    print("Results: ")
    print("Mean reward (source): ", source_rewards)
    print("Mean std (source): ", source_std_devs)

    print("Mean reward (target): ", target_rewards)
    print("Mean std (target): ", target_std_devs)
    
    plot_results(source_rewards, source_std_devs, True)
    plot_results(target_rewards, target_std_devs, False)

    
    """ Results: 350K
Mean reward (source):  [1605.3641672800002, 1105.9063194266666, 1220.9362943466667, 1290.9993876400001, 1236.5631144066667, 684.18743, 359.88818033999996]
Mean std (source):  [11.17039062269587, 15.043798201570285, 51.98070239372015, 12.005295854724757, 20.937502168532163, 2.7108233792057526, 65.40588011109445]
Mean reward (target):  [1013.7628226333333, 985.7868127266665, 839.10362436, 987.6827566066668, 940.1514930866666, 900.4854377533334, 589.1728635533333]
Mean std (target):  [22.870646054778728, 8.926406439818642, 17.279050101282422, 99.78937895297503, 26.500015407937315, 5.117145905353805, 21.340576390671984] """

""" Results: 1M
Mean reward (source):  [1537.7231714466668, 1172.9636771066666, 1067.9648539266666, 1156.5622223933333, 742.5758025133332, 883.465588, 502.18778109333334]
Mean std (source):  [158.40769039543417, 138.09815521908536, 122.36977876509957, 119.42375934957208, 77.46179510293861, 28.787820328124138, 103.17841901905702]
Mean reward (target):  [768.1654110533333, 792.1414004466666, 961.3012604066668, 833.6904494533334, 686.2474419666665, 711.5534274733333, 668.8697461733333]
Mean std (target):  [36.3837587328591, 48.889113324049255, 112.95701640062298, 27.38863409775075, 16.2801035548944, 91.61529216198113, 104.2934099006958]
"""

if __name__ == '__main__':
    main()