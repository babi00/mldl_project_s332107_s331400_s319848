"""Training PPO for 1M timesteps in 3 different seed, to see how the training mean reward goes. Then evaluating the models on source and 
target environment, and getting the mean reward for the three seeds.
"""

import gym
from env.custom_hopper import *

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from timeit import default_timer as timer

class LoggingCallback(BaseCallback):

    def __init__(self, seed_number, verbose: int = 0):
        super().__init__(verbose)

        #ONLY USED FOR WANDB LOGGING
        self.seed_number = seed_number
        #self.num_episodes = 0
        self.tot_reward = []

        
    def _on_training_start(self) -> None: # before the first rollout
        print("--Starting the rollout--")

        # Initialize wandb logging
        wandb.init(project="1M_PPO_training_v7", config={
            "seed" : self.seed_number
        })
        wandb.run.name = "SEED_ " + f'{self.seed_number}'
        wandb.run.save() 
        

    def _on_step(self) -> bool: # after each env.step()
        if any(self.locals['dones']): #array of size 1, False if episode not done, True if last step of the episode
            #self.num_episodes += 1 #keping count of number of episodes for wandb logging of mean reward
            self.tot_reward.append(self.locals['infos'][0]['episode']['r']) #keeping track of total reward for mean reward

            self.training_env.get_attr("env")[0].reset()  # Reset the environment before modification

            #CHECK SELF.LOCALS IF CURIOUS TO KNOW WHY EPISODE REWARDS ARE LOGGED LIKE THAT
            wandb.log({"step": self.num_timesteps, "mean_reward" : np.mean(self.tot_reward),
             "mean+std" : np.mean(self.tot_reward) + np.std(self.tot_reward),
             "mean-std" : np.mean(self.tot_reward) - np.std(self.tot_reward),
             "std" : np.std(self.tot_reward) })

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print("--Training has ended--")
        wandb.finish()

def main():

    source_env = gym.make('CustomHopper-source-v0')
    source_env = Monitor(source_env)
    target_env = gym.make('CustomHopper-target-v0')
    target_env = Monitor(target_env)

    total_timesteps = 1_000_000
    
    best_hp = {
        'lr': 0.0012541462101916157,
        'gamma': 0.9885202328222382,
        'batch_size': 123,
        'n_epochs': 11,
        'gl': 0.9472233613269306,
        'nsteps': 3936}


    start = timer()
    model_1 = PPO("MlpPolicy", source_env, learning_rate= best_hp['lr'], n_steps=best_hp['nsteps'], gamma=best_hp['gamma'], batch_size=best_hp['batch_size'], n_epochs=best_hp['n_epochs'], gae_lambda=best_hp['gl'], seed=0, verbose=2)
    model_1.learn(total_timesteps=total_timesteps, callback=LoggingCallback(seed_number=0, verbose=2))
    mean_reward_1, std_reward_1 = evaluate_policy(model_1, source_env, n_eval_episodes=50, deterministic=True)
    mean_reward_target_1, std_reward_target_1 = evaluate_policy(model_1, target_env, n_eval_episodes=50, deterministic=True)
    end = timer()


    model_2 = PPO("MlpPolicy", source_env, learning_rate= best_hp['lr'], n_steps=best_hp['nsteps'], gamma=best_hp['gamma'], batch_size=best_hp['batch_size'], n_epochs=best_hp['n_epochs'], gae_lambda=best_hp['gl'], seed=14, verbose=2)
    model_2.learn(total_timesteps=total_timesteps, callback=LoggingCallback(seed_number=14, verbose=2))
    mean_reward_2, std_reward_2 = evaluate_policy(model_2, source_env, n_eval_episodes=50, deterministic=True)
    mean_reward_target_2, std_reward_target_2 = evaluate_policy(model_2, target_env, n_eval_episodes=50, deterministic=True)


    model_3 = PPO("MlpPolicy", source_env, learning_rate= best_hp['lr'], n_steps=best_hp['nsteps'], gamma=best_hp['gamma'], batch_size=best_hp['batch_size'], n_epochs=best_hp['n_epochs'], gae_lambda=best_hp['gl'], seed=42, verbose=2)
    model_3.learn(total_timesteps=total_timesteps, callback=LoggingCallback(seed_number=42, verbose=2))
    mean_reward_3, std_reward_3 = evaluate_policy(model_3, source_env, n_eval_episodes=50, deterministic=True)
    mean_reward_target_3, std_reward_target_3 = evaluate_policy(model_3, target_env, n_eval_episodes=50, deterministic=True)

    print(f"Mean reward 1: {mean_reward_1}, Mean reward 2: {mean_reward_2}, Mean reward 3: {mean_reward_3}")
    # Mean reward 1: 1153.18382046, Mean reward 2: 1401.9460041800003, Mean reward 3: 1612.98179253998
    # Mean reward 1: 1518.5769801999998, Mean reward 2: 1319.43394196, Mean reward 3: 1381.0427715 WITH 350_000

    #for source->source
    mean_mean_reward = (mean_reward_1 + mean_reward_2 + mean_reward_3)/3  
    mean_std_reward = (std_reward_1 + std_reward_2 + std_reward_3)/3
    print("For source->source")
    print("Mean reward (source): ", mean_mean_reward) 
    print("Mean std (source): ", mean_std_reward)

    mean_reward_target = (mean_reward_target_1 + mean_reward_target_2 + mean_reward_target_3)/3   
    std_reward_target = (std_reward_target_1 + std_reward_target_2 + std_reward_target_3)/3
    print("For source->target")
    print("Mean reward (target): ", mean_reward_target) 
    print("Mean std (source): ", std_reward_target)

    print(f"[Seed 0] Total training time for {total_timesteps} steps: {end - start:.3f} s")

    #1M
    """ Mean reward:  1406.35123122  
    Std reward:  84.98911874395765"""

#5M - 5M results where not used in the final version
    """Mean reward 1: 852.5140894799999, Mean reward 2: 842.5254582, Mean reward 3: 993.23230186
Mean reward:  896.0906165133333
Mean std:  210.88089021882524
"""



if __name__ == '__main__':
    main()


#Here i will log the results for 1M, 17/06/2024
""" For source->source
Mean reward (source):  1287.5811857733333
Mean std (source):  124.74593667729384
For source->target
Mean reward (target):  846.1674052933332
Mean std (source):  51.699251929435725
[Seed 0] Total training time for 1000000 steps: 788.604 s

"""
