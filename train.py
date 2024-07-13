"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE algorithms
    #Based on train.py from https://github.com/gabrieletiboni/mldl_2024_template

"""
import argparse

import pandas as pd

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy
from agent_state_value import Agent_states, Policy_states, BaselineNetwork #Needed only for REINFORCE with baseline value=state-vaue
from agent_ac import Agent_ac, Policy_ac #Needed for Actor-Critic

from timeit import default_timer as timer

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	env_target = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	"""
		Initialize wandb
	"""
	config = {
	    "policy_type": "MlpPolicy",
	    "total_timesteps": 1000000, #Here insert the number of timesteps needed
	    "env_id": "CartPole-v1",
	    "test_episodes": 50
	}

	run = wandb.init(
	    project="reinforce_test",
	    config=config,
	    sync_tensorboard=True
	)
	wandb.run.name = "Reinforce_Run_" #Optional: change the name of the wandb run (for better clarity)
	wandb.run.save()

	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	"""
		Initialize policy and agent: choose which one according to the algorithm selected
	"""

	#For Actor-Critic:
    #policy = Policy_ac(observation_space_dim, action_space_dim)
	#agent = Agent_ac(policy, device=args.device)	

    #Else for REINFORCE with baseline value = state-value:
    #policy = Policy_states(observation_space_dim, action_space_dim)
	#baseline_network = BaselineNetwork(observation_space_dim)
	#agent = Agent_states(policy, baseline_network, device=args.device)
    
    #Otherwise:
	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)
	
	"""
		Training
	"""
    #
    # TASK 2 and 3: interleave data collection to policy updates
    #
	
    #Initialize lists to store rewards and timesteps for logging
	all_mean_rewards = []
	all_std_rewards = []
	all_timestep_logs = []
	
    #Timer needed to take track of time consuptions
	start = timer()

	total_rewards = []
	
	state = env.reset()  # Reset the environment and observe the initial state
	train_reward = 0
	total_timesteps = 0


	# Training loop for the number of timesteps prior specified
	for total_timesteps in range (config["total_timesteps"]):
	
        #Get the action from the agent
		action, action_probabilities = agent.get_action(state)
		previous_state = state

        #Take action in the environment and get next state and reward values
		state, reward, done, info = env.step(action.detach().cpu().numpy())

        #Store the outcome in the agent
		agent.store_outcome(previous_state, state, action_probabilities, reward, done)

		train_reward += reward
		total_timesteps += 1
	
		if done:
            #Episod ended => update policy
			agent.update_policy()
			total_rewards.append(train_reward)
			state = env.reset()
			train_reward = 0
			
			
			#Log every episode		
			mean_reward = np.mean(total_rewards)
			std_reward = np.std(total_rewards)

			all_mean_rewards.append(mean_reward)
			all_std_rewards.append(std_reward)
			wandb.log({
                f"mean_reward_": mean_reward, 
                f"std_reward_": std_reward,
                "timestep": total_timesteps + 1})
			
			all_timestep_logs.append(total_timesteps + 1)
	


	print(f"Training completed!")
		
	end = timer()
			
	"""		
	    Testing the trained policy on the source environment
	"""
	all_test_rewards_source=[]
	all_test_std_source=[]
	test_rewards_source=[]
	for _ in range(config["test_episodes"]):
		state = env.reset()
		test_reward = 0
		done = False
		while not done:
			action, action_probabilities = agent.get_action(state)
			previous_state = state
			state, reward, done, info = env.step(action.detach().cpu().numpy())
			agent.store_outcome(previous_state, state, action_probabilities, reward, done)
			test_reward += reward
		test_rewards_source.append(test_reward)
	all_test_rewards_source.append(np.mean(test_rewards_source))
	all_test_std_source.append(np.std(test_rewards_source))
		
	wandb.log({
        "test_source_mean_reward": np.mean(all_test_rewards_source), 
        "test_source_std_reward": np.mean(all_test_std_source)})

	print("Test on source environment completed!")
	


	"""		
	    Testing the trained policy on the target environment
	"""	
	all_test_rewards_target=[]
	all_test_std_target=[]
	test_rewards_target=[]
	for episode in range(config["test_episodes"]):
		state = env_target.reset()
		test_reward = 0
		done = False
		while not done:
			action, action_probabilities = agent.get_action(state)
			previous_state = state
			state, reward, done, info = env_target.step(action.detach().cpu().numpy())
			agent.store_outcome(previous_state, state, action_probabilities, reward, done)
			test_reward += reward
		test_rewards_target.append(test_reward)
	all_test_rewards_target.append(np.mean(test_rewards_target))
	all_test_std_target.append(np.std(test_rewards_target))			

	wandb.log({
        "test_target_mean_reward": np.mean(all_test_rewards_target), 
        "test_target_std_reward": np.mean(all_test_std_target)})
	
	print("Test on target environment completed!")

	
    
	print(f"Total training time for {config['total_timesteps']} steps: {end - start:.3f} s")

	run.finish()

    #Optional: save the trained policy model
	torch.save(agent.policy.state_dict(), "model_reinforce.mdl")

	

if __name__ == '__main__':
	main()