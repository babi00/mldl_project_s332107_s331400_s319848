# Project 4: Reinforcement Learning (Course project MLDL 2024 - POLITO)

This repository contains all the code used for the project for the Reinforcement Learning topic in the 2024 Machine Learning and Deep Learning course, developed by students
- **s319848** Guido Spina
- **s331400** Sara Mola
- **s332107** Barbara Frittella

The different files were used for the training and testing of different models using different approach. The content of each file is reported below. Additional comments can be found inside the specific files. 
Note that since many models were tested in the same file used for training, the model has not been saved. Other models can be found in the `models` folder.
We also mention that we added some methods to `/env/custom_hopper.py`, that were exploited when training UDR and ADR. 

## Policy Gradient Algorithms

- `agent.py`: Agent for REINFORCE without baseline, REINFORCE with baseline = [20, 100, mean-reward]
- `agent_state_value.py`: Agent for REINFORCE with baseline=state-value
- `agent_ac.py`: Agent for Actor-Critic
- `train.py`: Training file for all previous agents

### PPO

- `train_and_test_ppo.py` : tuning of PPO hyperparameters (on source environment) using wandb sweep.
- `1M_PPO_SS.py`: Training of PPO model on env_source and evaluation on env_source
- `1M_PPO_ST.py`: Evaluation of `1M_PPO_SS` model on env_target
- `1M_PPO_TT.py`: Training of PPO model on env_target and evaluation on env_target

- `/models/best_model_ppo_ss.zip` : Best model trained in `1M_PPO_SS`
- `/models/best_model_ppo_tt.zip` : Best model trained in `1M_PPO_TT`
- `/models/best_model_ppo_ss_350.zip` : Best model found during the tuning of hyperparameters, therefore trained for 350k timesteps on the source environment.
- `/models/best_model_ppo_tt_350.zip` : Model trained for 350k timesteps on the target environment with the hyperparameters found during the tuning

## Domain Randomization

### Uniform Domain Randomization

- `1M_DR.py` : Tuning, Training and Evaluation of Uniform Domain Randomization

### Automatic Domain Randomization

- `AutoDR_tuning.py` : Tuning of the hyperparameters for AutoDR. Note that it appears as if 'threshold_high' has been tuned on one value only instead of three, because the tuning has been done separately (each member of the group tested one value of 'threshold_high' with all the values of 'delta').
- `AutoDR_0` : Training (on source_env) and testing of AutoDR for 10M timesteps on three seeds, on source and target enviroment. There is no separate file for "closed distribution" and "50% range distribution" approaches, we just commented out the portion of the code to separate the two. 
- `AutoDR_4M` : Training (on source_env) and testing of AutoDR for 4M timesteps on three seeds, on source and target enviroment. There is no separate file for "closed distribution" and "50% range distribution" approaches, we just commented out the portion of the code to separate the two. 

- `/models/model_1_autoDR_4M.zip` : Model trained for 4M timesteps with AutoDR on seed 0. 


