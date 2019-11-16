# policy_transfer

Train examples:

1. DQN

A) Gridworld

python main.py -a dqn -c dqn_conf -g grid -d grid_conf -n 30000 -e 99 -s 112 -o adam learning_rate=3e-4 memory_size=2000000 replace_target_iter=1000 e_greedy=0.95 e_greedy_increment=2e-5 reward_decay=0.99 batch_size=128 soft_update=False save_model=True n_layer_1=20 save_per_episodes=1000 task=324 learning_step=10000

B) Pinball

python main.py -a dqn -c dqn_conf -g pinball -d pinball_conf -n 30000 -e 499 -s 112 -o adam learning_rate=3e-4 memory_size=2000000 replace_target_iter=1000 e_greedy=0.95 e_greedy_increment=2e-5 reward_decay=0.99 batch_size=128 soft_update=False save_model=True n_layer_1=20 configuration=game/pinball_hard_single.cfg sequential_state=False continuous_action=False start_position=[[0.6,0.4]] target_position=[0.1,0.1] learning_step=10000 save_per_episods=1000 


2. A3C

A) Gridworld

python main.py -a a3c -c a3c_conf -g grid -d grid_conf -n 20000 -e 99 -s 1 -o adam ENTROPY_BETA=0.01 n_layer_a_1=64 n_layer_c_1=64 learning_rate_a=3e-4 learning_rate_c=3e-4 reward_decay=0.99 save_model=True save_per_episodes=1000 task=324 batch_size=32 

B) Pinball

python main.py -a a3c -c a3c_conf -g pinball -d pinball_conf  -n 20000 -e 499 -s 3 -o adam ENTROPY_BETA=0.0001 n_layer_a_1=128 n_layer_c_1=128 learning_rate_a=3e-4 learning_rate_c=3e-4 reward_decay=0.99 batch_size=32 sequential_state=False continuous_action=True configuration=game/pinball_hard_single.cfg random_start=True start_position=[[0.6,0.4]] target_position=[0.1,0.1] save_model=True save_per_episodes=1000 action_clip=1 reward_normalize=False clip_value=0.2

C) Reacher

python main.py -a a3c -c a3c_conf -g reacher -d reacher_conf -n 10000 -e 1000 -s 3 -o adam ENTROPY_BETA=0.0003 n_layer_a_1=256 n_layer_c_1=256 learning_rate_a=1e-4 learning_rate_c=1e-4 reward_decay=0.99 task=hard clip_value=10 batch_size=256 reward_normalize=True done_reward=10 learning_times=1 N_WORKERS=8 USE_CPU_COUNT=False

3. PPO

A) Gridworld

python main.py -a ppo -c ppo_conf -g grid -d grid_conf -n 30000 -e 99 -s 20 -o adam n_layer_a_1=100 n_layer_c_1=100 c2=0.001 learning_rate_a=3e-4 learning_rate_c=3e-4 reward_decay=0.99 clip_value=0.2 save_model=True save_per_episodes=1000 task=324 batch_size=32 

B) Pinball

nohup python main.py -a ppo -c ppo_conf -g pinball -d pinball_conf -n 20000 -e 499 -s 5 -o adam n_layer_a_1=64 n_layer_c_1=64 c2=0.0001 learning_rate_a=3e-4 learning_rate_c=3e-4 reward_decay=0.99 clip_value=0.2 continuous_action=True configuration=game/pinball_hard_single.cfg random_start=True start_position=[[0.6,0.4]] target_position=[0.1,0.1] save_model=True save_per_episodes=10000 sequential_state=False reward_normalize=False action_clip=1 batch_size=64 use_layer_2=False > out5ppo0101.log 2>&1 &

C) Reacher

python main.py -a ppo -c ppo_conf -g reacher -d reacher_conf -n 20000 -e 1000 -s 2 -o adam n_layer_a_1=256 n_layer_c_1=256 learning_rate_a=1e-4 learning_rate_c=1e-4 reward_decay=0.99 save_per_episodes=1000 task=hard c1=0.0003 clip_value=10 batch_size=256 reward_normalize=True done_reward=10

4. Caps

A) Gridworld

python main.py -a caps -c caps_conf -g grid -d grid_conf -n 20000 -e 99 -s 100 -o adam option_layer_1=32 learning_rate_o=1e-3 learning_rate_t=1e-3 e_greedy=0.95 e_greedy_increment=1e-3 replace_target_iter=1000 batch_size=32 reward_decay=0.99 option_model_path=[source_policies/grid/81/81,source_policies/grid/459/459,source_policies/grid/65/65,source_policies/grid/295/295] save_model=True save_per_episodes=1000 learning_step=1000 task=324

B) Pinball

python main.py -a caps -c caps_conf -g pinball -d pinball_conf -n 20000 -e 499 -s 1 -o adam option_layer_1=32 learning_rate_o=3e-4 learning_rate_t=3e-4 e_greedy=0.95 e_greedy_increment=1e-3 replace_target_iter=1000 batch_size=32 reward_decay=0.99 option_model_path=['source_policies/a3c/0.90.9/model','source_policies/a3c/0.90.2/model','source_policies/a3c/0.20.9/model'] learning_step=10000 save_per_episodes=1000 sequential_state=False continuous_action=True configuration=game/pinball_hard_single.cfg random_start=True start_position=[[0.6,0.4]] target_position=[0.1,0.1] source_policy=a3c save_model=True action_clip=1 reward_normalize=False

5. PTF_A3C

A) Gridworld

python main.py -a ptf_a3c -c ptf_a3c_conf -g grid -d grid_conf -n 20000 -e 99 -s 37 -o adam ENTROPY_BETA=0.0001 n_layer_a_1=64 n_layer_c_1=64 learning_rate_a=3e-4 learning_rate_c=3e-4 learning_rate_o=1e-3 learning_rate_t=1e-3 e_greedy=0.95 e_greedy_increment=1e-3 replace_target_iter=1000 option_batch_size=32 batch_size=32 reward_decay=0.99 option_model_path=[source_policies/grid/81/81,source_policies/grid/459/459,source_policies/grid/65/65,source_policies/grid/295/295] learning_step=10000 save_per_episodes=1000 save_model=True task=324 c1=0.001 N_WORKERS=8 USE_CPU_COUNT=False clip_value=10 option_clip_value=10 option_layer_1=32

B) Pinball

python main.py -a ptf_a3c -c ptf_a3c_conf -g pinball -d pinball_conf -n 20000 -e 499 -s 1 -o adam ENTROPY_BETA=0.0001 n_layer_a_1=64 n_layer_c_1=64 option_layer_1=32 learning_rate_a=3e-4 learning_rate_c=3e-4 learning_rate_o=1e-3 learning_rate_t=1e-3  e_greedy=0.95 e_greedy_increment=1e-3 replace_target_iter=1000 option_batch_size=32 batch_size=32 reward_decay=0.99 option_model_path=['source_policies/a3c/0.90.9/model','source_policies/a3c/0.90.2/model','source_policies/a3c/0.20.9/model'] learning_step=10000 save_per_episodes=1000 sequential_state=False continuous_action=True configuration=game/pinball_hard_single.cfg random_start=True start_position=[[0.6,0.4]] target_position=[0.1,0.1] c1=0.0005 source_policy=a3c save_model=True action_clip=1 reward_normalize=False

C) Reacher

python main.py -a ptf_a3c -c ptf_a3c_conf -g reacher -d reacher_conf -n 20000 -e 1000 -s 3 -o adam ENTROPY_BETA=0.0003 n_layer_a_1=256 n_layer_c_1=256 learning_rate_a=1e-4 learning_rate_c=1e-4 learning_rate_o=5e-4 learning_rate_t=5e-4 e_greedy=0.95 e_greedy_increment=1e-3 replace_target_iter=1000 reward_decay=0.99 option_model_path=['source_policies/reacher/t1/model','source_policies/reacher/t2/model','source_policies/reacher/t3/model','source_policies/reacher/t4/model'] learning_step=1000 save_per_episodes=1000 save_model=True task=hard c1=0.0005 source_policy=a3c clip_value=10 option_clip_value=10 batch_size=256 option_batch_size=32 reward_normalize=True done_reward=10 learning_times=1 option_layer_1=32 N_WORKERS=8 USE_CPU_COUNT=False

6. PTF_PPO

A) Gridworld

python main.py -a ptf_ppo -c ptf_ppo_conf -g grid -d grid_conf -n 20000 -e 99 -s 19 -o adam n_layer_a_1=64 n_layer_c_1=64 c2=0.001 learning_rate_a=3e-4 learning_rate_c=3e-4 learning_rate_o=3e-4 learning_rate_t=3e-4 reward_decay=0.99 clip_value=0.3 e_greedy=0.95 e_greedy_increment=1e-3 replace_target_iter=1000 option_batch_size=32 batch_size=32 option_model_path=[source_policies/grid/81/81,source_policies/grid/459/459,source_policies/grid/65/65,source_policies/grid/295/295] learning_step=10000 save_per_episodes=1000 save_model=True task=324 c3=0.001

B) Pinball

python main.py -a ptf_ppo -c ptf_ppo_conf -g pinball -d pinball_conf -n 20000 -e 499 -s 2 -o adam n_layer_a_1=64 n_layer_c_1=64 learning_rate_a=3e-4 learning_rate_c=3e-4 learning_rate_o=1e-3 learning_rate_t=1e-3 e_greedy=0.95 e_greedy_increment=5e-4 replace_target_iter=1000 option_batch_size=32 batch_size=64 reward_decay=0.99 option_model_path=['source_policies/a3c/0.90.9/model','source_policies/a3c/0.90.2/model','source_policies/a3c/0.20.9/model'] learning_step=1000 save_per_episodes=10000 sequential_state=False continuous_action=True configuration=game/pinball_hard_single.cfg start_position=[[0.6,0.4]] random_start=True target_position=[0.95,0.05] c2=0.0001 source_policy=a3c save_model=True action_clip=1 reward_normalize=False option_layer_1=32 c3=0.001 xi=0.001


C) Reacher
python main.py -a ptf_ppo -c ptf_ppo_conf -g reacher -d reacher_conf -n 20000 -e 1000 -s 2 -o adam n_layer_a_1=256 n_layer_c_1=256 learning_rate_a=1e-4 learning_rate_c=1e-4 learning_rate_o=1e-3 learning_rate_t=1e-3 e_greedy=0.95 e_greedy_increment=1e-2 replace_target_iter=1000 reward_decay=0.99 option_model_path=['source_policies/reacher/t1/model','source_policies/reacher/t2/model','source_policies/reacher/t3/model','source_policies/reacher/t4/model'] learning_step=10000 save_per_episodes=1000 task=hard c1=0.001 source_policy=a3c clip_value=10 batch_size=256 option_batch_size=32 reward_normalize=True done_reward=10 option_layer_1=32