import numpy as np
import tensorflow as tf
from alg.source_actor import SourceActor as SA
from alg.PRM_actor import PRM_actor as PRM_AC

def run(args, env, alg, logger):
    if args['load_model']:
        alg.load_model(args['load_model_path'])

    memory = np.zeros(args['reward_memory'])
    discount_memory = np.zeros (args['reward_memory'])
    numGames = args['numGames']
    totalreward = np.zeros(numGames)
    for episode in range(numGames):
        # initial observation
        observation = env.reset()
        observation = np.array(observation)
        step = 0
        episode_reward = 0
        episode_discount_reward = 0
        while True:
            # RL choose action based on observation
            if args['algorithm'] == 'caps':
                SOURCE_TASK = args['option_model_path']
                if args['continuous_action']:
                    action_dim = len(env.pri_action)
                else:
                    action_dim = args['action_dim']
                N_O = len(args['option_model_path']) + action_dim
                g = [tf.Graph() for i in range(N_O)]
                actor_sess = [tf.Session(graph=i) for i in g]
                actor = []
                for i in range(len(SOURCE_TASK)):
                    with actor_sess[i].as_default():
                        with g[i].as_default():
                            dqn = SA(SOURCE_TASK[i], args, actor_sess[i])
                            actor.append(dqn)
                if args['continuous_action']:
                    for i in range(len(env.pri_action)):
                        with actor_sess[i].as_default():
                            with g[i].as_default():
                                AC = PRM_AC(env.pri_action[i], len(env.pri_action))
                                actor.append(AC)
                else:
                    for i in range(action_dim):
                        with actor_sess[i].as_default():
                            with g[i].as_default():
                                AC = PRM_AC(i, action_dim)
                                actor.append(AC)
                option = alg.choose_o(observation)
                action = actor[option].choose_action_g(observation)
            else:
                action = alg.choose_action(observation)
            # RL take action and get next observation and reward
            if type(action) is tuple:
                observation_, reward, done, _ = env.step(action[0])
            else:
                observation_, reward, done, _ = env.step(action)
            observation_ = np.array(observation_)
            env.render()
            episode_reward += reward
            episode_discount_reward = episode_discount_reward + round (
                reward * np.power (args['reward_decay'], step),
                8)
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done or step > args['epi_step']:
                memory[episode % args['reward_memory']] = episode_reward
                discount_memory[episode % args['reward_memory']] = episode_discount_reward
                totalreward[episode] = np.mean(np.mean(memory))
                print('done: ', done, ' step: ', step, ' reward: ', episode_reward, 'discount_reward: ',
                      episode_discount_reward, ' epi: ', episode)
                break
            step += 1

    import matplotlib.pyplot as plt
    ax1 = plt.subplot(221)
    plt.sca(ax1)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(np.arange(len(totalreward)), totalreward)
    # end of game
    print('game over')
    # env.destroy()
    ax2 = plt.subplot(222)
    plt.sca(ax2)
