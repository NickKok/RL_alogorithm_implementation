import numpy as np
import gym
from policy_gradient_torch import Agent
from utils import plotLearning
#from gym import wrappers


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(lr=0.001, input_dims=[8], gamma=0.99, n_actions=4,
                  l1_size=128, l2_size=128)
    
    score_history = []
    score = 0
    n_episodes = 25000

    # env = wrappers.Monitor(env, 'tmp/lunar-lander', video_callable=lambda
    #                        episode_id: True, force=True)

    for i in range(n_episodes):
        print('episode: ',i,  'score: %.3f' % score)
        done = False
        score = 0
        observation = env.reset()

        while not done:
            env.render()
            action = agent.choose_action(observation)
            obs, reward, done, info = env.step(action)
            agent.store_rewards(reward)
            observation = obs
            score += reward
        score_history.append(score)
        agent.learn()

    filename = 'lunar-lander.png'
    plotLearning(score_history, filename, x=None, window=25)
