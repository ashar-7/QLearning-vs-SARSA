from sarsa import SARSA_Agent
from q_learning import QLearning_Agent
import time
import os
import gym

# CliffWalking-v0 and Taxi-v2 environments work without any changes
# Some other environments may require some changes in the SARSA_Agent and QLearning_Agent classes

N_EPISODES = 8000 # number of episodes for learning

env = gym.make('CliffWalking-v0')

sarsa_agent = SARSA_Agent(env)
qlearning_agent = QLearning_Agent(env)

sarsa_agent.learn(N_EPISODES)
qlearning_agent.learn(N_EPISODES)

os.system('cls')
time.sleep(1)
sarsa_reward = sarsa_agent.generate_test_episode()

os.system('cls')
time.sleep(1)
qlearning_reward = qlearning_agent.generate_test_episode()

print('Total reward for SARSA: {}'.format(sarsa_reward))
print('Total reward for Q-Learning: {}'.format(qlearning_reward))