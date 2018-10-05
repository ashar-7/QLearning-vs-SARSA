"""This module contains a class for the SARSA agent."""

from collections import defaultdict
import os
import time
import numpy as np
import gym

class SARSA_Agent:
    """Q Learning agent for an Open AI gym environment.
    You may need to modify the class a little bit for some environments"""

    def __init__(self, env):
        self.env = env # environment
        self.possible_actions = [a for a in range(env.action_space.n)] # list of possible actions
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n)) # action-values
        self.policy = self.create_empty_policy() # policy

    def create_empty_policy(self):
        """Returns a defaultdict empty policy"""
        policy = defaultdict(self.env.action_space.sample)
        return policy

    def update_policy(self, state):
        """Updates policy[state] from Q[state]"""

        # loop over the states present in Q
        best_action = np.argmax(self.Q[state]) # action that has the maximum action-value
        self.policy[state] = best_action

    def e_greedy_action(self, state, epsilon):
        """Assuming the policy contains greedy actions derived from self.Q, this function returns a random
        action with epsilon probability, or the optimal action with (1-epsilon) probability"""

        if np.random.random() < (1-epsilon):
            return self.policy[state] # return the optimal action

        return np.random.choice(self.possible_actions) # return a random action

    def learn(self, n_episodes, gamma=0.999, epsilon=0.01, alpha=0.2):
        """Learn the optimal policy and action-values using the SARSA algorithm"""
        rewards_list = []

        # loop over the number of episodes
        for i in range(n_episodes):
            total_reward = 0 # initialize total reward for current episode

            S = self.env.reset()
            A = self.e_greedy_action(S, epsilon)

            while True:
                # S_ = S' (next state), A_ = A' (next action)

                S_, R, done, _ = self.env.step(A)
                A_ = self.e_greedy_action(S_, epsilon) # select an action based on e-greedy policy
                self.Q[S][A] = self.Q[S][A] + alpha * (R + gamma*self.Q[S_][A_] - self.Q[S][A]) # update  Q(s,a)
                S = S_
                A = A_

                # update the policy to be greedy with respect to Q
                self.update_policy(S)

                total_reward += R
                if done:
                    break

            rewards_list.append(total_reward)

            if i % 100 == 0 and i != 0:
                print('SARSA Agent: Episode #{} Average Rewards: {}'.format(i, sum(rewards_list)/i))

        return self.policy, self.Q

    def generate_test_episode(self):
        """Generates and renders an episode"""

        total_reward = 0
        S = self.env.reset()
        while True:
            A = self.policy[S]
            S, R, done, _ = self.env.step(A)

            # render the environment
            os.system('cls')
            print('SARSA:')
            self.env.render()
            time.sleep(0.5)

            total_reward += R

            if done:
                break

        return total_reward


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')

    agent = SARSA_Agent(env)
    policy, Q = agent.learn(20000)

    print('Total reward for test episode: ', agent.generate_test_episode())
