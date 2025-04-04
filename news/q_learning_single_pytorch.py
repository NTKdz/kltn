# q_learning_single_pytorch.py
from environment import Environment
from parameters import *
import numpy as np
import random

class QLearningAgentSingle:
    def __init__(self):
        self.env = Environment(num_users=1)
        self.q_table = np.zeros((num_states, num_actions))
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999

    def get_action(self, state_idx):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        possible_actions = self.env.get_possible_actions(0)
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            q_values = self.q_table[state_idx]
            return max(possible_actions, key=lambda a: q_values[a])

    def train(self):
        total_reward = 0
        state = self.env.get_state()
        for i in range(T):
            state_idx = self.env.get_discrete_state(0)
            action = self.get_action(state_idx)
            total_reward_step, next_state, individual_rewards = self.env.step([action])
            total_reward += total_reward_step
            next_state_idx = self.env.get_discrete_state(0)
            max_next_q = np.max(self.q_table[next_state_idx])
            current_q = self.q_table[state_idx][action]
            self.q_table[state_idx][action] = (
                (1 - learning_rate_Q) * current_q + 
                learning_rate_Q * (individual_rewards[0] + gamma_Q * max_next_q)
            )
            state = next_state
            if (i + 1) % step == 0:
                avg_reward = total_reward / (i + 1)
                print(f"Iteration {i + 1}, Avg Reward: {avg_reward:.4f}")
        return total_reward / T

if __name__ == "__main__":
    agent = QLearningAgentSingle()
    avg_reward_single = agent.train()
    print(f"Single-user Q-learning average reward: {avg_reward_single:.4f}")