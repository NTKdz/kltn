# cmma_agent.py
# Agent for Document 1 (Yao and Jia, 2019) - Coordinated Multi-Agent Q-learning (CMAA)

from environment_doc1 import environment_doc1
import numpy as np
from parameters import *

class cmma_agent:
    def __init__(self, num_users=num_users, num_channels=num_channels):
        self.env = environment_doc1(num_users, num_channels)
        self.num_users = num_users
        self.num_channels = num_channels
        self.num_states = num_channels * (num_channels ** num_users)
        self.q_tables = [np.zeros((self.num_states, num_channels)) for _ in range(num_users)]
        self.learning_rate = learning_rate_doc1
        self.discount_factor = discount_factor_doc1
        self.epsilon = epsilon_start_doc1
        self.epsilon_min = epsilon_min_doc1
        self.epsilon_decay = epsilon_decay_doc1

    def state_to_index(self, state):
        jammer_channel, joint_actions = state
        joint_action_idx = sum(a * (self.num_channels ** i) for i, a in enumerate(joint_actions))
        return jammer_channel * (self.num_channels ** self.num_users) + joint_action_idx

    def get_action(self, state_idx, user):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_channels)
        else:
            return np.argmax(self.q_tables[user][state_idx])

    def coordinate(self, state_idx):
        jammer_channel = state_idx // (self.num_channels ** self.num_users)
        best_joint_action = None
        best_q_sum = -float("inf")
        if self.num_users == 1:
            for a1 in range(self.num_channels):
                joint_action = (a1,)
                joint_state = (jammer_channel, joint_action)
                joint_state_idx = self.state_to_index(joint_state)
                q_sum = self.q_tables[0][joint_state_idx][a1]
                if q_sum > best_q_sum:
                    best_q_sum = q_sum
                    best_joint_action = joint_action
        else:
            for a1 in range(self.num_channels):
                for a2 in range(self.num_channels):
                    joint_action = (a1, a2)
                    joint_state = (jammer_channel, joint_action)
                    joint_state_idx = self.state_to_index(joint_state)
                    q_sum = sum(self.q_tables[user][joint_state_idx][joint_action[user]] for user in range(self.num_users))
                    if q_sum > best_q_sum:
                        best_q_sum = q_sum
                        best_joint_action = joint_action
        return list(best_joint_action)

    def learn(self, T=T):
        total_rewards = [0] * self.num_users
        total_packets = [0] * self.num_users
        for t in range(T):
            state = self.env.get_state()
            state_idx = self.state_to_index(state)
            actions = self.coordinate(state_idx)
            next_state, rewards, packets = self.env.step(actions)
            next_state_idx = self.state_to_index(next_state)
            for user in range(self.num_users):
                q_value = self.q_tables[user][state_idx][actions[user]]
                max_next_q = np.max(self.q_tables[user][next_state_idx])
                self.q_tables[user][state_idx][actions[user]] = (1 - self.learning_rate) * q_value + self.learning_rate * (rewards[user] + self.discount_factor * max_next_q)
                total_rewards[user] += rewards[user]
                total_packets[user] += packets[user]
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if (t + 1) % step == 0:
                avg_throughput = [total_packets[user] / (t + 1) for user in range(self.num_users)]
                print(f"Iteration {t + 1}, CMAA Rewards: {total_rewards}, Packets: {total_packets}, Avg Throughput: {avg_throughput}, Actions: {actions}, Jammer Channel: {self.env.jammer_channel}")
        return total_rewards, total_packets