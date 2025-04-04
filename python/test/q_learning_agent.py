# q_learning_agent.py
# Q-learning agent for Document 3 (Chapter 4)

from environment_doc3 import environment_doc3
import numpy as np
from parameters import *

class q_learning_agent:
    def __init__(self, num_users=num_users, num_channels=num_channels):
        self.env = environment_doc3(num_users, num_channels)
        self.num_users = num_users
        self.num_channels = num_channels
        self.num_states = 2 * (d_queue_size + 1) * (e_queue_size + 1) * num_channels
        self.q_tables = [np.zeros((self.num_states, num_actions_doc3 * num_channels)) for _ in range(num_users)]
        self.learning_rate = learning_rate_Q
        self.discount_factor = gamma_Q
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999

    def state_to_index(self, state):
        jammer_state, data_state, energy_state, channel = state
        return (jammer_state * (d_queue_size + 1) * (e_queue_size + 1) * self.num_channels +
                data_state * (e_queue_size + 1) * self.num_channels +
                energy_state * self.num_channels +
                channel)

    def action_to_index(self, action, channel):
        return action * self.num_channels + channel

    def learn(self, T=T):
        total_rewards = [0] * self.num_users
        total_packets = [0] * self.num_users
        for t in range(T):
            # Get actions for each user
            actions = []
            for user in range(self.num_users):
                state = self.env.get_state(user)
                state_idx = self.state_to_index(state)
                possible_actions = self.env.get_possible_action(user)
                if np.random.random() < self.epsilon:
                    action = possible_actions[np.random.randint(len(possible_actions))]
                else:
                    max_q = -float("inf")
                    best_action = possible_actions[0]
                    for action_channel in possible_actions:
                        action_idx = self.action_to_index(action_channel[0], action_channel[1])
                        if self.q_tables[user][state_idx][action_idx] > max_q:
                            max_q = self.q_tables[user][state_idx][action_idx]
                            best_action = action_channel
                    action = best_action
                actions.append(action)

            # Perform actions
            rewards, packets = self.env.perform_action(actions)

            # Update Q-tables
            for user in range(self.num_users):
                state = self.env.get_state(user)
                state_idx = self.state_to_index(state)
                action_idx = self.action_to_index(actions[user][0], actions[user][1])
                next_state = self.env.get_state(user)
                next_state_idx = self.state_to_index(next_state)
                possible_next_actions = self.env.get_possible_action(user)
                max_next_q = -float("inf")
                for action_channel in possible_next_actions:
                    next_action_idx = self.action_to_index(action_channel[0], action_channel[1])
                    if self.q_tables[user][next_state_idx][next_action_idx] > max_next_q:
                        max_next_q = self.q_tables[user][next_state_idx][next_action_idx]
                self.q_tables[user][state_idx][action_idx] = (1 - self.learning_rate) * self.q_tables[user][state_idx][action_idx] + self.learning_rate * (rewards[user] + self.discount_factor * max_next_q)
                total_rewards[user] += rewards[user]
                total_packets[user] += packets[user]

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if (t + 1) % step == 0:
                print(f"Iteration {t + 1}, Q-learning Rewards: {total_rewards}, Packets: {total_packets}")
        return total_rewards, total_packets