# environment.py
import random
from parameters import *
import numpy as np
from scipy.stats import poisson

class Environment:
    def __init__(self, num_users):
        self.num_users = num_users
        self.num_channels = num_channels
        self.data_states = [0] * num_users
        self.energy_states = [0] * num_users
        self.jammer_state = 0  # 0 = off, 1 = on (Document 1 style)
        self.jammed_channel = 0  # Current jammed channel (Document 2 style)

    def get_state(self):
        return np.array([self.jammed_channel] + self.data_states + self.energy_states)

    def get_possible_actions(self, user_idx):
        energy = self.energy_states[user_idx]
        data = self.data_states[user_idx]
        possible = []
        for base in range(7):
            if base == 0 or (base == 2 and energy < e_queue_size) or \
               (base in [1, 4, 5, 6] and energy >= e_t and data > 0) or \
               (base == 3 and data > 0):
                for ch in range(num_channels):
                    possible.append(base * num_channels + ch)
        return possible

    def active_transmit(self, user_idx, rate):
        if self.jammed_channel == (user_idx % num_channels):  # Simplified channel check
            return 0
        return min(rate, self.data_states[user_idx])

    def calculate_reward(self, actions):
        rewards = [0] * self.num_users
        losses = [0] * self.num_users
        channel_users = [[] for _ in range(num_channels)]  # Track users per channel
        for i, action in enumerate(actions):
            channel = action % num_channels
            base_action = action // num_channels
            channel_users[channel].append(i)

        for i, action in enumerate(actions):
            channel = action % num_channels
            base_action = action // num_channels
            active_users_on_channel = len(channel_users[channel])
            interference = active_users_on_channel > 1

            if base_action == 0:
                rewards[i] = 0
            elif base_action == 1:
                rewards[i] = self.active_transmit(i, d_t)
                if interference:
                    rewards[i] /= active_users_on_channel
            elif base_action == 2:
                rewards[i] = random.choices(e_hj_arr, nu_p, k=1)[0]
            elif base_action == 3:
                d_bj = random.choices(d_bj_arr, nu_p, k=1)[0]
                max_rate = min(b_dagger, self.data_states[i])
                rewards[i] = min(d_bj, self.data_states[i])
                if max_rate > rewards[i]:
                    losses[i] = max_rate - rewards[i]
            elif base_action in [4, 5, 6]:
                max_ra = random.choices(dt_ra_arr, nu_p, k=1)[0]
                idx = base_action - 4
                rewards[i] = self.active_transmit(i, dt_ra_arr[idx])
                if interference:
                    rewards[i] /= active_users_on_channel
                if dt_ra_arr[idx] > max_ra:
                    losses[i] = rewards[i]
                    rewards[i] = 0

        return rewards, losses

    def step(self, actions):
        rewards, losses = self.calculate_reward(actions)
        total_reward = 0
        packets_arrived = [0] * self.num_users
        packets_lost = [0] * self.num_users

        for i, action in enumerate(actions):
            base_action = action // num_channels
            channel = action % num_channels
            reward = rewards[i]
            loss = losses[i]
            if base_action == 1 or base_action in [4, 5, 6]:
                if reward > 0:
                    self.data_states[i] = max(0, self.data_states[i] - reward)
                    self.energy_states[i] = max(0, self.energy_states[i] - reward * e_t)
            elif base_action == 2:
                if self.energy_states[i] < e_queue_size:
                    self.energy_states[i] = min(e_queue_size, self.energy_states[i] + reward)
                rewards[i] = 0
            elif base_action == 3:
                max_rate = min(b_dagger, self.data_states[i])
                self.data_states[i] = max(0, self.data_states[i] - max_rate)
                packets_lost[i] += loss

            total_reward += rewards[i]
            packets_lost[i] += loss

        for i in range(self.num_users):
            data_arrive = poisson.rvs(mu=arrival_rate)
            packets_arrived[i] = data_arrive
            new_data_state = self.data_states[i] + data_arrive
            if new_data_state > d_queue_size:
                packets_lost[i] += (new_data_state - d_queue_size)
            self.data_states[i] = min(d_queue_size, new_data_state)

        # Sweep jammer (Document 2 style)
        self.jammed_channel = (self.jammed_channel + 1) % num_channels

        next_state = self.get_state()
        return total_reward, next_state, rewards, packets_arrived, packets_lost