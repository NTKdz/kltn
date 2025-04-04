# environment_doc3.py
# Environment for Document 3 (Chapter 4)

from parameters import *
import numpy as np
from scipy.stats import poisson

class environment_doc3:
    def __init__(self, num_users=num_users, num_channels=num_channels):
        self.num_users = num_users
        self.num_channels = num_channels
        self.jammer_state = 0  # 0: idle, 1: active
        self.jammer_power = 0  # Power level of the jammer
        self.data_states = [0] * num_users  # Data queue for each user
        self.energy_states = [0] * num_users  # Energy queue for each user
        self.user_channels = [0] * num_users  # Current channel for each user

    def get_state(self, user):
        return (self.jammer_state, self.data_states[user], self.energy_states[user], self.user_channels[user])

    def get_state_deep(self, user):
        return np.array([self.jammer_state, self.data_states[user], self.energy_states[user], self.user_channels[user]])

    def get_possible_action(self, user):
        list_actions = [(0, self.user_channels[user])]  # Stay idle
        if self.num_channels > 1:
            for channel in range(self.num_channels):
                list_actions.append((1, channel))  # Select channel and transmit
                list_actions.append((2, channel))  # Select channel and harvest
                list_actions.append((3, channel))  # Select channel and backscatter
                list_actions.extend([(4 + m, channel) for m in range(3)])  # Rate adaptation
        else:
            if self.jammer_state == 0 and self.data_states[user] > 0 and self.energy_states[user] >= e_t:
                list_actions.append((1, 0))  # Actively transmit
            if self.jammer_state == 1:
                list_actions.append((2, 0))  # Harvest energy
                if self.data_states[user] > 0:
                    list_actions.append((3, 0))  # Backscatter
                    if self.energy_states[user] >= e_t:
                        list_actions.extend([(4 + m, 0) for m in range(3)])  # Rate adaptation
        return list_actions

    def perform_action(self, actions):
        rewards = [0] * self.num_users
        packets_transmitted = [0] * self.num_users
        for user in range(self.num_users):
            action, channel = actions[user]
            self.user_channels[user] = channel
            # Check for mutual interference
            collision = False
            for other_user in range(self.num_users):
                if other_user != user and self.user_channels[other_user] == channel:
                    collision = True
                    break
            if collision and action in [1, 4, 5, 6]:  # Transmit or rate adaptation
                rewards[user] = 0
                continue

            # Process action
            if action == 0:
                rewards[user] = 0
            elif action == 1:
                num_transmitted = self.active_transmit(user, d_t)
                rewards[user] = num_transmitted
                packets_transmitted[user] = num_transmitted
                self.data_states[user] -= num_transmitted
                self.energy_states[user] -= num_transmitted * e_t
            elif action == 2:
                rewards[user] = 0
                harvested = np.random.choice(e_hj_arr, p=nu_p)
                self.energy_states[user] += harvested
                if self.energy_states[user] > e_queue_size:
                    self.energy_states[user] = e_queue_size
            elif action == 3:
                d_bj = np.random.choice(d_bj_arr, p=nu_p)
                max_rate = min(b_dagger, self.data_states[user])
                num_transmitted = min(d_bj, self.data_states[user])
                rewards[user] = num_transmitted
                packets_transmitted[user] = num_transmitted
                self.data_states[user] -= max_rate
            elif action in [4, 5, 6]:
                max_ra = np.random.choice(dt_ra_arr, p=nu_p)
                num_transmitted = self.active_transmit(user, dt_ra_arr[action - 4])
                if dt_ra_arr[action - 4] > max_ra:
                    rewards[user] = 0
                else:
                    rewards[user] = num_transmitted
                    packets_transmitted[user] = num_transmitted
                    self.data_states[user] -= num_transmitted
                    self.energy_states[user] -= num_transmitted * e_t

            # Data arrival
            data_arrive = poisson.rvs(mu=arrival_rate, size=1)[0]
            self.data_states[user] += data_arrive
            if self.data_states[user] > d_queue_size:
                self.data_states[user] = d_queue_size

        # Update jammer state
        if self.jammer_state == 0 and np.random.random() <= 1 - nu:
            self.jammer_state = 1
            self.jammer_power = np.random.choice([1, 2, 3], p=nu_p)
        elif self.jammer_state == 1 and np.random.random() <= nu:
            self.jammer_state = 0
            self.jammer_power = 0

        return rewards, packets_transmitted

    def active_transmit(self, user, max_packets):
        num_transmitted = 0
        if 0 < self.data_states[user] <= max_packets:
            if self.energy_states[user] >= e_t * self.data_states[user]:
                num_transmitted = self.data_states[user]
            elif self.energy_states[user] >= e_t:
                num_transmitted = self.energy_states[user] // e_t
        else:
            if self.energy_states[user] >= e_t * max_packets:
                num_transmitted = max_packets
            elif self.energy_states[user] >= e_t:
                num_transmitted = self.energy_states[user] // e_t
        return num_transmitted