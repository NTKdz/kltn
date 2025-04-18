from parameters import *
import numpy as np
import random
from scipy.stats import poisson


class Environment:
    def __init__(self, num_users=num_users, num_channels=num_channels):
        self.num_users = num_users
        self.num_channels = num_channels
        self.jammer_state = 0
        self.data_states = [0] * num_users
        self.energy_states = [0] * num_users
        self.time_slot = 0

    def get_state(self):
        # Include average queue size of other users (normalized)
        avg_other_queue = np.mean(self.data_states) / d_queue_size
        return np.array([self.jammer_state, self.time_slot] +
                        [d / d_queue_size for d in self.data_states] +
                        [e / e_queue_size for e in self.energy_states] +
                        [avg_other_queue])
        # return np.array([self.jammer_state, self.time_slot] + self.data_states +
        #                 self.energy_states)

    def get_discrete_state(self, user_idx):
        # Discretize the continuous state components
        j = self.jammer_state  # Binary: 0 or 1
        t = self.time_slot  # Discrete: 0 to num_users-1
        d = self.data_states[user_idx] / d_queue_size  # Normalized: [0, 1]
        e = self.energy_states[user_idx] / e_queue_size  # Normalized: [0, 1]
        avg_q = np.mean(self.data_states) / d_queue_size  # Normalized: [0, 1]

        # Bin normalized values into 4 levels
        def discretize(value):
            if value < 0.25:
                return 0
            elif value < 0.5:
                return 1
            elif value < 0.75:
                return 2
            else:
                return 3

        d_bin = discretize(d)  # 0, 1, 2, or 3
        e_bin = discretize(e)  # 0, 1, 2, or 3
        q_bin = discretize(avg_q)  # 0, 1, 2, or 3

        # Compute state index
        state_idx = (j * self.num_users * 4 * 4 * 4 +
                     t * 4 * 4 * 4 +
                     d_bin * 4 * 4 +
                     e_bin * 4 +
                     q_bin)
        total_states = 2 * self.num_users * 4 * 4 * 4  # 640 for num_users=5
        assert 0 <= state_idx < total_states, f"Invalid state_idx {state_idx}"
        return state_idx

    def get_possible_actions(self, user_idx):
        list_actions = [0]  # Idle always possible
        if self.jammer_state == 0 and self.data_states[user_idx] > 0 and self.energy_states[user_idx] >= e_t:
            list_actions.append(1)  # Active transmit
        if self.jammer_state == 1:
            list_actions.append(2)  # Harvest energy
            if self.data_states[user_idx] > 0:
                list_actions.append(3)  # Backscatter
            if self.energy_states[user_idx] >= e_t:
                list_actions.extend([4, 5, 6])  # Rate adaptation
        return list_actions

    def active_transmit(self, user_idx, max_packets):
        num_transmitted = 0
        if 0 < self.data_states[user_idx] < max_packets:
            if self.energy_states[user_idx] >= e_t * self.data_states[user_idx]:
                num_transmitted = self.data_states[user_idx]
            elif self.energy_states[user_idx] >= e_t:
                num_transmitted = self.energy_states[user_idx] // e_t
        else:
            if self.energy_states[user_idx] >= e_t * max_packets:
                num_transmitted = max_packets
            elif self.energy_states[user_idx] >= e_t:
                num_transmitted = self.energy_states[user_idx] // e_t
        return num_transmitted

    def calculate_reward(self, actions):
        rewards = [0] * self.num_users
        losses = [0] * self.num_users

        for i, action in enumerate(actions):
            if action == 0:
                rewards[i] = 0
            elif action == 1 and self.jammer_state == 0:
                rewards[i] = self.active_transmit(i, d_t)
            elif action == 2 and self.jammer_state == 1:
                rewards[i] = random.choices(e_hj_arr, nu_p, k=1)[0]
            elif action == 3 and self.jammer_state == 1:
                d_bj = random.choices(d_bj_arr, nu_p, k=1)[0]
                max_rate = min(b_dagger, self.data_states[i])
                rewards[i] = min(d_bj, self.data_states[i])
                if max_rate > rewards[i]:
                    losses[i] = max_rate - rewards[i]
            elif action in [4, 5, 6] and self.jammer_state == 1:
                max_ra = random.choices(dt_ra_arr, nu_p, k=1)[0]
                idx = action - 4
                rewards[i] = self.active_transmit(i, dt_ra_arr[idx])
                if dt_ra_arr[idx] > max_ra:
                    losses[i] = rewards[i]
                    rewards[i] = 0

        return rewards, losses

    def step(self, actions):
        rewards, losses = self.calculate_reward(actions)
        packets_arrived = [0] * self.num_users
        packets_lost = [0] * self.num_users
        packets_lost_by_transmit = [0] * self.num_users

        for i in range(self.num_users):
            if actions[i] == 1:
                self.data_states[i] = max(0, self.data_states[i] - rewards[i])
                self.energy_states[i] = max(
                    0, self.energy_states[i] - rewards[i] * e_t)
            elif actions[i] == 2:
                if self.energy_states[i] < e_queue_size:
                    self.energy_states[i] = min(
                        e_queue_size, self.energy_states[i] + rewards[i])
                rewards[i] = 0
            elif actions[i] == 3:
                max_rate = min(b_dagger, self.data_states[i])
                self.data_states[i] = max(0, self.data_states[i] - max_rate)
                packets_lost[i] += losses[i]
            elif actions[i] in [4, 5, 6]:
                if rewards[i] > 0:
                    self.data_states[i] = max(
                        0, self.data_states[i] - rewards[i])
                    self.energy_states[i] = max(
                        0, self.energy_states[i] - rewards[i] * e_t)
                packets_lost[i] += losses[i]

        # Data arrival
        for i in range(self.num_users):
            packets_lost_by_transmit[i] += packets_lost[i]
            data_arrive = poisson.rvs(mu=arrival_rate)
            packets_arrived[i] = data_arrive
            new_data_state = self.data_states[i] + data_arrive
            if new_data_state > d_queue_size:
                packets_lost[i] += (new_data_state - d_queue_size)
            self.data_states[i] = min(d_queue_size, new_data_state)

        # Jammer state update
        if self.jammer_state == 0:
            if np.random.random() <= 1 - nu:
                self.jammer_state = 1
        else:
            if np.random.random() <= nu:
                self.jammer_state = 0

        # Advance time slot
        self.time_slot = (self.time_slot + 1) % self.num_users
        total_reward = sum(rewards)
        next_state = self.get_state()
        return total_reward, next_state, rewards, packets_arrived, packets_lost, packets_lost_by_transmit
