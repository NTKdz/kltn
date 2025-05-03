from calendar import c
from parameters import *
import numpy as np
import random
from scipy.stats import poisson


class Environment:
    def __init__(self, num_users=num_users, num_channels=num_channels, backscatter=True):
        self.num_users = num_users
        self.num_channels = num_channels
        self.jammer_state = 0
        self.data_states = [0] * num_users
        self.energy_states = [0] * num_users
        self.time_slot = 0
        self.current_d_bj = None
        self.current_d_ra = None
        self.current_e_hj = None
        self.backscatter = backscatter

    def get_state(self):
        avg_d_queue = np.mean(self.data_states) / d_queue_size
        avg_e_queue = np.mean(self.energy_states) / e_queue_size

        return np.array([self.jammer_state, self.time_slot] + self.data_states +
                        self.energy_states + [avg_d_queue] +
                        [avg_e_queue])

    def get_discrete_state(self, user_idx):
        def discretize(value):
            if value < 0.1667:
                return 0
            elif value < 0.3333:
                return 1
            elif value < 0.5:
                return 2
            elif value < 0.6667:
                return 3
            elif value < 0.8333:
                return 4
            else:
                return 5

        avg_q = np.mean(self.data_states) / d_queue_size
        avg_e = np.mean(self.energy_states) / e_queue_size

        q_bin = discretize(avg_q)
        e_avg_bin = discretize(avg_e)

        data_state = self.data_states[user_idx]
        energy_state = self.energy_states[user_idx]
        # q_bin = min(5, max(0, int(q_bin)))
        # e_avg_bin = min(5, max(0, int(e_avg_bin)))
        state = (
            user_idx * 2 * d_queue_size * e_queue_size * 6 * 6 +
            self.jammer_state * d_queue_size * e_queue_size * 6 * 6 +
            data_state * e_queue_size * 6 * 6 +
            energy_state * 6 * 6 +
            q_bin * 6 +
            e_avg_bin
        )
        return state

    def greedy_action(self, user_idx):
        possible_actions = self.get_possible_actions(user_idx)
        if user_idx == self.time_slot:  # Active user
            if self.jammer_state == 0 and 1 in possible_actions:
                return 1  # Active transmit when jammer idle
            elif self.jammer_state == 1:
                if 3 in possible_actions:
                    return 3  # Backscatter if data available
                elif 2 in possible_actions:
                    return 2  # Harvest energy
        else:  # Inactive user
            if self.jammer_state == 1 and 2 in possible_actions:
                return 2  # Harvest energy when jammed
        return 0  # Default to idle

    def sinr_based_action(self, user_idx):
        possible_actions = self.get_possible_actions(user_idx)
        if user_idx == self.time_slot:  # Active user
            if self.jammer_state == 0 and 1 in possible_actions:
                return 1  # Active transmit when jammer idle (high SINR)
            elif self.jammer_state == 1:
                # Use stored jamming power level (SINR proxy)
                d_bj = self.current_d_bj if self.current_d_bj is not None else random.choices(
                    d_bj_arr, nu_p, k=1)[0]
                d_ra = self.current_d_ra if self.current_d_ra is not None else random.choices(
                    dt_ra_arr, nu_p, k=1)[0]

                curr = []
                transmit = False
                for action in possible_actions:
                    if action in [3, 4, 5, 6]:
                        curr.append(action)
                        transmit = True
                if transmit:
                    return random.choice(curr)  # Randomly choose among actions
                else:
                    return 2  # Harvest energy if no other actions available
        else:  # Inactive user
            if self.jammer_state == 1 and 2 in possible_actions:
                return 2  # Harvest energy when jammed
        return 0  # Default to idle

    def random_action_strategy(self, user_idx):
        possible_actions = self.get_possible_actions(user_idx)
        if user_idx == self.time_slot:  # Active user
            return random.choice(possible_actions)  # Random action for
        else:
            actions = [0]
            if 2 in possible_actions:
                actions.append(2)

    def get_possible_actions(self, user_idx):
        list_actions = [0]  # Idle always possible
        if self.jammer_state == 0 and self.data_states[user_idx] > 0 and self.energy_states[user_idx] >= e_t:
            list_actions.append(1)  # Active transmit
        if self.jammer_state == 1:
            list_actions.append(2)
            if self.data_states[user_idx] > 0 and self.backscatter:
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
                rewards[i] = self.current_e_hj if self.current_e_hj is not None else random.choices(
                    e_hj_arr, nu_p, k=1)[0]
            elif action == 3 and self.jammer_state == 1:
                d_bj = self.current_d_bj if self.current_d_bj is not None else random.choices(
                    d_bj_arr, nu_p, k=1)[0]
                max_rate = min(b_dagger, self.data_states[i])
                rewards[i] = min(d_bj, self.data_states[i])
                if max_rate > rewards[i]:
                    losses[i] = max_rate - rewards[i]
            elif action in [4, 5, 6] and self.jammer_state == 1:
                max_ra = self.current_d_ra if self.current_d_ra is not None else random.choices(
                    dt_ra_arr, nu_p, k=1)[0]
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
        energy_used = 0

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
                    energy_used += rewards[i] * e_t
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
        return total_reward, next_state, rewards, packets_arrived, packets_lost, packets_lost_by_transmit, energy_used
