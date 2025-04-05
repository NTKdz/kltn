# # environment.py
# from parameters import *
# import numpy as np
# import random
# from scipy.stats import poisson

# class Environment:
#     def __init__(self, num_users=num_users):
#         self.num_users = num_users
#         self.jammer_state = 0
#         self.data_states = [0] * num_users
#         self.energy_states = [0] * num_users

#     def get_state(self):
#         return np.array([self.jammer_state] + self.data_states + self.energy_states)

#     def get_possible_actions(self, user_idx):
#         list_actions = [0]
#         if self.jammer_state == 0 and self.data_states[user_idx] > 0 and self.energy_states[user_idx] >= e_t:
#             list_actions.append(1)
#         if self.jammer_state == 1:
#             list_actions.append(2)
#             if self.data_states[user_idx] > 0:
#                 list_actions.append(3)
#             if self.energy_states[user_idx] >= e_t:
#                 list_actions.extend([4, 5, 6])
#         return list_actions

#     def active_transmit(self, user_idx, max_packets):
#         num_transmitted = 0
#         if 0 < self.data_states[user_idx] < max_packets:
#             if self.energy_states[user_idx] >= e_t * self.data_states[user_idx]:
#                 num_transmitted = self.data_states[user_idx]
#             elif self.energy_states[user_idx] >= e_t:
#                 num_transmitted = self.energy_states[user_idx] // e_t
#         else:
#             if self.energy_states[user_idx] >= e_t * max_packets:
#                 num_transmitted = max_packets
#             elif self.energy_states[user_idx] >= e_t:
#                 num_transmitted = self.energy_states[user_idx] // e_t
#         return num_transmitted

#     def calculate_reward(self, actions):
#         rewards = [0] * self.num_users
#         losses = [0] * self.num_users
#         active_users = sum(1 for a in actions if a in [1, 4, 5, 6])
#         interference = active_users > 1

#         for i, action in enumerate(actions):
#             if action == 0:
#                 rewards[i] = 0
#             elif action == 1:
#                 rewards[i] = self.active_transmit(i, d_t)
#                 if interference:
#                     rewards[i] /= 2
#             elif action == 2:
#                 rewards[i] = random.choices(e_hj_arr, nu_p, k=1)[0]
#             elif action == 3:
#                 d_bj = random.choices(d_bj_arr, nu_p, k=1)[0]
#                 max_rate = min(b_dagger, self.data_states[i])
#                 rewards[i] = min(d_bj, self.data_states[i])
#                 if max_rate > rewards[i]:
#                     losses[i] = max_rate - rewards[i]
#             elif action in [4, 5, 6]:
#                 max_ra = random.choices(dt_ra_arr, nu_p, k=1)[0]
#                 idx = action - 4
#                 rewards[i] = self.active_transmit(i, dt_ra_arr[idx])
#                 if interference:
#                     rewards[i] /= 2
#                 if dt_ra_arr[idx] > max_ra:
#                     losses[i] = rewards[i]
#                     rewards[i] = 0
#         return rewards, losses

#     def step(self, actions):
#         rewards, losses = self.calculate_reward(actions)
#         total_reward = 0
#         for i, (action, reward, loss) in enumerate(zip(actions, rewards, losses)):
#             if action == 1:
#                 self.data_states[i] -= reward
#                 self.energy_states[i] -= reward * e_t
#             elif action == 2:
#                 if self.energy_states[i] < e_queue_size:
#                     self.energy_states[i] += reward
#                     if self.energy_states[i] > e_queue_size:
#                         self.energy_states[i] = e_queue_size
#                 rewards[i] = 0  # Reward is packets, not energy
#             elif action == 3:
#                 max_rate = min(b_dagger, self.data_states[i])
#                 self.data_states[i] -= max_rate
#             elif action in [4, 5, 6]:
#                 if reward > 0:
#                     self.data_states[i] -= reward
#                     self.energy_states[i] -= reward * e_t
#                 self.data_states[i] -= loss
#                 self.energy_states[i] -= loss * e_t
#             total_reward += rewards[i]

#         for i in range(self.num_users):
#             data_arrive = poisson.rvs(mu=arrival_rate)
#             self.data_states[i] += data_arrive
#             if self.data_states[i] > d_queue_size:
#                 self.data_states[i] = d_queue_size

#         if self.jammer_state == 0:
#             if np.random.random() <= 1 - nu:
#                 self.jammer_state = 1
#         else:
#             if np.random.random() <= nu:
#                 self.jammer_state = 0

#         next_state = self.get_state()
#         return total_reward, next_state, rewards  # Return individual rewards too


# environment.py
from parameters import *
import numpy as np
import random
from scipy.stats import poisson


class Environment:
    def __init__(self, num_users=num_users):
        self.num_users = num_users
        self.jammer_state = 0
        self.data_states = [0] * num_users
        self.energy_states = [0] * num_users

    def get_state(self):
        return np.array([self.jammer_state] + self.data_states + self.energy_states)

    def get_discrete_state(self, user_idx):
        j = self.jammer_state
        d = self.data_states[user_idx]
        e = self.energy_states[user_idx]
        state_idx = j * (d_queue_size + 1) * (e_queue_size +
                                              1) + d * (e_queue_size + 1) + e
        assert 0 <= state_idx < num_states, f"Invalid state_idx {state_idx}"
        return state_idx

    def get_possible_actions(self, user_idx):
        list_actions = [0]
        if self.jammer_state == 0 and self.data_states[user_idx] > 0 and self.energy_states[user_idx] >= e_t:
            list_actions.append(1)
        if self.jammer_state == 1:
            list_actions.append(2)
            if self.data_states[user_idx] > 0:
                list_actions.append(3)
            if self.energy_states[user_idx] >= e_t:
                list_actions.extend([4, 5, 6])
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
        active_users = sum(1 for a in actions if a in [1, 3, 4, 5, 6])
        interference = active_users > 1
        for i, action in enumerate(actions):
            if action == 0:
                rewards[i] = 0
            elif action == 1:
                rewards[i] = self.active_transmit(i, d_t)
                if interference:
                    losses[i] = rewards[i] - rewards[i] / active_users
                    rewards[i] /= active_users
            elif action == 2:
                rewards[i] = random.choices(e_hj_arr, nu_p, k=1)[0]
            elif action == 3:
                d_bj = random.choices(d_bj_arr, nu_p, k=1)[0]
                max_rate = min(b_dagger, self.data_states[i])
                rewards[i] = min(d_bj, self.data_states[i])

                if max_rate > rewards[i]:
                    losses[i] = max_rate - rewards[i]

                if interference:
                    # rewards[i] /= active_users
                    losses[i] = rewards[i] - rewards[i] / active_users
                    rewards[i] /= active_users
            elif action in [4, 5, 6]:
                max_ra = random.choices(dt_ra_arr, nu_p, k=1)[0]
                idx = action - 4
                rewards[i] = self.active_transmit(i, dt_ra_arr[idx])
                if interference:
                    # rewards[i] /= active_users
                    losses[i] = rewards[i] - rewards[i] / active_users
                    rewards[i] /= active_users
                if dt_ra_arr[idx] > max_ra:
                    losses[i] = rewards[i]
                    rewards[i] = 0
        print(f"Active users: {active_users}, Action: {actions}, Reward: {rewards}, Loss: {losses}")

        return rewards, losses

    def step(self, actions):
        rewards, losses = self.calculate_reward(actions)
        total_reward = 0
        packets_arrived = [0] * self.num_users
        packets_lost = [0] * self.num_users
        for i, (action, reward, loss) in enumerate(zip(actions, rewards, losses)):
            if action == 1:
                self.data_states[i] = max(
                    0, self.data_states[i] - reward - loss)
                self.energy_states[i] = max(
                    0, self.energy_states[i] - reward * e_t)
            elif action == 2:
                if self.energy_states[i] < e_queue_size:
                    self.energy_states[i] = min(
                        e_queue_size, self.energy_states[i] + reward)
                rewards[i] = 0
            elif action == 3:
                max_rate = min(b_dagger, self.data_states[i])
                self.data_states[i] = max(0, self.data_states[i] - max_rate)
                packets_lost[i] += losses[i]
            elif action in [4, 5, 6]:
                if reward > 0:
                    self.data_states[i] = max(0, self.data_states[i] - reward)
                    self.energy_states[i] = max(
                        0, self.energy_states[i] - reward * e_t)
                self.data_states[i] = max(0, self.data_states[i] - loss)
                self.energy_states[i] = max(
                    0, self.energy_states[i] - loss * e_t)
                packets_lost[i] += losses[i]
            total_reward += rewards[i]

        for i in range(self.num_users):
            data_arrive = poisson.rvs(mu=arrival_rate)
            packets_arrived[i] = data_arrive
            new_data_state = self.data_states[i] + data_arrive
            if new_data_state > d_queue_size:
                packets_lost[i] += (new_data_state - d_queue_size)
            self.data_states[i] = min(d_queue_size, new_data_state)

        if self.jammer_state == 0:
            if np.random.random() <= 1 - nu:
                self.jammer_state = 1
        else:
            if np.random.random() <= nu:
                self.jammer_state = 0

        next_state = self.get_state()
        return total_reward, next_state, rewards, packets_arrived, packets_lost
