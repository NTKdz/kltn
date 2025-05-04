# qlearning.py
from typing import final
from environment import Environment
from parameters import *
import numpy as np
import random
import os
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(self, num_users=num_users, load_path=None):
        self.env = Environment(num_users)
        self.num_users = num_users
        # Initialize Q-table: [num_states, num_actions] for each user
        self.q_tables = [np.zeros((num_states, num_actions))
                         for _ in range(self.num_users)]
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.count_changes = []

        if load_path and os.path.exists(load_path):
            try:
                self.load_q_tables(load_path)
                print(f"Loaded Q-tables from {load_path}")
            except Exception as e:
                print(
                    f"Failed to load Q-tables: {e}. Initializing new Q-tables.")

    def get_action(self, user_idx):
        discrete_state = self.env.get_discrete_state(user_idx)
        possible_actions = self.env.get_possible_actions(user_idx)
        if random.random() < self.epsilon:
            action = random.choice(possible_actions)
            q_value = self.q_tables[user_idx][discrete_state, action]
        else:
            q_values = self.q_tables[user_idx][discrete_state]
            action = max(possible_actions, key=lambda a: q_values[a])
            q_value = q_values[action]
        return action, q_value

    def update_q_table(self, state, action, reward, next_state, user_idx):
        discrete_state = state
        next_discrete_state = self.env.get_discrete_state(user_idx)
        # print(f"User {user_idx}, State: {discrete_state}, Action: {action}, Reward: {reward}, Next State: {next_discrete_state}")
        possible_actions = self.env.get_possible_actions(user_idx)
        current_q = self.q_tables[user_idx][discrete_state, action]
        max_next_q = max(
            self.q_tables[user_idx][next_discrete_state, a] for a in possible_actions)
        # Q-learning update rule
        self.q_tables[user_idx][discrete_state, action] = current_q + learning_rate_Q * (
            reward + gamma_Q * max_next_q - current_q
        )

    def save_q_tables(self, path):
        np.savez(path, q_tables=self.q_tables, epsilon=self.epsilon)

    def load_q_tables(self, path):
        checkpoint = np.load(path)
        self.q_tables = list(checkpoint['q_tables'])
        self.epsilon = checkpoint['epsilon']

    def plot_progress(self, total_history, per_user_history, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(total_history, label='Total Avg Reward', color='blue')
        for u in range(self.num_users):
            plt.plot(
                per_user_history[u], label=f'User {u+1} Avg Reward', linestyle='--')
        plt.xlabel('Iteration (Time Slot)')
        plt.ylabel('Average Reward')
        plt.title('Multi-User Training Progression (Q-Learning)')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Saved training progress plot to {filename}")

    def log_to_file(self, filename, iteration, avg_total, avg_per_user, packet_loss_ratios, count_change, packet_lost_by_transmit, energy_used):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        header = "Iteration\tAvg_Total_Reward\t" + "\t".join([f"Avg_Reward_User_{u}" for u in range(self.num_users)]) + \
                 "\t" + "\t".join([f"Packet_Loss_Ratio_User_{u}" for u in range(self.num_users)]) + \
                 "\t" + "\t".join([f"Packet_Loss_By_Transmit_Ratio_User_{u}" for u in range(self.num_users)]) + \
                 "\tEnergy_Used"+"\tCount_Change\n"
        data_line = f"{iteration}\t{avg_total:.4f}\t" + \
            "\t".join([f"{r:.4f}" for r in avg_per_user]) + "\t" + \
            "\t".join([f"{r:.4f}" for r in packet_loss_ratios]) + "\t" + \
            "\t".join([f"{r:.4f}" for r in packet_lost_by_transmit]) + \
            f"\t{energy_used}"+f"\t{count_change}\n"
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(header)
        with open(filename, 'a') as f:
            f.write(data_line)

    def train(self, save_path="multi_user_qlearning.npz",
              plot_path="multi_user_training_progress_qlearning.png",
              log_path="log/multi_user_training_data_qlearning.txt"):
        total_reward = 0
        per_user_totals = [0] * self.num_users
        total_packets_arrived = [0] * self.num_users
        total_packets_lost = [0] * self.num_users
        total_packets_lost_by_transmit = [0] * self.num_users
        total_history = []
        total_energy_used = []
        per_user_history = [[] for _ in range(self.num_users)]
        state = []
        for u in range(self.num_users):
            state.append(self.env.get_discrete_state(u))
        transmission_actions = [1, 3, 4, 5, 6]
        data_state_history = [[] for _ in range(self.num_users)]
        energy_state_history = [[] for _ in range(self.num_users)]
        count_change = 0

        for i in range(T):
            actions = [0] * self.num_users
            current_agent = self.env.time_slot

            # Step 1: Evaluate the current time slot's agent
            possible_actions = self.env.get_possible_actions(current_agent)
            current_action, _ = self.get_action(current_agent)

            # Step 2: If current agent selects a transmission action, they transmit
            if current_action in transmission_actions:
                actions[current_agent] = current_action
                for u in range(self.num_users):
                    if u != current_agent:
                        possible_actions = [a for a in self.env.get_possible_actions(
                            u) if a not in transmission_actions]
                        if not possible_actions:
                            # Ensure idle is always possible
                            possible_actions = [0]
                        if random.random() < self.epsilon:
                            action = random.choice(possible_actions)
                        else:
                            discrete_state = self.env.get_discrete_state(u)
                            action = max(
                                possible_actions, key=lambda a: self.q_tables[u][discrete_state, a])
                        actions[u] = action
            else:
                count_change += 1
                # Step 3: Current agent does not transmit, evaluate other agents
                candidates = []
                for u in range(self.num_users):
                    if u == current_agent:
                        continue
                    possible_transmissions = [
                        a for a in self.env.get_possible_actions(u) if a in transmission_actions]
                    if possible_transmissions:
                        action, q_value = self.get_action(u)
                        if action in transmission_actions:
                            # Optional: Add queue bonus for fair comparison with DQN
                            # q_value += self.env.data_states[u] / d_queue_size
                            candidates.append((u, action, q_value))

                # Step 4: Select the best candidate to transmit (if any)
                if candidates:
                    if random.random() < self.epsilon:
                        best_user, best_action, _ = random.choice(candidates)
                    else:
                        best_user, best_action, _ = max(
                            candidates, key=lambda x: x[2])
                    actions[best_user] = best_action
                    for u in range(self.num_users):
                        if u != best_user:
                            possible_actions = [a for a in self.env.get_possible_actions(
                                u) if a not in transmission_actions]
                            if not possible_actions:
                                # Ensure idle is always possible
                                possible_actions = [0]
                            if random.random() < self.epsilon:
                                action = random.choice(possible_actions)
                            else:
                                discrete_state = self.env.get_discrete_state(u)
                                action = max(
                                    possible_actions, key=lambda a: self.q_tables[u][discrete_state, a])
                            actions[u] = action
                else:
                    for u in range(self.num_users):
                        possible_actions = [a for a in self.env.get_possible_actions(
                            u) if a not in transmission_actions]
                        if not possible_actions:
                            # Ensure idle is always possible
                            possible_actions = [0]
                        if random.random() < self.epsilon:
                            action = random.choice(possible_actions)
                        else:
                            discrete_state = self.env.get_discrete_state(u)
                            action = max(
                                possible_actions, key=lambda a: self.q_tables[u][discrete_state, a])
                        actions[u] = action

            # Environment step
            total_reward_step, next_state, individual_rewards, packets_arrived, packets_lost, packets_lost_by_transmit, energy_used = self.env.step(
                actions)
            total_reward += total_reward_step
            total_energy_used.append(energy_used)
            # Update Q-tables for all users
            for u in range(self.num_users):
                self.update_q_table(
                    state[u], actions[u], individual_rewards[u], next_state, u)

            state.clear()
            # Logging and tracking
            for u in range(self.num_users):
                state.append(self.env.get_discrete_state(u))
                per_user_totals[u] += individual_rewards[u]
                total_packets_arrived[u] += individual_rewards[u]
                total_packets_lost[u] += packets_lost[u]
                total_packets_lost_by_transmit[u] += packets_lost_by_transmit[u]
                per_user_history[u].append(per_user_totals[u] / (i + 1))
                data_state_history[u].append(self.env.data_states[u])
                energy_state_history[u].append(self.env.energy_states[u])
            total_history.append(total_reward / (i + 1))

            if (i + 1) % step == 0:
                avg_data_state = [np.mean(data_state_history[u])
                                  for u in range(self.num_users)]
                avg_energy_state = [np.mean(energy_state_history[u])
                                    for u in range(self.num_users)]
                print(f"Avg Data States: {[f'{s:.4f}' for s in avg_data_state]}, "
                      f"Avg Energy States: {[f'{s:.4f}' for s in avg_energy_state]}, "
                      f"Epsilon: {self.epsilon:.4f}, Count Change: {count_change}")
                self.count_changes.append(count_change)
                count_change = 0
                avg_total = np.mean(total_history[-step:])
                avg_per_user = [np.mean(per_user_history[u][-step:])
                                for u in range(self.num_users)]
                packet_loss_ratios = [total_packets_lost[u] / (total_packets_arrived[u] + total_packets_lost[u]) if total_packets_arrived[u] > 0 else 0
                                      for u in range(self.num_users)]
                packet_loss_by_transmit_ratios = [total_packets_lost_by_transmit[u] / (total_packets_arrived[u] + total_packets_lost_by_transmit[u]) if total_packets_arrived[u] > 0 else 0
                                                  for u in range(self.num_users)]
                avg_energy_used = np.mean(total_energy_used[-step:])
                print(f"Time Slot {i + 1}, Avg Total Reward: {avg_total:.4f}, "
                      f"Avg Per-User Rewards: {[f'{r:.4f}' for r in avg_per_user]}, "
                      f"Packet Loss Ratios: {[f'{r:.4f}' for r in packet_loss_ratios]}, "
                      f"Packet Loss Ratios By Transmit: {[f'{r:.4f}' for r in packet_loss_by_transmit_ratios]}"
                      f"Energy Efficiency: {avg_total/avg_energy_used:.4f}")
                # self.save_q_tables(save_path)
                self.log_to_file(log_path, i + 1, avg_total, avg_per_user,
                                 packet_loss_ratios, count_change, packet_loss_by_transmit_ratios, sum(total_energy_used[-step:]))

                if (i + 1) == 100000:
                    self.plot_progress(total_history, per_user_history,
                                       f"plot/test_q1/multi_user_progress_at_100k_qlearning_{num_users}_rate_{arrival_rate}_eps_{self.epsilon_decay}2.png")
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)

        avg_per_user_final = [per_user_totals[u] /
                              T for u in range(self.num_users)]
        final_packet_loss_ratios = [total_packets_lost[u] / (total_packets_arrived[u] + total_packets_lost[u]) if total_packets_arrived[u] > 0 else 0
                                    for u in range(self.num_users)]
        final_packet_loss_ratios_by_transmit = [total_packets_lost_by_transmit[u] / (total_packets_arrived[u] + total_packets_lost_by_transmit[u]) if total_packets_arrived[u] > 0 else 0
                                                for u in range(self.num_users)]
        final_energy_used = np.mean(total_energy_used[-step:])
        # self.save_q_tables(save_path)
        self.plot_progress(total_history, per_user_history, plot_path)
        self.log_to_file(log_path, T, total_reward / T, avg_per_user_final,
                         final_packet_loss_ratios, count_change, final_packet_loss_ratios_by_transmit, sum(total_energy_used[-step:]))
        print(
            f"Final Packet Loss Ratios: {[f'{r:.4f}' for r in final_packet_loss_ratios]}")
        print(
            f"Final Packet Loss Ratios By Transmit: {[f'{r:.4f}' for r in final_packet_loss_ratios_by_transmit]}")
        return total_reward / T, avg_per_user_final


if __name__ == "__main__":
    agent = QLearningAgent(load_path=None)
    avg_total_multi, avg_per_user_multi = agent.train(
        save_path=f"checkpoint/test_q1/multi_user_qlearning_{num_users}_rate_{arrival_rate}_eps_{agent.epsilon_decay}2.npz",
        plot_path=f"plot/test_q1/multi_user_training_progress_qlearning_{num_users}_rate_{arrival_rate}_eps_{agent.epsilon_decay}2.png",
        log_path=f"log/test_q1/multi_user_training_data_qlearning_{num_users}_rate_{arrival_rate}_eps_{agent.epsilon_decay}2.txt"
    )
    print(f"Multi-user total average reward: {avg_total_multi:.4f}")
    print(
        f"Multi-user per-user average rewards: {[f'{r:.4f}' for r in avg_per_user_multi]}")
    print(
        f"Multi-user average per-user reward (mean): {np.mean(avg_per_user_multi):.4f}")
