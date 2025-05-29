import time
from typing import final
from environment import Environment
from parameters import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.fc5(x)
        return x


class DeepQLearningAgent:
    def __init__(self, num_users=num_users, input_dim=num_features, load_path=None, backscatter=True):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.env = Environment(num_users, backscatter=backscatter)
        self.num_users = num_users
        self.models = [QNetwork(input_dim, num_actions).to(
            self.device) for _ in range(num_users)]
        self.target_models = [QNetwork(input_dim, num_actions).to(
            self.device) for _ in range(num_users)]
        self.optimizers = [optim.Adam(
            model.parameters(), lr=learning_rate_deepQ) for model in self.models]
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.criterion = nn.SmoothL1Loss()

        if load_path and os.path.exists(load_path):
            try:
                self.load_models(load_path)
                print(f"Loaded models from {load_path}")
            except RuntimeError as e:
                print(
                    f"Failed to load checkpoint due to mismatch: {e}. Initializing new models.")
                for target, model in zip(self.target_models, self.models):
                    target.load_state_dict(model.state_dict())
        else:
            for target, model in zip(self.target_models, self.models):
                target.load_state_dict(model.state_dict())

    def get_action(self, state, user_idx):
        possible_actions = self.env.get_possible_actions(user_idx)
        if random.random() < self.epsilon:
            action = random.choice(possible_actions)
            state_tensor = torch.FloatTensor(
                state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.models[user_idx](state_tensor)[0].cpu().numpy()
            q_value = q_values[action]
        else:
            state_tensor = torch.FloatTensor(
                state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.models[user_idx](state_tensor)[0].cpu().numpy()
            action = max(possible_actions, key=lambda a: q_values[a])
            q_value = q_values[action]
        return action, q_value

    def remember(self, state, actions, reward, next_state, individual_rewards):
        self.memory.append(
            (state, actions, reward, next_state, individual_rewards))

    def replay(self):
        if len(self.memory) < batch_size:
            return None
        # batch = random.sample(self.memory, batch_size)
        # states, actions, rewards, next_states, _ = zip(*batch)
        # states = torch.FloatTensor(states).to(self.device)
        # actions = torch.LongTensor(actions).to(self.device)
        # rewards = torch.FloatTensor(rewards).to(self.device)
        # next_states = torch.FloatTensor(next_states).to(self.device)

        batch = random.sample(self.memory, batch_size)
        states, actions, _, next_states, individual_rewards = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        individual_rewards = torch.FloatTensor(individual_rewards).to(self.device)

        losses = []
        for user_idx in range(self.num_users):
            self.optimizers[user_idx].zero_grad()
            q_values = self.models[user_idx](states)
            target_q_values = self.target_models[user_idx](
                next_states).detach()
            max_next_q_values = target_q_values.max(dim=1)[0]
            # target_q = rewards + gamma_deepQ * max_next_q_values
            target_q = individual_rewards[:, user_idx] + gamma_deepQ * max_next_q_values
            q_action = q_values.gather(
                1, actions[:, user_idx].unsqueeze(1)).squeeze(1)
            loss = self.criterion(q_action, target_q)
            loss.backward()
            self.optimizers[user_idx].step()
            losses.append(loss.item())
        return losses

    def update_target(self):
        for target, model in zip(self.target_models, self.models):
            target.load_state_dict(model.state_dict())

    def save_models(self, path):
        checkpoint = {
            'models_state_dict': [model.state_dict() for model in self.models],
            'target_models_state_dict': [target.state_dict() for target in self.target_models],
            'optimizers_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, path)

    def load_models(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        for idx, (model, target, optimizer) in enumerate(zip(self.models, self.target_models, self.optimizers)):
            model.load_state_dict(checkpoint['models_state_dict'][idx])
            target.load_state_dict(checkpoint['target_models_state_dict'][idx])
            optimizer.load_state_dict(checkpoint['optimizers_state_dict'][idx])
        self.epsilon = checkpoint['epsilon']

    def plot_progress(self, total_history, per_user_history, filename="multi_user_training_progress_tdma_reverted.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(total_history, label='Total Avg Reward', color='blue')
        for u in range(self.num_users):
            plt.plot(
                per_user_history[u], label=f'User {u+1} Avg Reward', linestyle='--')
        plt.xlabel('Iteration (Time Slot)')
        plt.ylabel('Average Reward')
        plt.title(
            'Multi-User Training Progression (TDMA with Centralized Coordination)')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Saved training progress plot to {filename}")

    def log_to_file(self, filename, iteration, avg_total, avg_per_user, packet_loss_ratios, avg_losses, count_change, packet_lost_by_transmit, energy_used):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        header = "Iteration\tAvg_Total_Reward\t" + "\t".join([f"Avg_Reward_User_{u}" for u in range(self.num_users)]) + \
                 "\t" + "\t".join([f"Packet_Loss_Ratio_User_{u}" for u in range(self.num_users)]) + \
                 "\t" + "\t".join([f"Packet_Loss_By_Transmit_Ratio_User_{u}" for u in range(self.num_users)]) + \
                 "\t" + \
            "\t".join([f"Avg_Loss_User_{u}" for u in range(
                self.num_users)]) + "\tEnergy_used" + "\tCount_Change" + "\n"
        data_line = f"{iteration}\t{avg_total:.4f}\t" + \
            "\t".join([f"{r:.4f}" for r in avg_per_user]) + "\t" + \
            "\t".join([f"{r:.4f}" for r in packet_loss_ratios]) + "\t" + \
            "\t".join([f"{r:.4f}" for r in packet_lost_by_transmit]) + "\t" + \
            "\t".join([f"{l:.6f}" for l in avg_losses]) + \
            f"\t{energy_used}" + f"\t{count_change}\n"
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(header)
        with open(filename, 'a') as f:
            f.write(data_line)

    
    def train(self, save_path="multi_user_checkpoint_tdma_reverted.pth",
              plot_path="multi_user_training_progress_tdma_reverted.png",
              log_path="log/multi_user_training_data_tdma_reverted.txt"):
        total_reward = 0
        per_user_totals = [0] * self.num_users
        total_packets_arrived = [0] * self.num_users
        total_packets_lost = [0] * self.num_users
        total_packets_lost_by_transmit = [0] * self.num_users
        total_history = []
        total_energy_used = []
        per_user_history = [[] for _ in range(self.num_users)]
        loss_history = [[] for _ in range(self.num_users)]
        state = self.env.get_state()
        transmission_actions = [1, 4, 5, 6]
        data_state_history = [[] for _ in range(self.num_users)]
        energy_state_history = [[] for _ in range(self.num_users)]
        count_changes = []
        count_change = 0
        start = time.time()
        for i in range(T):
            actions = [0] * self.num_users
            current_agent = self.env.time_slot

            # Step 1: Evaluate the current time slot's agent
            possible_actions = self.env.get_possible_actions(current_agent)
            # print(f"Possible actions for agent {current_agent}: {possible_actions}")
            current_action, current_q_value = self.get_action(
                state, current_agent)

            # Step 2: If current agent selects a transmission action, they transmit
            if current_action in transmission_actions:
                actions[current_agent] = current_action
                # Non-transmitting agents harvest if possible
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
                            state_tensor = torch.FloatTensor(
                                state).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                q_values = self.models[u](state_tensor)[
                                    0].cpu().numpy()
                            action = max(possible_actions,
                                         key=lambda a: q_values[a])
                        actions[u] = action
            else:
                count_change += 1
                # Step 3: Current agent does not transmit, evaluate other agents
                candidates = []
                for u in range(self.num_users):
                    if u == current_agent:
                        continue  # Skip the current agent
                    possible_transmissions = [
                        a for a in self.env.get_possible_actions(u) if a in transmission_actions]
                    if possible_transmissions:
                        action, q_value = self.get_action(state, u)
                        if action in transmission_actions:
                            # q_value += self.env.data_states[u] / d_queue_size  # Queue bonus
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
                                state_tensor = torch.FloatTensor(
                                    state).unsqueeze(0).to(self.device)
                                with torch.no_grad():
                                    q_values = self.models[u](state_tensor)[
                                        0].cpu().numpy()
                                action = max(possible_actions,
                                             key=lambda a: q_values[a])
                            actions[u] = action
                else:
                    # No one transmits, all agents harvest if possible
                    for u in range(self.num_users):
                        possible_actions = [a for a in self.env.get_possible_actions(
                            u) if a not in transmission_actions]
                        if not possible_actions:
                            # Ensure idle is always possible
                            possible_actions = [0]
                        if random.random() < self.epsilon:
                            action = random.choice(possible_actions)
                        else:
                            state_tensor = torch.FloatTensor(
                                state).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                q_values = self.models[u](state_tensor)[
                                    0].cpu().numpy()
                            action = max(possible_actions,
                                         key=lambda a: q_values[a])
                        actions[u] = action

            # Environment step
            total_reward_step, next_state, individual_rewards, packets_arrived, packets_lost, packets_lost_by_transmit, energy_used = self.env.step(
                actions)
            total_reward += total_reward_step
            total_energy_used.append(energy_used)
            packetLost = 0
            for u in range(self.num_users):
                per_user_totals[u] += individual_rewards[u]
                total_packets_arrived[u] += individual_rewards[u]
                total_packets_lost[u] += packets_lost[u]
                total_packets_lost_by_transmit[u] += packets_lost_by_transmit[u]
                packetLost += packets_lost[u]
                per_user_history[u].append(per_user_totals[u] / (i + 1))
                data_state_history[u].append(self.env.data_states[u])
                energy_state_history[u].append(self.env.energy_states[u])
            total_history.append(total_reward / (i + 1))
            self.remember(state, actions, total_reward_step,
                          next_state, individual_rewards)
            losses = self.replay()
            if losses is not None:
                for u in range(self.num_users):
                    loss_history[u].append(losses[u])
            state = next_state

            if (i + 1) % 2000 == 0:
                self.update_target()
            if (i + 1) % step == 0:
                avg_data_state = [np.mean(data_state_history[u])
                                  for u in range(self.num_users)]
                avg_energy_state = [np.mean(energy_state_history[u])
                                    for u in range(self.num_users)]
                end = time.time()
                print(f"Avg Data States: {[f'{s:.4f}' for s in avg_data_state]}, "
                      f"Avg Energy States: {[f'{s:.4f}' for s in avg_energy_state]}, Epsilon: {self.epsilon:.4f}, Count Change: {count_change}, Time: {end - start:.2f}s")
                start = end
                count_changes.append(count_change)
                count_change = 0
                avg_total = np.mean(total_history[-step:])
                avg_per_user = [np.mean(per_user_history[u][-step:])
                                for u in range(self.num_users)]
                packet_loss_ratios = [total_packets_lost[u] / (total_packets_arrived[u] + total_packets_lost[u]) if total_packets_arrived[u] > 0 else 0
                                      for u in range(self.num_users)]
                packet_loss_by_transmit_ratios = [total_packets_lost_by_transmit[u] / (total_packets_arrived[u] + total_packets_lost_by_transmit[u]) if total_packets_arrived[u] > 0 else 0
                                                  for u in range(self.num_users)]
                avg_loss_per_user = [np.mean(loss_history[u]) for u in range(
                    self.num_users)] if loss_history[0] else [0] * self.num_users
                avg_energy_used = np.mean(total_energy_used[-step:])
                if losses is not None:
                    loss_str = ", ".join(
                        [f"User {u}: {avg_loss_per_user[u]:.6f}" for u in range(self.num_users)])
                    print(f"Time Slot {i + 1}, Avg Total Reward: {avg_total:.4f}, "
                          f"Avg Per-User Rewards: {[f'{r:.4f}' for r in avg_per_user]}, "
                          f"Avg Losses: [{loss_str}], "
                          f"Packet Loss Ratios: {[f'{r:.4f}' for r in packet_loss_ratios]}, "
                          f"Packet Loss Ratios By Transmit: {[f'{r:.4f}' for r in packet_loss_by_transmit_ratios]}"
                          f"Energy Efficiency: {avg_total/avg_energy_used:.4f}")
                else:
                    print(f"Time Slot {i + 1}, Avg Total Reward: {avg_total:.4f}, "
                          f"Avg Per-User Rewards: {[f'{r:.4f}' for r in avg_per_user]}, "
                          f"Packet Loss Ratios: {[f'{r:.4f}' for r in packet_loss_ratios]}, "
                          f"Packet Loss Ratios By Transmit: {[f'{r:.4f}' for r in packet_loss_by_transmit_ratios]}"
                          f"Energy Efficiency: {avg_total/avg_energy_used:.4f}")
                self.save_models(save_path)
                self.log_to_file(log_path, i + 1, avg_total,
                                 avg_per_user, packet_loss_ratios, avg_loss_per_user, count_changes[-1], packet_loss_by_transmit_ratios, sum(total_energy_used[-step:]))

                if (i + 1) == 100000:
                    self.plot_progress(total_history, per_user_history,
                                       f"plot/test_final11/multi_user_progress_at_100k_tdma_reverted_{num_users}_rate_{arrival_rate}_eps_{self.epsilon_decay}_nu_{nu}_HTT1.png")

            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)
        avg_per_user_final = [per_user_totals[u] /
                              T for u in range(self.num_users)]
        final_packet_loss_ratios = [total_packets_lost[u] / (total_packets_arrived[u] + total_packets_lost[u]) if total_packets_arrived[u] > 0 else 0
                                    for u in range(self.num_users)]
        final_packet_loss_ratios_by_transmit = [total_packets_lost_by_transmit[u] / (total_packets_arrived[u] + total_packets_lost_by_transmit[u]) if total_packets_arrived[u] > 0 else 0
                                                for u in range(self.num_users)]
        final_avg_loss_per_user = [np.mean(loss_history[u]) for u in range(
            self.num_users)] if loss_history[0] else [0] * self.num_users
        final_energy_used = np.mean(total_energy_used[-step:])
        self.save_models(save_path)
        self.plot_progress(total_history, per_user_history, plot_path)
        self.log_to_file(log_path, T, total_reward / T, avg_per_user_final,
                         final_packet_loss_ratios, final_avg_loss_per_user, count_changes[-1], final_packet_loss_ratios_by_transmit, sum(total_energy_used[-step:]))
        print(
            f"Final Packet Loss Ratios: {[f'{r:.4f}' for r in final_packet_loss_ratios]}")
        print(
            f"Final Packet Loss Ratios: {[f'{r:.4f}' for r in final_packet_loss_ratios_by_transmit]}")
        print(
            f"Final Avg Losses: {[f'{r:.6f}' for r in final_avg_loss_per_user]}")
        return total_reward / T, avg_per_user_final


if __name__ == "__main__":
    agent = DeepQLearningAgent(load_path=None, backscatter=False)
    avg_total_multi, avg_per_user_multi = agent.train(
        save_path=f"checkpoint/test_final11/multi_user_checkpoint_tdma_reverted_{num_users}_rate_{arrival_rate}_eps_{agent.epsilon_decay}_nu_{nu}_HTT1.pth",
        plot_path=f"plot/test_final11/multi_user_training_progress_tdma_reverted_{num_users}_rate_{arrival_rate}_eps_{agent.epsilon_decay}_nu_{nu}_HTT1.png",
        log_path=f"log/test_final11/multi_user_training_data_tdma_reverted_{num_users}_rate_{arrival_rate}_eps_{agent.epsilon_decay}_nu_{nu}_HTT1.txt"
    )
    print(f"Multi-user total average reward: {avg_total_multi:.4f}")
    print(
        f"Multi-user per-user average rewards: {[f'{r:.4f}' for r in avg_per_user_multi]}")
    print(
        f"Multi-user average per-user reward (mean): {np.mean(avg_per_user_multi):.4f}")
