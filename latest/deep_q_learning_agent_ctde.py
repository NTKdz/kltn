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
    def __init__(self, input_dim, output_dim, num_users):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.heads = nn.ModuleList([nn.Linear(64, output_dim) for _ in range(num_users)])
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        actions = [head(x) for head in self.heads]
        return torch.stack(actions, dim=1)  # Shape: [batch, num_users, num_actions]

class DeepQLearningAgent:
    def __init__(self, num_users=num_users, input_dim=num_features, load_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = Environment(num_users)
        self.num_users = num_users
        self.model = QNetwork(input_dim, num_actions, num_users).to(self.device)
        self.target_model = QNetwork(input_dim, num_actions, num_users).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate_deepQ)
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
                print(f"Failed to load checkpoint: {e}. Initializing new models.")
                self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, user_idx):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        possible_actions = self.env.get_possible_actions(user_idx)
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)[0, user_idx].cpu().numpy()
        return max(possible_actions, key=lambda a: q_values[a])

    def remember(self, state, actions, reward, next_state, individual_rewards):
        self.memory.append((state, actions, reward, next_state, individual_rewards))

    def replay(self):
        if len(self.memory) < batch_size:
            return None
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, _ = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        self.optimizer.zero_grad()
        q_values = self.model(states)  # [batch, num_users, num_actions]
        next_q_values = self.target_model(next_states).detach()
        target_q = rewards.unsqueeze(1).repeat(1, self.num_users) + gamma_deepQ * next_q_values.max(dim=2)[0]
        q_action = torch.gather(q_values, 2, actions.unsqueeze(2)).squeeze(2)
        loss = self.criterion(q_action, target_q)
        loss.backward()
        self.optimizer.step()
        return [loss.item()] * self.num_users

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_models(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, path)

    def load_models(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

    def plot_progress(self, total_history, per_user_history, filename="multi_user_training_progress_tdma_ctde.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(total_history, label='Total Avg Reward', color='blue')
        for u in range(self.num_users):
            plt.plot(per_user_history[u], label=f'User {u+1} Avg Reward', linestyle='--')
        plt.xlabel('Iteration (Time Slot)')
        plt.ylabel('Average Reward')
        plt.title('Multi-User Training Progression (TDMA with CTDE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Saved training progress plot to {filename}")

    def log_to_file(self, filename, iteration, avg_total, avg_per_user, packet_loss_ratios, avg_losses):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        header = "Iteration\tAvg_Total_Reward\t" + "\t".join([f"Avg_Reward_User_{u}" for u in range(self.num_users)]) + \
                 "\t" + "\t".join([f"Packet_Loss_Ratio_User_{u}" for u in range(self.num_users)]) + \
                 "\t" + "\t".join([f"Avg_Loss_User_{u}" for u in range(self.num_users)]) + "\n"
        data_line = f"{iteration}\t{avg_total:.4f}\t" + "\t".join([f"{r:.4f}" for r in avg_per_user]) + \
                    "\t" + "\t".join([f"{r:.4f}" for r in packet_loss_ratios]) + \
                    "\t" + "\t".join([f"{l:.6f}" for l in avg_losses]) + "\n"
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(header)
        with open(filename, 'a') as f:
            f.write(data_line)

    def train(self, save_path="multi_user_checkpoint_tdma_ctde.pth", plot_path="multi_user_training_progress_tdma_ctde.png", log_path="log/multi_user_training_data_tdma_ctde.txt"):
        total_reward = 0
        per_user_totals = [0] * self.num_users
        total_packets_arrived = [0] * self.num_users
        total_packets_lost = [0] * self.num_users
        total_history = []
        per_user_history = [[] for _ in range(self.num_users)]
        loss_history = [[] for _ in range(self.num_users)]
        state = self.env.get_state()
        for i in range(T):
            actions = [self.get_action(state, u) for u in range(self.num_users)]
            total_reward_step, next_state, individual_rewards, packets_arrived, packets_lost = self.env.step(actions)
            total_reward += total_reward_step
            for u in range(self.num_users):
                per_user_totals[u] += individual_rewards[u]
                total_packets_arrived[u] += packets_arrived[u]
                total_packets_lost[u] += packets_lost[u]
                per_user_history[u].append(per_user_totals[u] / (i + 1))
            total_history.append(total_reward / (i + 1))
            self.remember(state, actions, total_reward_step, next_state, individual_rewards)
            losses = self.replay()
            if losses is not None:
                for u in range(self.num_users):
                    loss_history[u].append(losses[u])
            state = next_state
            if (i + 1) % 10000 == 0:
                self.update_target()
            if (i + 1) % step == 0:
                print(self.env.data_states)
                avg_total = total_reward / (i + 1)
                avg_per_user = [per_user_totals[u] / (i + 1) for u in range(self.num_users)]
                packet_loss_ratios = [total_packets_lost[u] / total_packets_arrived[u] if total_packets_arrived[u] > 0 else 0 
                                      for u in range(self.num_users)]
                avg_loss_per_user = [np.mean(loss_history[u]) for u in range(self.num_users)] if loss_history[0] else [0] * self.num_users
                if losses is not None:
                    loss_str = ", ".join([f"User {u}: {avg_loss_per_user[u]:.6f}" for u in range(self.num_users)])
                    print(f"Time Slot {i + 1}, Avg Total Reward: {avg_total:.4f}, "
                          f"Avg Per-User Rewards: {[f'{r:.4f}' for r in avg_per_user]}, "
                          f"Avg Losses: [{loss_str}], "
                          f"Packet Loss Ratios: {[f'{r:.4f}' for r in packet_loss_ratios]}")
                else:
                    print(f"Time Slot {i + 1}, Avg Total Reward: {avg_total:.4f}, "
                          f"Avg Per-User Rewards: {[f'{r:.4f}' for r in avg_per_user]}, "
                          f"Packet Loss Ratios: {[f'{r:.4f}' for r in packet_loss_ratios]}")
                self.save_models(save_path)
                self.log_to_file(log_path, i + 1, avg_total, avg_per_user, packet_loss_ratios, avg_loss_per_user)
                if (i + 1) == 100000:
                    self.plot_progress(total_history, per_user_history, "plot/multi_user_progress_at_100k_tdma_ctde_6.png")
        avg_per_user_final = [per_user_totals[u] / T for u in range(self.num_users)]
        final_packet_loss_ratios = [total_packets_lost[u] / total_packets_arrived[u] if total_packets_arrived[u] > 0 else 0 
                                    for u in range(self.num_users)]
        final_avg_loss_per_user = [np.mean(loss_history[u]) for u in range(self.num_users)] if loss_history[0] else [0] * self.num_users
        self.save_models(save_path)
        self.plot_progress(total_history, per_user_history, plot_path)
        self.log_to_file(log_path, T, total_reward / T, avg_per_user_final, final_packet_loss_ratios, final_avg_loss_per_user)
        print(f"Final Packet Loss Ratios: {[f'{r:.4f}' for r in final_packet_loss_ratios]}")
        print(f"Final Avg Losses: {[f'{r:.6f}' for r in final_avg_loss_per_user]}")
        return total_reward / T, avg_per_user_final

if __name__ == "__main__":
    agent = DeepQLearningAgent(load_path=None)
    avg_total_multi, avg_per_user_multi = agent.train(
        save_path="checkpoint/multi_user_checkpoint_tdma_ctde_6.pth",
        plot_path="plot/multi_user_training_progress_tdma_ctde_6.png",
        log_path="log/multi_user_training_data_tdma_ctde_6.txt"
    )
    print(f"Multi-user total average reward: {avg_total_multi:.4f}")
    print(f"Multi-user per-user average rewards: {[f'{r:.4f}' for r in avg_per_user_multi]}")
    print(f"Multi-user average per-user reward (mean): {np.mean(avg_per_user_multi):.4f}")