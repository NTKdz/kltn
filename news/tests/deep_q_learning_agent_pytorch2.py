# deep_q_learning_agent_pytorch_jal.py
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

class QNetworkJAL(nn.Module):
    def __init__(self, input_dim, num_actions, num_users):
        super(QNetworkJAL, self).__init__()
        self.num_actions = num_actions
        self.num_users = num_users
        self.joint_action_size = num_actions ** num_users  # 49 for 2 users, 7 actions each
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, self.joint_action_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)  # Outputs Q-values for all 49 joint actions
        return x

class DeepQLearningAgentJAL:
    def __init__(self, num_users=num_users, input_dim=num_features, load_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = Environment(num_users)
        self.num_users = num_users
        self.num_actions = num_actions  # 7
        self.joint_action_size = num_actions ** num_users  # 49
        self.models = [QNetworkJAL(input_dim, num_actions, num_users).to(self.device) for _ in range(num_users)]
        self.target_models = [QNetworkJAL(input_dim, num_actions, num_users).to(self.device) for _ in range(num_users)]
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate_deepQ) for model in self.models]
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.criterion = nn.SmoothL1Loss()

        if load_path and os.path.exists(load_path):
            self.load_models(load_path)
            print(f"Loaded models from {load_path}")
        else:
            for target, model in zip(self.target_models, self.models):
                target.load_state_dict(model.state_dict())

    def joint_action_to_index(self, actions):
        """Convert list of actions [a_1, a_2] to a single index (0-48)."""
        return actions[0] * self.num_actions + actions[1]  # For 2 users: a_1 * 7 + a_2

    def index_to_joint_action(self, index):
        """Convert index (0-48) back to joint actions [a_1, a_2]."""
        a1 = index // self.num_actions
        a2 = index % self.num_actions
        return [a1, a2]

    def get_action(self, state, user_idx):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        possible_actions = self.env.get_possible_actions(user_idx)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            with torch.no_grad():
                # Get Q-values for all joint actions from both users' models
                q_values_self = self.models[user_idx](state_tensor)[0].cpu().numpy()  # Shape: (49,)
                q_values_other = self.models[1 - user_idx](state_tensor)[0].cpu().numpy()  # Other user's Q-values

            # Assume other user acts greedily
            other_user_idx = 1 - user_idx
            other_possible = self.env.get_possible_actions(other_user_idx)
            best_other_action = max(other_possible, key=lambda a: q_values_other[self.joint_action_to_index([0, a] if user_idx == 1 else [a, 0])])

            # Select own action assuming other's greedy choice
            best_action = None
            best_q = float('-inf')
            for a in possible_actions:
                joint_action = [a, best_other_action] if user_idx == 0 else [best_other_action, a]
                joint_idx = self.joint_action_to_index(joint_action)
                q_val = q_values_self[joint_idx]
                if q_val > best_q:
                    best_q = q_val
                    best_action = a
            return best_action

    def remember(self, state, actions, reward, next_state, individual_rewards):
        self.memory.append((state, actions, reward, next_state, individual_rewards))

    def replay(self):
        if len(self.memory) < batch_size:
            return None
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, _ = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        joint_actions = torch.LongTensor([self.joint_action_to_index(a) for a in actions]).to(self.device)  # Shape: (batch_size,)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        losses = []
        for user_idx in range(self.num_users):
            self.optimizers[user_idx].zero_grad()
            q_values = self.models[user_idx](states)  # Shape: (batch_size, 49)
            next_q_values = self.target_models[user_idx](next_states).detach()  # Shape: (batch_size, 49)
            target_q = rewards + gamma_deepQ * next_q_values.max(dim=1)[0]  # Shape: (batch_size,)
            q_action = q_values.gather(1, joint_actions.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)
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

    def plot_progress(self, total_history, per_user_history, filename="jal_training_progress.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(total_history, label='Total Avg Reward', color='blue')
        for u in range(self.num_users):
            plt.plot(per_user_history[u], label=f'User {u+1} Avg Reward', linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title('JAL Multi-User Training Progression')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Saved training progress plot to {filename}")

    def train(self, save_path="jal_checkpoint.pth", plot_path="jal_training_progress.png"):
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
                avg_total = total_reward / (i + 1)
                avg_per_user = [per_user_totals[u] / (i + 1) for u in range(self.num_users)]
                packet_loss_ratios = [total_packets_lost[u] / total_packets_arrived[u] if total_packets_arrived[u] > 0 else 0 
                                      for u in range(self.num_users)]
                avg_loss_per_user = [np.mean(loss_history[u]) for u in range(self.num_users)] if loss_history[0] else [0] * self.num_users
                if losses is not None:
                    loss_str = ", ".join([f"User {u}: {avg_loss_per_user[u]:.6f}" for u in range(self.num_users)])
                    print(f"Iteration {i + 1}, Avg Total Reward: {avg_total:.4f}, "
                          f"Avg Per-User Rewards: {[f'{r:.4f}' for r in avg_per_user]}, "
                          f"Avg Losses: [{loss_str}], "
                          f"Packet Loss Ratios: {[f'{r:.4f}' for r in packet_loss_ratios]}")
                else:
                    print(f"Iteration {i + 1}, Avg Total Reward: {avg_total:.4f}, "
                          f"Avg Per-User Rewards: {[f'{r:.4f}' for r in avg_per_user]}, "
                          f"Packet Loss Ratios: {[f'{r:.4f}' for r in packet_loss_ratios]}")
                self.save_models(save_path)
                if (i + 1) == 100000:
                    self.plot_progress(total_history, per_user_history, "jal_progress_at_100k.png")
        avg_per_user_final = [per_user_totals[u] / T for u in range(self.num_users)]
        final_packet_loss_ratios = [total_packets_lost[u] / total_packets_arrived[u] if total_packets_arrived[u] > 0 else 0 
                                    for u in range(self.num_users)]
        final_avg_loss_per_user = [np.mean(loss_history[u]) for u in range(self.num_users)] if loss_history[0] else [0] * self.num_users
        self.save_models(save_path)
        self.plot_progress(total_history, per_user_history, plot_path)
        print(f"Final Packet Loss Ratios: {[f'{r:.4f}' for r in final_packet_loss_ratios]}")
        print(f"Final Avg Losses: {[f'{r:.6f}' for r in final_avg_loss_per_user]}")
        return total_reward / T, avg_per_user_final

if __name__ == "__main__":
    agent = DeepQLearningAgentJAL(load_path="jal_checkpoint.pth")
    avg_total_multi, avg_per_user_multi = agent.train(
        save_path="jal_checkpoint.pth",
        plot_path="jal_training_progress.png"
    )
    print(f"JAL total average reward: {avg_total_multi:.4f}")
    print(f"JAL per-user average rewards: {[f'{r:.4f}' for r in avg_per_user_multi]}")
    print(f"JAL average per-user reward (mean): {np.mean(avg_per_user_multi):.4f}")