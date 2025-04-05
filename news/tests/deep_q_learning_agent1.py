# deep_q_learning_single_pytorch.py
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
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

class DeepQLearningAgentSingle:
    def __init__(self, input_dim=num_features_single, load_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = Environment(num_users=1)
        self.model = QNetwork(input_dim, num_actions).to(self.device)
        self.target_model = QNetwork(input_dim, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate_deepQ)
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.criterion = nn.SmoothL1Loss()

        if load_path and os.path.exists(load_path):
            self.load_model(load_path)
            print(f"Loaded model from {load_path}")
        else:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        possible_actions = self.env.get_possible_actions(0)
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)[0].cpu().numpy()
            return max(possible_actions, key=lambda a: q_values[a])

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self):
        if len(self.memory) < batch_size:
            return None
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        self.optimizer.zero_grad()
        q_values = self.model(states)
        next_q_values = self.target_model(next_states).detach()
        target_q = rewards + gamma_deepQ * next_q_values.max(dim=1)[0]
        q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.criterion(q_action, target_q)
        loss.backward()
        self.optimizer.step()
        return loss.item()  # Return scalar loss

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

    def plot_progress(self, reward_history, filename="single_user_training_progress.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(reward_history, label='Avg Reward', color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title('Single-User Training Progression')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Saved training progress plot to {filename}")

    def train(self, save_path="single_user_checkpoint.pth", plot_path="single_user_training_progress.png"):
        total_reward = 0
        total_packets_arrived = 0
        total_packets_lost = 0
        reward_history = []
        loss_history = []  # Track loss over time
        state = self.env.get_state()
        for i in range(T):
            action = self.get_action(state)
            reward, next_state, individual_rewards, packets_arrived, packets_lost = self.env.step([action])
            total_reward += reward
            total_packets_arrived += packets_arrived[0]
            total_packets_lost += packets_lost[0]
            reward_history.append(total_reward / (i + 1))
            self.remember(state, action, reward, next_state)
            loss = self.replay()
            if loss is not None:
                loss_history.append(loss)
            state = next_state
            if (i + 1) % 10000 == 0:
                self.update_target()
            if (i + 1) % step == 0:
                avg_reward = total_reward / (i + 1)
                packet_loss_ratio = total_packets_lost / total_packets_arrived if total_packets_arrived > 0 else 0
                avg_loss = np.mean(loss_history) if loss_history else 0
                if loss is not None:
                    print(f"Iteration {i + 1}, Avg Reward: {avg_reward:.4f}, "
                          f"Avg Loss: [User 0: {avg_loss:.6f}], "
                          f"Packet Loss Ratio: [{packet_loss_ratio:.4f}]")
                else:
                    print(f"Iteration {i + 1}, Avg Reward: {avg_reward:.4f}, "
                          f"Packet Loss Ratio: [{packet_loss_ratio:.4f}]")
                self.save_model(save_path)
                if (i + 1) == 100000:
                    self.plot_progress(reward_history, "single_user_progress_at_100k_0.png")
        final_packet_loss_ratio = total_packets_lost / total_packets_arrived if total_packets_arrived > 0 else 0
        final_avg_loss = np.mean(loss_history) if loss_history else 0
        self.save_model(save_path)
        self.plot_progress(reward_history, plot_path)
        print(f"Final Avg Loss: [{final_avg_loss:.6f}]")
        print(f"Final Packet Loss Ratio: [{final_packet_loss_ratio:.4f}]")
        return total_reward / T

if __name__ == "__main__":
    agent = DeepQLearningAgentSingle(load_path="single_user_checkpoint_0.pth")
    avg_reward_single = agent.train(
        save_path="single_user_checkpoint_0.pth",
        plot_path="single_user_training_progress_0.png"
    )
    print(f"Single-user average reward: {avg_reward_single:.4f}")