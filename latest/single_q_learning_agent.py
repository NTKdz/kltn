from parameters import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Single-agent environment (one user, one channel)
class SingleEnvironment:
    def __init__(self):
        self.jammer_state = 0  # Single channel jammer state
        self.data_state = 0
        self.energy_state = 0
        self.channel = 0  # Single channel

    def get_state(self):
        return np.array([self.jammer_state, self.data_state, self.energy_state])

    def get_possible_actions(self):
        list_actions = [0]  # Stay idle
        if self.jammer_state == 0 and self.data_state > 0 and self.energy_state >= e_t:
            list_actions.append(1)  # Active transmit
        if self.jammer_state == 1:
            list_actions.append(2)  # Harvest energy
            if self.data_state > 0:
                list_actions.append(3)  # Backscatter
            if self.energy_state >= e_t:
                list_actions.extend([4, 5, 6])  # Rate adaptation
        return list_actions

    def active_transmit(self, max_packets):
        num_transmitted = 0
        if 0 < self.data_state < max_packets:
            if self.energy_state >= e_t * self.data_state:
                num_transmitted = self.data_state
            elif self.energy_state >= e_t:
                num_transmitted = self.energy_state // e_t
        else:
            if self.energy_state >= e_t * max_packets:
                num_transmitted = max_packets
            elif self.energy_state >= e_t:
                num_transmitted = self.energy_state // e_t
        return num_transmitted

    def calculate_reward(self, action):
        reward = 0
        loss = 0
        if action == 0:
            reward = 0
        elif action == 1 and self.jammer_state == 0:
            reward = self.active_transmit(d_t)
        elif action == 2 and self.jammer_state == 1:
            reward = random.choices(e_hj_arr, nu_p, k=1)[0]
        elif action == 3 and self.jammer_state == 1:
            d_bj = random.choices(d_bj_arr, nu_p, k=1)[0]
            max_rate = min(b_dagger, self.data_state)
            reward = min(d_bj, self.data_state)
            if max_rate > reward:
                loss = max_rate - reward
        elif action in [4, 5, 6] and self.jammer_state == 1:
            max_ra = random.choices(dt_ra_arr, nu_p, k=1)[0]
            idx = action - 4
            reward = self.active_transmit(dt_ra_arr[idx])
            if dt_ra_arr[idx] > max_ra:
                loss = reward
                reward = 0
        return reward, loss

    def step(self, action):
        reward, loss = self.calculate_reward(action)
        packets_arrived = 0
        packets_lost = 0

        if action == 1:
            self.data_state = max(0, self.data_state - reward)
            self.energy_state = max(0, self.energy_state - reward * e_t)
        elif action == 2:
            if self.energy_state < e_queue_size:
                self.energy_state = min(e_queue_size, self.energy_state + reward)
            reward = 0  # Reward is in energy gain
        elif action == 3:
            max_rate = min(b_dagger, self.data_state)
            self.data_state = max(0, self.data_state - max_rate)
            packets_lost += loss
        elif action in [4, 5, 6]:
            if reward > 0:
                self.data_state = max(0, self.data_state - reward)
                self.energy_state = max(0, self.energy_state - reward * e_t)
            packets_lost += loss

        # Data arrival
        data_arrive = poisson.rvs(mu=arrival_rate)
        packets_arrived = data_arrive
        new_data_state = self.data_state + data_arrive
        if new_data_state > d_queue_size:
            packets_lost += (new_data_state - d_queue_size)
        self.data_state = min(d_queue_size, new_data_state)

        # Jammer state
        if self.jammer_state == 0:
            if np.random.random() <= 1 - nu:
                self.jammer_state = 1
        else:
            if np.random.random() <= nu:
                self.jammer_state = 0

        next_state = self.get_state()
        return reward, next_state, packets_arrived, packets_lost

# Q-Network (same as multi-agent)
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

class SingleDeepQLearningAgent:
    def __init__(self, input_dim=3, load_path=None):  # 3 features: jammer, data, energy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = SingleEnvironment()
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
        possible_actions = self.env.get_possible_actions()
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
        return loss.item()

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

    def plot_progress(self, reward_history, filename="single_training_progress.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(reward_history, label='Avg Reward', color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title('Single-Agent Training Progression')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Saved training progress plot to {filename}")

    def log_to_file(self, filename, iteration, avg_reward, packet_loss_ratio, avg_loss):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        header = "Iteration\tAvg_Reward\tPacket_Loss_Ratio\tAvg_Loss\n"
        data_line = f"{iteration}\t{avg_reward:.4f}\t{packet_loss_ratio:.4f}\t{avg_loss:.6f}\n"
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(header)
        with open(filename, 'a') as f:
            f.write(data_line)

    def train(self, save_path="single_checkpoint.pth", plot_path="single_training_progress.png", log_path="log/single_training_data.txt"):
        total_reward = 0
        total_packets_arrived = 0
        total_packets_lost = 0
        reward_history = []
        loss_history = []
        state = self.env.get_state()
        for i in range(T):
            action = self.get_action(state)
            reward, next_state, packets_arrived, packets_lost = self.env.step(action)
            total_reward += reward
            total_packets_arrived += packets_arrived
            total_packets_lost += packets_lost
            reward_history.append(total_reward / (i + 1))
            self.remember(state, action, reward, next_state)
            loss = self.replay()
            if loss is not None:
                loss_history.append(loss)
            state = next_state
            if (i + 1) % 10000 == 0:
                self.update_target()
            if (i + 1) % step == 0:
                print(self.env.data_state)
                avg_reward = total_reward / (i + 1)
                packet_loss_ratio = total_packets_lost / total_packets_arrived if total_packets_arrived > 0 else 0
                avg_loss = np.mean(loss_history) if loss_history else 0
                if loss is not None:
                    print(f"Iteration {i + 1}, Avg Reward: {avg_reward:.4f}, Loss: {avg_loss:.6f}, Packet Loss Ratio: {packet_loss_ratio:.4f}")
                else:
                    print(f"Iteration {i + 1}, Avg Reward: {avg_reward:.4f}, Packet Loss Ratio: {packet_loss_ratio:.4f}")
                self.save_model(save_path)
                self.log_to_file(log_path, i + 1, avg_reward, packet_loss_ratio, avg_loss)
                if (i + 1) == 100000:
                    self.plot_progress(reward_history, "plot/single_progress_at_100k.png")
        final_packet_loss_ratio = total_packets_lost / total_packets_arrived if total_packets_arrived > 0 else 0
        final_avg_loss = np.mean(loss_history) if loss_history else 0
        self.save_model(save_path)
        self.plot_progress(reward_history, plot_path)
        self.log_to_file(log_path, T, total_reward / T, final_packet_loss_ratio, final_avg_loss)
        print(f"Final Packet Loss Ratio: {final_packet_loss_ratio:.4f}")
        print(f"Final Avg Loss: {final_avg_loss:.6f}")
        return total_reward / T

if __name__ == "__main__":
    agent = SingleDeepQLearningAgent(load_path="checkpoint/single_checkpoint.pth")
    avg_reward = agent.train(
        save_path="checkpoint/single_checkpoint.pth",
        plot_path="plot/single_training_progress.png",
        log_path="log/single_training_data.txt"
    )
    print(f"Single-agent average reward: {avg_reward:.4f}")