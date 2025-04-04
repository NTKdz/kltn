# q_learning_agent_pytorch.py
from environment import Environment
from parameters import *
import numpy as np
import random
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, num_users=num_users):
        self.env = Environment(num_users)
        self.num_users = num_users
        self.q_tables = [np.zeros((num_states, num_actions)) for _ in range(num_users)]
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999

    def get_action(self, state_idx, user_idx):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        possible_actions = self.env.get_possible_actions(user_idx)
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            q_values = self.q_tables[user_idx][state_idx]
            return max(possible_actions, key=lambda a: q_values[a])

    def train(self):
        total_reward = 0
        per_user_totals = [0] * self.num_users
        reward_history = []
        per_user_history = [[] for _ in range(self.num_users)]
        state = self.env.get_state()
        for i in range(T):
            state_indices = [self.env.get_discrete_state(u) for u in range(self.num_users)]
            actions = [self.get_action(state_idx, u) for u, state_idx in enumerate(state_indices)]
            total_reward_step, next_state, individual_rewards = self.env.step(actions)
            total_reward += total_reward_step
            for u in range(self.num_users):
                per_user_totals[u] += individual_rewards[u]
                per_user_history[u].append(per_user_totals[u] / (i + 1))
                next_state_idx = self.env.get_discrete_state(u)
                print(f"User {u+1} - State: {state_indices[u]}, Action: {actions[u]}, ")
                print(self.q_tables[u][next_state_idx])
                max_next_q = np.max(self.q_tables[u][next_state_idx])
                current_q = self.q_tables[u][state_indices[u]][actions[u]]
                self.q_tables[u][state_indices[u]][actions[u]] = (
                    (1 - learning_rate_Q) * current_q + 
                    learning_rate_Q * (individual_rewards[u] + gamma_Q * max_next_q)
                )
            reward_history.append(total_reward / (i + 1))
            state = next_state
            if (i + 1) % step == 0:
                avg_total = total_reward / (i + 1)
                avg_per_user = [per_user_totals[u] / (i + 1) for u in range(self.num_users)]
                print(f"Iteration {i + 1}, Avg Total Reward: {avg_total:.4f}, "
                      f"Avg Per-User Rewards: {[f'{r:.4f}' for r in avg_per_user]}")
        avg_per_user_final = [per_user_totals[u] / T for u in range(self.num_users)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(reward_history, label='Total Avg Reward')
        for u in range(self.num_users):
            plt.plot(per_user_history[u], label=f'User {u+1} Avg Reward')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.title('Q-Learning Multi-User Reward Convergence')
        plt.show()
        
        return total_reward / T, avg_per_user_final

if __name__ == "__main__":
    agent = QLearningAgent()
    avg_total_multi, avg_per_user_multi = agent.train()
    print(f"Multi-user Q-learning total average reward: {avg_total_multi:.4f}")
    print(f"Multi-user Q-learning per-user average rewards: {[f'{r:.4f}' for r in avg_per_user_multi]}")
    print(f"Multi-user Q-learning average per-user reward (mean): {np.mean(avg_per_user_multi):.4f}")