# deep_q_learning_agent.py
# Deep Q-learning agent for Document 3 (Chapter 4)

from environment_doc3 import environment_doc3
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
from parameters import *

class deep_q_learning_agent:
    def __init__(self, num_users=num_users, num_channels=num_channels, dueling=False):
        self.env = environment_doc3(num_users, num_channels)
        self.num_users = num_users
        self.num_channels = num_channels
        self.action_space_size = num_actions_doc3 * num_channels
        self.state_history = [[] for _ in range(num_users)]
        self.action_history = [[] for _ in range(num_users)]
        self.reward_history = [[] for _ in range(num_users)]
        self.next_state_history = [[] for _ in range(num_users)]
        self.models = [self.create_model() for _ in range(num_users)]
        self.target_models = [self.create_model() for _ in range(num_users)]
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.loss_function = tf.keras.losses.Huber()
        self.optimizers = [tf.keras.optimizers.Adam(learning_rate=learning_rate_deepQ) for _ in range(num_users)]

    def create_model(self):
        input_shape = (num_features,)
        X_input = Input(input_shape)
        X = Dense(512, input_shape=input_shape, activation="tanh")(X_input)
        X = Dense(256, activation="tanh")(X)
        X = Dense(64, activation="tanh")(X)
        X = Dense(self.action_space_size, activation="linear")(X)
        model = Model(inputs=X_input, outputs=X)
        return model

    def remember(self, user, state, action, reward, next_state):
        self.state_history[user].append(state)
        self.action_history[user].append(action)
        self.reward_history[user].append(reward)
        self.next_state_history[user].append(next_state)
        if len(self.reward_history[user]) > memory_size:
            del self.state_history[user][0]
            del self.action_history[user][0]
            del self.reward_history[user][0]
            del self.next_state_history[user][0]

    def replay(self, user):
        if len(self.reward_history[user]) < batch_size:
            return
        indices = np.random.choice(len(self.reward_history[user]), size=batch_size)
        state_sample = np.array([self.state_history[user][i] for i in indices]).reshape((batch_size, num_features))
        action_sample = np.array([self.action_history[user][i] for i in indices])
        reward_sample = np.array([self.reward_history[user][i] for i in indices])
        next_state_sample = np.array([self.next_state_history[user][i] for i in indices]).reshape((batch_size, num_features))
        future_rewards = self.target_models[user].predict(next_state_sample, verbose=0)
        updated_q_values = reward_sample + gamma_deepQ * np.max(future_rewards, axis=1)
        mask = tf.one_hot(action_sample, self.action_space_size)
        with tf.GradientTape() as tape:
            q_values = self.models[user](state_sample, training=True)
            q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            loss = self.loss_function(updated_q_values, q_action)
        grads = tape.gradient(loss, self.models[user].trainable_variables)
        self.optimizers[user].apply_gradients(zip(grads, self.models[user].trainable_variables))

    def target_update(self, user):
        self.target_models[user].set_weights(self.models[user].get_weights())

    def get_action(self, user, state):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        possible_actions = self.env.get_possible_action(user)
        if np.random.random() < self.epsilon:
            action_channel = possible_actions[np.random.randint(len(possible_actions))]
            return action_channel, self.action_to_index(action_channel[0], action_channel[1])
        else:
            state = np.reshape(state, (1, num_features))
            q_values = self.models[user].predict(state, verbose=0)[0]
            max_q = -float("inf")
            best_action = possible_actions[0]
            for action_channel in possible_actions:
                action_idx = self.action_to_index(action_channel[0], action_channel[1])
                if q_values[action_idx] > max_q:
                    max_q = q_values[action_idx]
                    best_action = action_channel
            return best_action, self.action_to_index(best_action[0], best_action[1])

    def action_to_index(self, action, channel):
        return action * self.num_channels + channel

    def learn(self, T=T):
        total_rewards = [0] * self.num_users
        total_packets = [0] * self.num_users
        for t in range(T):
            actions = []
            action_indices = []
            for user in range(self.num_users):
                state = self.env.get_state_deep(user)
                action, action_idx = self.get_action(user, state)
                actions.append(action)
                action_indices.append(action_idx)

            rewards, packets = self.env.perform_action(actions)

            for user in range(self.num_users):
                state = self.env.get_state_deep(user)
                next_state = self.env.get_state_deep(user)
                self.remember(user, state, action_indices[user], rewards[user], next_state)
                self.replay(user)
                if (t + 1) % update_target_network == 0:
                    self.target_update(user)
                total_rewards[user] += rewards[user]
                total_packets[user] += packets[user]

            if (t + 1) % step == 0:
                print(f"Iteration {t + 1}, Deep Q-learning Rewards: {total_rewards}, Packets: {total_packets}")
        return total_rewards, total_packets