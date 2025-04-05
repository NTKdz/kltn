# parameters.py
nu = 0.1  # Probability jammer is idle
arrival_rate = 3  # Data arrival rate
nu_p = [0.6, 0.2, 0.2]  # Jamming power level probabilities
d_t = 1  # Packets per active transmission
e_t = 1  # Energy per packet
d_bj_arr = [1, 2, 3]  # Backscatter packets per jamming level
e_hj_arr = [1, 2, 3]  # Harvested energy per jamming level
dt_ra_arr = [2, 1, 0]  # Rate adaptation packets per jamming level
d_queue_size = 10
e_queue_size = 10
b_dagger = 3  # Fixed backscatter packets
num_users = 2  # Multi-user addition
num_actions = 7  # Actions: idle, transmit, harvest, backscatter, RA1, RA2, RA3
num_states = 2 * (d_queue_size + 1) * (e_queue_size + 1)  # Per user
learning_rate_deepQ = 0.0001
gamma_deepQ = 0.99
num_features = 5  # jammer_state, data_state, energy_state, other_user_action1, other_user_action2
memory_size = 10000
batch_size = 52
update_target_network = 10000
T = 1000000  # Training iterations
step = 10000  # Print reward interval
epsilon_decay = 0.9999
epsilon_min = 0.01

# environment.py
import numpy as np
from scipy.stats import poisson

class Environment:
    def __init__(self, user_id):
        self.user_id = user_id
        self.jammer_state = 0
        self.data_state = 0
        self.energy_state = 0
        self.other_users_actions = [0] * (num_users - 1)  # Track other users' actions

    def get_state_deep(self):
        # State includes jammer, data, energy, and actions of other users
        return np.array([self.jammer_state, self.data_state, self.energy_state] + self.other_users_actions)

    def get_possible_action(self):
        list_actions = [0]  # Stay idle
        if self.jammer_state == 0 and self.data_state > 0 and self.energy_state >= e_t:
            list_actions.append(1)  # Active transmit
        if self.jammer_state == 1:
            list_actions.append(2)  # Harvest energy
            if self.data_state > 0:
                list_actions.append(3)  # Backscatter
                if self.energy_state >= e_t:
                    list_actions.extend([4, 5, 6])  # RA1, RA2, RA3
        return list_actions

    def calculate_reward(self, action, other_users_actions):
        reward = 0
        loss = 0
        collision = any(a in [1, 4, 5, 6] for a in other_users_actions)  # Collision if others transmit
        if action == 0:
            reward = 0
        elif action == 1:  # Active transmit
            reward = self.active_transmit(d_t) if not collision else 0
        elif action == 2:  # Harvest energy
            reward = random.choices(e_hj_arr, nu_p, k=1)[0]
        elif action == 3:  # Backscatter
            d_bj = random.choices(d_bj_arr, nu_p, k=1)[0]
            max_rate = min(b_dagger, self.data_state)
            reward = min(d_bj, max_rate) if not collision else 0
            loss = max_rate - reward if max_rate > reward else 0
        elif action in [4, 5, 6]:  # RA
            max_ra = random.choices(dt_ra_arr, nu_p, k=1)[0]
            reward = self.active_transmit(dt_ra_arr[action - 4]) if not collision else 0
            if dt_ra_arr[action - 4] > max_ra:
                loss = reward
                reward = 0
        return reward, loss

    def active_transmit(self, maximum_transmit_packets):
        num_transmitted = 0
        if 0 < self.data_state < maximum_transmit_packets:
            if self.energy_state >= e_t * self.data_state:
                num_transmitted = self.data_state
            elif self.energy_state >= e_t:
                num_transmitted = self.energy_state // e_t
        else:
            if self.energy_state >= e_t * maximum_transmit_packets:
                num_transmitted = maximum_transmit_packets
            elif self.energy_state >= e_t:
                num_transmitted = self.energy_state // e_t
        return num_transmitted

    def perform_action_deep(self, action, other_users_actions):
        self.other_users_actions = other_users_actions
        reward, loss = self.calculate_reward(action, other_users_actions)
        if action == 1:
            self.data_state -= reward
            self.energy_state -= reward * e_t
        elif action == 2:
            if self.energy_state < e_queue_size:
                self.energy_state += reward
                if self.energy_state > e_queue_size:
                    self.energy_state = e_queue_size
            reward = 0  # Reward is successful packets, not harvested energy
        elif action == 3:
            max_rate = min(b_dagger, self.data_state)
            self.data_state -= max_rate
        elif action in [4, 5, 6]:
            if reward > 0:
                self.data_state -= reward
                self.energy_state -= reward * e_t
            self.data_state -= loss
            self.energy_state -= loss * e_t

        # Data arrival
        data_arrive = poisson.rvs(mu=arrival_rate)
        self.data_state += data_arrive
        if self.data_state > d_queue_size:
            self.data_state = d_queue_size

        # Jammer state
        self.jammer_state = 1 if (self.jammer_state == 0 and np.random.random() <= 1 - nu) else \
                            0 if (self.jammer_state == 1 and np.random.random() <= nu) else self.jammer_state

        next_state = self.get_state_deep()
        return reward, next_state

# deep_q_learning_agent.py
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
import random

class DeepQLearningAgent:
    def __init__(self, user_id):
        self.user_id = user_id
        self.env = Environment(user_id)
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.next_state_history = []
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.epsilon = 1.0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_deepQ)
        self.loss_function = tf.keras.losses.Huber()

    def create_model(self):
        input_shape = (num_features,)
        X_input = Input(input_shape)
        X = Dense(512, activation="tanh")(X_input)
        X = Dense(256, activation="tanh")(X)
        X = Dense(64, activation="tanh")(X)
        X = Dense(num_actions, activation="linear")(X)
        return Model(inputs=X_input, outputs=X)

    def remember(self, state, action, reward, next_state):
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.next_state_history.append(next_state)
        if len(self.reward_history) > memory_size:
            del self.state_history[0], self.action_history[0], self.reward_history[0], self.next_state_history[0]

    def replay(self):
        if len(self.reward_history) < batch_size:
            return
        indices = np.random.choice(len(self.state_history), batch_size)
        state_sample = np.array([self.state_history[i] for i in indices]).reshape((batch_size, num_features))
        action_sample = np.array([self.action_history[i] for i in indices])
        reward_sample = np.array([self.reward_history[i] for i in indices])
        next_state_sample = np.array([self.next_state_history[i] for i in indices]).reshape((batch_size, num_features))

        future_rewards = self.target_model.predict(next_state_sample, verbose=0)
        updated_q_values = reward_sample + gamma_deepQ * tf.reduce_max(future_rewards, axis=1)

        mask = tf.one_hot(action_sample, num_actions)
        with tf.GradientTape() as tape:
            q_values = self.model(state_sample, training=True)
            q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            loss = self.loss_function(updated_q_values, q_action)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def get_action(self, state):
        self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.get_possible_action())
        q_values = self.model.predict(state, verbose=0)[0]
        possible_actions = self.env.get_possible_action()
        return max(possible_actions, key=lambda x: q_values[x])

    def target_update(self):
        self.target_model.set_weights(self.model.get_weights())

# Main simulation
agents = [DeepQLearningAgent(i) for i in range(num_users)]
total_rewards = [0] * num_users

for t in range(T):
    states = [np.reshape(agent.env.get_state_deep(), (1, num_features)) for agent in agents]
    actions = [agent.get_action(states[i]) for i, agent in enumerate(agents)]
    
    # Share actions among users (coordination)
    other_actions = [[actions[j] for j in range(num_users) if j != i] for i in range(num_users)]
    rewards, next_states = zip(*[agent.env.perform_action_deep(actions[i], other_actions[i]) 
                                for i, agent in enumerate(agents)])
    
    for i, agent in enumerate(agents):
        agent.remember(states[i], actions[i], rewards[i], np.reshape(next_states[i], (1, num_features)))
        agent.replay()
        total_rewards[i] += rewards[i]
    
    if (t + 1) % update_target_network == 0:
        for agent in agents:
            agent.target_update()
    
    if (t + 1) % step == 0:
        print(f"Iteration {t + 1}: Rewards {', '.join(f'User {i}: {total_rewards[i] / (t + 1):.4f}' for i in range(num_users))}")