import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
N_USERS = 2
N_CHANNELS = 5
T_SLOTS = 10000
LAMBDA = 0.5  # Reduced for stability
GAMMA = 0.6
EPSILON_START = 0.2
EPSILON_END = 0.01
EPSILON_DECAY = 0.995  # Decay factor per run
PN0 = 20
DWELL_TIME = 2.28e-3
SLOT_DURATION = 1.18e-3
N_RUNS = 200

# Initialize Q-tables
state_space_size = (N_CHANNELS ** N_USERS) * N_CHANNELS
action_space_size = N_CHANNELS
Q_tables = [np.zeros((state_space_size, action_space_size)) for _ in range(N_USERS)]

# Helper functions
def get_state_index(actions, jammer_channel):
    action_index = sum(a * (N_CHANNELS ** i) for i, a in enumerate(actions))
    return action_index * N_CHANNELS + jammer_channel

def get_reward(user_action, jammer_channel, all_actions):
    if user_action == jammer_channel:
        return 0
    if all_actions.count(user_action) > 1:
        return 0
    return 1

def choose_joint_action(state, Q_tables, epsilon):
    if np.random.rand() < epsilon:
        return [np.random.randint(N_CHANNELS) for _ in range(N_USERS)]
    # Compute sum of Q-values for all joint actions
    best_action = None
    best_value = -float('inf')
    for a1 in range(N_CHANNELS):
        for a2 in range(N_CHANNELS):
            joint_action = [a1, a2]
            total_q = sum(Q_tables[n][state, joint_action[n]] for n in range(N_USERS))
            if total_q > best_value:
                best_value = total_q
                best_action = joint_action
    return best_action

# CMAA Simulation
def run_cmaa(run_idx):
    global Q_tables
    Q_tables = [np.zeros((state_space_size, action_space_size)) for _ in range(N_USERS)]
    successes = np.zeros(T_SLOTS)
    jammer_channel = 0
    dwell_counter = 0
    epsilon = EPSILON_START
    
    for t in range(T_SLOTS):
        dwell_counter += SLOT_DURATION
        if dwell_counter >= DWELL_TIME:
            jammer_channel = (jammer_channel + 1) % N_CHANNELS
            dwell_counter = 0
        
        actions = [np.random.randint(N_CHANNELS) for _ in range(N_USERS)]
        state = get_state_index(actions, jammer_channel)
        
        # Choose joint action with coordination
        new_actions = choose_joint_action(state, Q_tables, epsilon)
        
        rewards = [get_reward(a, jammer_channel, new_actions) for a in new_actions]
        successes[t] = sum(rewards) / N_USERS
        
        next_state = get_state_index(new_actions, jammer_channel)
        for n in range(N_USERS):
            Q_n = Q_tables[n]
            max_next_Q = np.max(Q_n[next_state])
            Q_n[state, new_actions[n]] = (1 - LAMBDA) * Q_n[state, new_actions[n]] + \
                                         LAMBDA * (rewards[n] + GAMMA * max_next_Q)
    
    # Compute normalized rate with smoothing
    normalized_rate = []
    for i in range(0, T_SLOTS, PN0):
        window = successes[i:i + PN0]
        if len(window) == PN0:
            normalized_rate.append(np.mean(window))
    return normalized_rate

# Sensing-based Method
def run_sensing_based(run_idx):
    successes = np.zeros(T_SLOTS)
    jammer_channel = 0
    dwell_counter = 0
    
    for t in range(T_SLOTS):
        dwell_counter += SLOT_DURATION
        if dwell_counter >= DWELL_TIME:
            jammer_channel = (jammer_channel + 1) % N_CHANNELS
            dwell_counter = 0
        
        actions = []
        for n in range(N_USERS):
            available = [ch for ch in range(N_CHANNELS) if ch != jammer_channel and ch not in actions]
            if available:
                actions.append(np.random.choice(available))
            else:
                actions.append(np.random.randint(N_CHANNELS))
        
        rewards = [get_reward(a, jammer_channel, actions) for a in actions]
        successes[t] = sum(rewards) / N_USERS
    
    normalized_rate = []
    for i in range(0, T_SLOTS, PN0):
        window = successes[i:i + PN0]
        if len(window) == PN0:
            normalized_rate.append(np.mean(window))
    return normalized_rate

# Independent Q-learning
def run_independent_q(run_idx):
    Q_ind = [np.zeros((N_CHANNELS, N_CHANNELS)) for _ in range(N_USERS)]
    successes = np.zeros(T_SLOTS)
    jammer_channel = 0
    dwell_counter = 0
    epsilon = EPSILON_START
    
    for t in range(T_SLOTS):
        dwell_counter += SLOT_DURATION
        if dwell_counter >= DWELL_TIME:
            jammer_channel = (jammer_channel + 1) % N_CHANNELS
            dwell_counter = 0
        
        actions = []
        for n in range(N_USERS):
            if np.random.rand() < epsilon:
                actions.append(np.random.randint(N_CHANNELS))
            else:
                actions.append(np.argmax(Q_ind[n][jammer_channel]))
        
        rewards = [get_reward(a, jammer_channel, actions) for a in actions]
        successes[t] = sum(rewards) / N_USERS
        
        for n in range(N_USERS):
            Q_n = Q_ind[n]
            max_next_Q = np.max(Q_n[jammer_channel])
            Q_n[jammer_channel, actions[n]] = (1 - LAMBDA) * Q_n[jammer_channel, actions[n]] + \
                                              LAMBDA * (rewards[n] + GAMMA * max_next_Q)
    
    normalized_rate = []
    for i in range(0, T_SLOTS, PN0):
        window = successes[i:i + PN0]
        if len(window) == PN0:
            normalized_rate.append(np.mean(window))
    return normalized_rate

# Run simulations with progress tracking
def run_simulations():
    cmaa_all_rates = []
    sensing_all_rates = []
    ind_q_all_rates = []
    
    print("Running simulations...")
    for run in tqdm(range(N_RUNS), desc="Simulation Progress"):
        cmaa_rates = run_cmaa(run)
        sensing_rates = run_sensing_based(run)
        ind_q_rates = run_independent_q(run)
        
        cmaa_all_rates.append(cmaa_rates)
        sensing_all_rates.append(sensing_rates)
        ind_q_all_rates.append(ind_q_rates)
        
        if (run + 1) % 20 == 0:
            cmaa_avg = np.mean(cmaa_all_rates[-20:], axis=0)[-1]
            sensing_avg = np.mean(sensing_all_rates[-20:], axis=0)[-1]
            ind_q_avg = np.mean(ind_q_all_rates[-20:], axis=0)[-1]
            print(f"Run {run + 1}/{N_RUNS}:")
            print(f"  CMAA Avg Normalized Rate (last 20 runs): {cmaa_avg:.3f}")
            print(f"  Sensing Avg Normalized Rate (last 20 runs): {sensing_avg:.3f}")
            print(f"  Ind Q Avg Normalized Rate (last 20 runs): {ind_q_avg:.3f}")
    
    cmaa_rates = np.mean(cmaa_all_rates, axis=0)
    sensing_rates = np.mean(sensing_all_rates, axis=0)
    ind_q_rates = np.mean(ind_q_all_rates, axis=0)
    
    # Smooth the curves using a moving average
    window_size = 10
    cmaa_rates = np.convolve(cmaa_rates, np.ones(window_size)/window_size, mode='valid')
    sensing_rates = np.convolve(sensing_rates, np.ones(window_size)/window_size, mode='valid')
    ind_q_rates = np.convolve(ind_q_rates, np.ones(window_size)/window_size, mode='valid')
    
    return cmaa_rates, sensing_rates, ind_q_rates

# Execute simulations
cmaa_rates, sensing_rates, ind_q_rates = run_simulations()

# Plot results
plt.figure(figsize=(10, 6))
time_steps = np.arange(len(cmaa_rates)) * PN0
plt.plot(time_steps, cmaa_rates, label="CMAA", color='blue')
plt.plot(time_steps, sensing_rates, label="Sensing-based", color='orange')
plt.plot(time_steps, ind_q_rates, label="Independent Q-learning", color='green')
plt.xlabel("Time Slots")
plt.ylabel("Normalized Rate")
plt.title("Performance Comparison of Normalized Rate")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()