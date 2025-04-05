# parameters.py
import numpy as np
from scipy.stats import poisson

# Existing parameters (example values)
T = 100000
step = 10000
num_users = 2
d_t = 2
e_t = 1
b_dagger = 3
arrival_rate = 1.5
d_queue_size = 10
e_queue_size = 10
nu = 0.3
learning_rate_deepQ = 0.001
gamma_deepQ = 0.99
memory_size = 10000
batch_size = 64
num_features = 1 + 2 * num_users  # jammer_state + data/energy per user
num_actions = 7  # Original

# New parameters for frequency hopping
num_channels = 5
num_actions = 7 * num_channels  # 35 actions (7 base actions x 5 channels)
dt_ra_arr = [1, 2, 3]  # Rate adaptation options
nu_p = [0.5, 0.3, 0.2]  # Probabilities for random choices
e_hj_arr = [1, 2, 3]    # Harvesting energy options
d_bj_arr = [1, 2, 3]    # Backscatter data options