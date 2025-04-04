# parameters.py
# Common parameters for both Document 1 and Document 3 methods

# General parameters
num_users = 2
num_channels = 1
T = 100000
runs = 10  # Reduced for faster testing

# Document 1 parameters
dwell_time = 2.28
slot_duration = 1.18
learning_rate_doc1 = 0.1
discount_factor_doc1 = 0.9
epsilon_start_doc1 = 1.0
epsilon_min_doc1 = 0.01
epsilon_decay_doc1 = 0.999954  # Adjusted for faster decay

# Document 3 parameters
nu = 0.1
arrival_rate = 3
nu_p = [0.6, 0.2, 0.2]
d_t = 1
e_t = 1
d_bj_arr = [1, 2, 3]
e_hj_arr = [1, 2, 3]
dt_ra_arr = [2, 1, 0]
d_queue_size = 10
e_queue_size = 10
b_dagger = 3
num_actions_doc3 = 7
num_features = 4

# Learning parameters for Document 3
learning_rate_Q = 0.1
gamma_Q = 0.9
learning_rate_deepQ = 0.0001
gamma_deepQ = 0.99
memory_size = 10000
batch_size = 52
update_target_network = 10000
step = 10000