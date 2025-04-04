# compare_methods.py
# Script to compare Document 1 and Document 3 methods with progress display

from cmma_agent import cmma_agent
from q_learning_agent import q_learning_agent
from deep_q_learning_agent import deep_q_learning_agent
import numpy as np
from parameters import *
from tqdm import tqdm  # Import tqdm for progress bar

def run_comparison(num_users, num_channels, T=T, runs=runs):
    results = {"CMAA": [], "Q-learning": [], "Deep Q-learning": []}
    
    # Use tqdm to show progress for each run
    for run in tqdm(range(runs), desc=f"Running {runs} runs for num_users={num_users}, num_channels={num_channels}"):
        # Document 1: CMAA
        if num_channels > 1:  # Only run in multi-channel case
            agent1 = cmma_agent(num_users, num_channels)
            rewards1, packets1 = agent1.learn(T)
            avg_packets1 = sum(packets1) / T / num_users
            results["CMAA"].append(avg_packets1)

        # Document 3: Q-learning
        agent3_q = q_learning_agent(num_users, num_channels)
        rewards3_q, packets3_q = agent3_q.learn(T)
        avg_packets3_q = sum(packets3_q) / T / num_users
        results["Q-learning"].append(avg_packets3_q)

        # # Document 3: Deep Q-learning
        # agent3_deepq = deep_q_learning_agent(num_users, num_channels)
        # rewards3_deepq, packets3_deepq = agent3_deepq.learn(T)
        # avg_packets3_deepq = sum(packets3_deepq) / T / num_users
        # results["Deep Q-learning"].append(avg_packets3_deepq)

    # Compute averages and standard deviations
    for method in results:
        if results[method]:
            avg = np.mean(results[method])
            std = np.std(results[method])
            print(f"{method}: Throughput = {avg:.2f} ± {std:.2f}")

# Run comparisons
print("Scenario 1: Single-Channel, Single-User, Probabilistic Jammer")
run_comparison(num_users=1, num_channels=1)

print("\nScenario 2: Single-Channel, Multi-User, Probabilistic Jammer")
run_comparison(num_users=2, num_channels=1)

print("\nScenario 3: Multi-Channel, Multi-User, Sweep Jammer")
run_comparison(num_users=2, num_channels=5)