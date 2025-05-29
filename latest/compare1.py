import pandas as pd
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv("log/test_nu1/multi_user_training_data_tdma_reverted_2_rate_3_eps_0.9999_nu_0.1_1.txt", sep="\t")

# Extract agent reward columns
reward_cols = [col for col in df.columns if col.startswith("Avg_Reward_User_")]

# Plot
plt.figure(figsize=(10, 6))
for col in reward_cols:
    plt.plot(df["Iteration"], df[col], label=col)

plt.xlabel("Iteration")
plt.ylabel("Average Reward")
plt.title("Agent Reward Progression Over Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
