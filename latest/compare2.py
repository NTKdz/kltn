import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df1 = pd.read_csv("log/multi_user_training_data_tdma_joint_penalty.txt", sep="\t")
df2 = pd.read_csv("log/multi_user_training_data_tdma_joint_penalty0.5.txt", sep="\t")
df3 = pd.read_csv("log\multi_user_training_data_tdma_joint4_exploration.txt", sep="\t")

# Dictionary of models
data = {
    "Model 1": df1,
    "Model 2": df2,
    "Model 3": df3
}

# Precompute the averaged columns for each DataFrame
for df in data.values():
    df["Avg_Reward"] = (df["Avg_Reward_User_0"] + df["Avg_Reward_User_1"]) / 2
    df["Packet_Loss_Ratio"] = (df["Packet_Loss_Ratio_User_0"] + df["Packet_Loss_Ratio_User_1"]) / 2
    df["Avg_Loss"] = (df["Avg_Loss_User_0"] + df["Avg_Loss_User_1"]) / 2

# Metrics to plot: single line per model
metrics = {
    "Avg_Total_Reward": "Average Total Reward",
    "Avg_Reward": "Average Reward (Users 0 & 1)",
    "Packet_Loss_Ratio": "Average Packet Loss Ratio (Users 0 & 1)",
    "Avg_Loss": "Average Loss (Users 0 & 1)"
}

# Plot each metric
for metric_key, metric_label in metrics.items():
    plt.figure(figsize=(10, 6))

    for model_name, df in data.items():
        plt.plot(df["Iteration"], df[metric_key], marker='o', label=model_name)

    plt.title(f"{metric_label} Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel(metric_label)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
