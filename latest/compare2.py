import pandas as pd
import matplotlib.pyplot as plt
import re

# Paths and model labels
file_paths = {
    "Model 1": "log/multi_user_training_data_tdma_reverted_2_rate_3_removepriority_removedividetimeslot.txt",
    "Model 2": "log/multi_user_training_data_tdma_reverted_2_rate_3test.txt",
    # "Model 3": "log/multi_user_training_data_tdma_reverted_2_test.txt",
    # "Model 4": "log/multi_user_training_data_tdma_reverted4.txt"
}

data = {}

# Load and process each file
for model_name, path in file_paths.items():
    df = pd.read_csv(path, sep="\t")

    # Detect user columns dynamically
    reward_cols = [col for col in df.columns if re.match(r"Avg_Reward_User_\d+", col)]
    loss_cols = [col for col in df.columns if re.match(r"Avg_Loss_User_\d+", col)]
    packet_loss_cols = [col for col in df.columns if re.match(r"Packet_Loss_Ratio_User_\d+", col)]

    # Compute averages over all users
    df["Avg_Reward"] = df[reward_cols].mean(axis=1)
    df["Avg_Loss"] = df[loss_cols].mean(axis=1)
    df["Packet_Loss_Ratio"] = df[packet_loss_cols].mean(axis=1)

    data[model_name] = df

# Metrics to plot
metrics = {
    "Avg_Total_Reward": "Average Total Reward",
    "Avg_Reward": "Average Reward (All Users)",
    "Packet_Loss_Ratio": "Average Packet Loss Ratio (All Users)",
    "Avg_Loss": "Average Loss (All Users)"
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
