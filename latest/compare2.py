import pandas as pd
import matplotlib.pyplot as plt
import re

# Paths and model labels
file_paths = {
    # "Model 1": "log/test_final3/multi_user_training_data_tdma_reverted_1_rate_3.txt",
    # "Model 2": "log/test_final3/multi_user_training_data_tdma_reverted_2_rate_3.txt",
    # "Model 3": "log/test_final3/multi_user_training_data_tdma_reverted_3_rate_3.txt",
    # "Model 4": "log/test_final3/multi_user_training_data_tdma_reverted_4_rate_3.txt",
    # "Model 5": "log/test_final3/multi_user_training_data_tdma_reverted_5_rate_3.txt",
    # "Model 6": "log/test_final3/multi_user_training_data_tdma_reverted_6_rate_3.txt",
    # "Model 8": "log/test_final3/multi_user_training_data_tdma_reverted_8_rate_3.txt",
    # "Model 10": "log/test_final3/multi_user_training_data_tdma_reverted_10_rate_3.txt",
    "Model q1": "log/test/multi_user_training_data_qlearning_3_rate_3.txt",
    "Model q2": "log/test_final3/multi_user_training_data_qlearning_3_rate_3.txt",
    # "Model q3": "log/test_final3/multi_user_training_data_qlearning_3_rate_3.txt",
    # "Model q4": "log/test_final3/multi_user_training_data_qlearning_4_rate_3.txt",
}

data = {}

# Load and process each file
for model_name, path in file_paths.items():
    df = pd.read_csv(path, sep="\t")

    # Detect user columns dynamically
    reward_cols = [col for col in df.columns if re.match(
        r"Avg_Reward_User_\d+", col)]
    loss_cols = [col for col in df.columns if re.match(
        r"Avg_Loss_User_\d+", col)]
    packet_loss_cols = [col for col in df.columns if re.match(
        r"Packet_Loss_Ratio_User_\d+", col)]
    transmit_loss_cols = [col for col in df.columns if re.match(
        r"Packet_Loss_By_Transmit_Ratio_User_\d+", col)]

    # Compute averages over all users
    df["Avg_Reward"] = df[reward_cols].mean(axis=1)
    df["Avg_Loss"] = df[loss_cols].mean(axis=1)
    df["Packet_Loss_Ratio"] = df[packet_loss_cols].mean(axis=1)
    df["Packet_Loss_By_Transmit"] = df[transmit_loss_cols].mean(axis=1)

    data[model_name] = df

# Metrics to plot
metrics = {
    "Avg_Total_Reward": "Average Total Reward",
    "Avg_Reward": "Average Reward (All Users)",
    "Packet_Loss_Ratio": "Average Packet Loss Ratio (All Users)",
    "Packet_Loss_By_Transmit": "Average Packet Loss by Transmit (All Users)",
    "Avg_Loss": "Average Loss (All Users)",
    "Count_Change": "Count Change"
}

# Plot each metric
for metric_key, metric_label in metrics.items():
    plt.figure(figsize=(10, 6))
    for model_name, df in data.items():
        if metric_key in df.columns:
            plt.plot(df["Iteration"], df[metric_key],
                     marker='o', label=model_name)

    plt.title(f"{metric_label} Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel(metric_label)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
