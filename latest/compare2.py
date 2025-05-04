import pandas as pd
import matplotlib.pyplot as plt
import re

# Paths and model labels
file_paths = {
    # "Model DQN1": "log/test_final10/multi_user_training_data_tdma_reverted_1_rate_3_eps_0.9999_1.txt",
    # "Model DQN2": "log/test_final10/multi_user_training_data_tdma_reverted_2_rate_3_eps_0.9999_1.txt",
    # "Model DQN3": "log/test_final10/multi_user_training_data_tdma_reverted_3_rate_3_eps_0.9999_1.txt",
    # "Model DQN4": "log/test_final10/multi_user_training_data_tdma_reverted_4_rate_3_eps_0.9999_1.txt",
    # "Model DQN6": "log/test_final10/multi_user_training_data_tdma_reverted_6_rate_3_eps_0.9999_1.txt",
    # "Model DQN8": "log/test_final10/multi_user_training_data_tdma_reverted_8_rate_3_eps_0.9999_1.txt",
    # "Model DQN10": "log/test_final10/multi_user_training_data_tdma_reverted_10_rate_3_eps_0.9999_1.txt",
    # "Model DQN1x": "log/test_final10/multi_user_training_data_tdma_reverted_1_rate_3_eps_0.9999_2.txt",
    # "Model DQN2x": "log/test_final10/multi_user_training_data_tdma_reverted_2_rate_3_eps_0.9999_3.txt",
    # "Model DQN3x": "log/test_final10/multi_user_training_data_tdma_reverted_3_rate_3_eps_0.9999_2.txt",
    # "Model DQN4x": "log/test_final10/multi_user_training_data_tdma_reverted_4_rate_3_eps_0.9999_3.txt",
    # "Model DQN6x": "log/test_final10/multi_user_training_data_tdma_reverted_6_rate_3_eps_0.9999_2.txt",
    # "Model DQN8x": "log/test_final9/multi_user_training_data_tdma_reverted_8_rate_3_eps_0.9999_2.txt",
    #  "Model DQN10x": "log/test_final9/multi_user_training_data_tdma_reverted_10_rate_3_eps_0.9999_2.txt",
    # "Model QL1": "log/test_q1/multi_user_training_data_qlearning_1_rate_3_eps_0.99991.txt",
    # "Model QL2": "log/test_q1/multi_user_training_data_qlearning_2_rate_3_eps_0.99991.txt",
    # "Model QL2test3x": "log/test_q1/multi_user_training_data_qlearning_2_rate_3_eps_0.9991.txt",
    # "Model QL2test3xx": "log/test_q1/multi_user_training_data_qlearning_2_rate_3_eps_0.991.txt",
    # "Model QL2test3xxx": "log/test_q1/multi_user_training_data_qlearning_2_rate_3_eps_0.91.txt",
    # "Model QL3": "log/test_q1/multi_user_training_data_qlearning_3_rate_3_eps_0.99991.txt",
    # "Model QL4": "log/test_q1/multi_user_training_data_qlearning_4_rate_3_eps_0.99991.txt",
    # "Model QL6": "log/test_q1/multi_user_training_data_qlearning_6_rate_3_eps_0.99991.txt",
    # "Model QL8": "log/test_q1/multi_user_training_data_qlearning_8_rate_3_eps_0.99991.txt",
    # "Model QL10": "log/test_q1/multi_user_training_data_qlearning_10_rate_3_eps_0.99991.txt",
    # "Model QL12": "log/test_q1/multi_user_training_data_qlearning_12_rate_3_eps_0.99991.txt",
    # "Model QL14": "log/test_q1/multi_user_training_data_qlearning_14_rate_3_eps_0.99991.txt",
}

file_paths1 = {
    "Model DQN1": "log/test_final10/multi_user_training_data_tdma_reverted_1_rate_3_eps_0.9999_1.txt",
    "Model DQN2": "log/test_final10/multi_user_training_data_tdma_reverted_2_rate_3_eps_0.9999_1.txt",
    "Model DQN3": "log/test_final10/multi_user_training_data_tdma_reverted_3_rate_3_eps_0.9999_1.txt",
    "Model DQN1_HTT": "log/test_final10/multi_user_training_data_tdma_reverted_1_rate_3_eps_0.9999_HTT1.txt",
    "Model DQN2_HTT": "log/test_final10/multi_user_training_data_tdma_reverted_2_rate_3_eps_0.9999_HTT1.txt",
    "Model DQN3_HTT": "log/test_final10/multi_user_training_data_tdma_reverted_3_rate_3_eps_0.9999_HTT1.txt",
}

data = {}

# Load and process each file
for model_name, path in file_paths1.items():
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

    # df = df[df["Iteration"] <= 100000]

    data[model_name] = df

# Metrics to plot
metrics = {
    "Avg_Total_Reward": "Average Total Throughput (packets/time unit)",
    "Avg_Reward": "Average Throughput (All Users) (packets/time unit)",
    "Packet_Loss_Ratio": "Average Packet Loss Ratio (All Users)",
    "Packet_Loss_By_Transmit": "Average Packet Loss by transmission (All Users)",
    # "Avg_Loss": "Average Loss (All Users)",
    # "Count_Change": "Count Change"
}

# Plot each metric
for metric_key, metric_label in metrics.items():
    plt.figure(figsize=(10, 6))
    for model_name, df in data.items():
        if metric_key in df.columns:
            plt.plot(df["Iteration"], df[metric_key],
                     label=model_name, linewidth=1.2, marker='o', markersize=4)

    plt.title(f"{metric_label} Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel(metric_label)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
