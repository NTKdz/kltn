import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Define the directory containing the files
directory = "log/test_nu"  # Matches your screenshot folder name

# Initialize dictionaries to store files for "Proposed Method" (normal) and "HTT"
normal_files = {}
htt_files = {}

# List all files in the directory and categorize them
for filename in os.listdir(directory):
    if not filename.endswith(".txt"):
        continue
    # Extract nu value using regex
    match = re.search(r"nu_(\d+\.\d+)", filename)
    if match:
        nu = float(match.group(1))
        filepath = os.path.join(directory, filename)
        if "HTT1" in filename:
            htt_files[nu] = filepath
        if "_1" in filename and "HTT1" not in filename:
            normal_files[nu] = filepath

# Ensure nu values are sorted
nu_values = sorted(set(normal_files.keys()).intersection(htt_files.keys()))
if not nu_values:
    raise ValueError("No matching nu values found between normal and HTT files.")

# Data storage for final results
results = {
    "Proposed Method": {nu: {} for nu in nu_values},
    "HTT": {nu: {} for nu in nu_values}
}

# Metrics to extract
metrics = {
    "Avg_Total_Reward": "Total Throughput (packets per time slot)",
    "Avg_Reward": "Average Throughput (packets per time slot)",
    "Packet_Loss_Ratio": "Average Packet Loss Ratio",
    "Packet_Loss_By_Transmit": "Average Packet Loss by Transmission",
    # "Energy_Efficiency": "Energy Efficiency (packets per joule)"  # Optional, uncomment if needed
}

# Load and process each file to extract final metrics
for nu in nu_values:
    try:
        # Process "Proposed Method" (normal) file
        if nu in normal_files:
            df = pd.read_csv(normal_files[nu], sep="\t")
            # Detect user columns dynamically
            reward_cols = [col for col in df.columns if re.match(r"Avg_Reward_User_\d+", col)]
            loss_cols = [col for col in df.columns if re.match(r"Avg_Loss_User_\d+", col)]
            packet_loss_cols = [col for col in df.columns if re.match(r"Packet_Loss_Ratio_User_\d+", col)]
            transmit_loss_cols = [col for col in df.columns if re.match(r"Packet_Loss_By_Transmit_Ratio_User_\d+", col)]

            # Compute averages over all users
            df["Avg_Reward"] = df[reward_cols].mean(axis=1)
            df["Avg_Loss"] = df[loss_cols].mean(axis=1)
            df["Packet_Loss_Ratio"] = df[packet_loss_cols].mean(axis=1)
            df["Packet_Loss_By_Transmit"] = df[transmit_loss_cols].mean(axis=1)

            # Extract the final result (last row)
            final_row = df.iloc[-1]
            print(f"Final row for nu={nu} in normal file: {final_row}")
            final_iteration = int(final_row["Iteration"])
            for metric_key, metric_label in metrics.items():
                if metric_key in df.columns:
                    results["Proposed Method"][nu][metric_key] = final_row[metric_key]
                else:
                    print(f"Warning: Metric {metric_key} not found in normal file for nu={nu}")

        # Process "HTT" file
        if nu in htt_files:
            df = pd.read_csv(htt_files[nu], sep="\t")
            # Detect user columns dynamically
            reward_cols = [col for col in df.columns if re.match(r"Avg_Reward_User_\d+", col)]
            loss_cols = [col for col in df.columns if re.match(r"Avg_Loss_User_\d+", col)]
            packet_loss_cols = [col for col in df.columns if re.match(r"Packet_Loss_Ratio_User_\d+", col)]
            transmit_loss_cols = [col for col in df.columns if re.match(r"Packet_Loss_By_Transmit_Ratio_User_\d+", col)]

            # Compute averages over all users
            df["Avg_Reward"] = df[reward_cols].mean(axis=1)
            df["Avg_Loss"] = df[loss_cols].mean(axis=1)
            df["Packet_Loss_Ratio"] = df[packet_loss_cols].mean(axis=1)
            df["Packet_Loss_By_Transmit"] = df[transmit_loss_cols].mean(axis=1)

            # Extract the final result (last row)
            final_row = df.iloc[-1]
            print(f"Final row for nu={nu} in HTT file: {final_row}")
            final_iteration = int(final_row["Iteration"])
            for metric_key, metric_label in metrics.items():
                if metric_key in df.columns:
                    results["HTT"][nu][metric_key] = final_row[metric_key]
                else:
                    print(f"Warning: Metric {metric_key} not found in HTT file for nu={nu}")

        print(f"Processed files for nu={nu} at final iteration {final_iteration}")

    except Exception as e:
        print(f"Error processing files for nu={nu}: {e}")

# Create a summary DataFrame for easier comparison
summary_data = []
for nu in nu_values:
    for method in ["Proposed Method", "HTT"]:
        row = {"nu": nu, "Method": method}
        for metric_key in metrics.keys():
            row[metric_key] = results[method][nu].get(metric_key, None)
        summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print("\nSummary of Final Results at Last Iteration:")
print(summary_df)

# Optional: Compute and add energy efficiency if Energy_used is present
if "Energy_used" in df.columns:
    for nu in nu_values:
        for method in ["Proposed Method", "HTT"]:
            if "Avg_Total_Reward" in results[method][nu] and "Energy_used" in df.columns:
                final_energy = df.iloc[-1]["Energy_used"]
                results[method][nu]["Energy_Efficiency"] = results[method][nu]["Avg_Total_Reward"] / final_energy
                summary_df.loc[(summary_df["nu"] == nu) & (summary_df["Method"] == method), "Energy_Efficiency"] = results[method][nu]["Energy_Efficiency"]
    metrics["Energy_Efficiency"] = "Energy Efficiency (packets per joule)"
    print("\nSummary of Final Results with Energy Efficiency:")
    print(summary_df)

# Plot each metric as a function of nu
for metric_key, metric_label in metrics.items():
    plt.figure(figsize=(8, 5))
    for method in ["Proposed Method", "HTT"]:
        method_data = summary_df[summary_df["Method"] == method]
        plt.plot(method_data["nu"], method_data[metric_key], label=method, marker='o', linewidth=1.2)

    # plt.title(f"{metric_label} vs. Idle Channel Probability (\( \nu \))")
    plt.xlabel("Jammer Idle Probability")
    plt.ylabel(metric_label)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()