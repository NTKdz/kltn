# =======================
# Plot Total Reward Comparison
# =======================
from matplotlib import pyplot as plt
import pandas as pd

single_df = pd.read_csv('log/single_training_data.txt', sep='\t')
multi2_df = pd.read_csv('log/multi_user_training_data_tdma_2.txt', sep='\t')
multi4_df = pd.read_csv('log/multi_user_training_data_tdma_4.txt', sep='\t')
multi6_df = pd.read_csv('log/multi_user_training_data_tdma_6.txt', sep='\t')
multi8_df = pd.read_csv('log/multi_user_training_data_tdma_8.txt', sep='\t')

plt.figure(figsize=(12, 7))

# Single-agent reward (1 user)
plt.plot(single_df['Iteration'], single_df['Avg_Reward'], label='1 User (Avg)', marker='o', markersize=4)

# Multi-agent total rewards
plt.plot(multi2_df['Iteration'], multi2_df['Avg_Total_Reward'], label='2 Users (Total)', marker='s', markersize=4)
plt.plot(multi4_df['Iteration'], multi4_df['Avg_Total_Reward'], label='4 Users (Total)', marker='^', markersize=4)
plt.plot(multi6_df['Iteration'], multi6_df['Avg_Total_Reward'], label='6 Users (Total)', marker='v', markersize=4)
plt.plot(multi8_df['Iteration'], multi8_df['Avg_Total_Reward'], label='8 Users (Total)', marker='<', markersize=4)
# plt.plot(multi10_df['Iteration'], multi10_df['Avg_Total_Reward'], label='10 Users (Total)', marker='D', markersize=4)

plt.xlabel('Iteration')
plt.ylabel('Total Reward')
plt.title('Total Reward vs Iteration (TDMA Multi-Agent vs Single Agent)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =============================
# Plot Average Packet Loss Comparison
# =============================
plt.figure(figsize=(12, 7))

# Single-agent packet loss
plt.plot(single_df['Iteration'], single_df['Packet_Loss_Ratio'], label='1 User', marker='o', markersize=4)

# Multi-user average packet loss ratios
pkt_loss_2 = (multi2_df['Packet_Loss_Ratio_User_0'] + multi2_df['Packet_Loss_Ratio_User_1']) / 2
plt.plot(multi2_df['Iteration'], pkt_loss_2, label='2 Users (Avg)', marker='s', markersize=4)

pkt_loss_4 = (
    multi4_df['Packet_Loss_Ratio_User_0'] + multi4_df['Packet_Loss_Ratio_User_1'] +
    multi4_df['Packet_Loss_Ratio_User_2'] + multi4_df['Packet_Loss_Ratio_User_3']
) / 4
plt.plot(multi4_df['Iteration'], pkt_loss_4, label='4 Users (Avg)', marker='^', markersize=4)

pkt_loss_6 = multi6_df[[f'Packet_Loss_Ratio_User_{i}' for i in range(6)]].mean(axis=1)
plt.plot(multi6_df['Iteration'], pkt_loss_6, label='6 Users (Avg)', marker='v', markersize=4)

pkt_loss_8 = multi8_df[[f'Packet_Loss_Ratio_User_{i}' for i in range(8)]].mean(axis=1)
plt.plot(multi8_df['Iteration'], pkt_loss_8, label='8 Users (Avg)', marker='<', markersize=4)

# Uncomment if you have data
# pkt_loss_10 = multi10_df[[f'Packet_Loss_Ratio_User_{i}' for i in range(10)]].mean(axis=1)
# plt.plot(multi10_df['Iteration'], pkt_loss_10, label='10 Users (Avg)', marker='D', markersize=4)

plt.xlabel('Iteration')
plt.ylabel('Average Packet Loss Ratio')
plt.title('Average Packet Loss Ratio vs Iteration for Different Numbers of Users')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()