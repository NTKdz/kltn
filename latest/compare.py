import pandas as pd
import matplotlib.pyplot as plt

# Use a clean style
# plt.style.use('ggplot')

# Load all datasets
single_df = pd.read_csv('log/single_training_data.txt', sep='\t')
multi2_df = pd.read_csv('log/multi_user_training_data_tdma_2.txt', sep='\t')
multi4_df = pd.read_csv('log/multi_user_training_data_tdma_4.txt', sep='\t')
multi6_df = pd.read_csv('log/multi_user_training_data_tdma_6.txt', sep='\t')
multi8_df = pd.read_csv('log/multi_user_training_data_tdma_8.txt', sep='\t')
# multi10_df = pd.read_csv('log/multi_user_training_data_tdma_10.txt', sep='\t')

# =====================
# Plot Average Reward
# =====================
plt.figure(figsize=(12, 7))
plt.plot(single_df['Iteration'], single_df['Avg_Reward'], label='1 User', marker='o', markersize=4)
avg_reward_2 = (multi2_df['Avg_Reward_User_0'] + multi2_df['Avg_Reward_User_1']) / 2
plt.plot(multi2_df['Iteration'], avg_reward_2, label='2 Users (avg)', marker='s', markersize=4)
avg_reward_4 = (
    multi4_df['Avg_Reward_User_0'] + multi4_df['Avg_Reward_User_1'] +
    multi4_df['Avg_Reward_User_2'] + multi4_df['Avg_Reward_User_3']
) / 4
plt.plot(multi4_df['Iteration'], avg_reward_4, label='4 Users (avg)', marker='^', markersize=4)
avg_reward_6 = multi6_df[[f'Avg_Reward_User_{i}' for i in range(6)]].mean(axis=1)
plt.plot(multi6_df['Iteration'], avg_reward_6, label='6 Users (avg)', marker='v', markersize=4)
avg_reward_8 = multi8_df[[f'Avg_Reward_User_{i}' for i in range(8)]].mean(axis=1)
plt.plot(multi8_df['Iteration'], avg_reward_8, label='8 Users (avg)', marker='<', markersize=4)
# avg_reward_10 = multi10_df[[f'Avg_Reward_User_{i}' for i in range(10)]].mean(axis=1)
# plt.plot(multi10_df['Iteration'], avg_reward_10, label='10 Users (avg)', marker='D', markersize=4)

plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Iteration for Different Numbers of Users')
# plt.xlim(0, 100000)  # Limit X-axis to 100k
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================
# Plot Packet Loss Comparison
# =============================
plt.figure(figsize=(12, 7))
plt.plot(single_df['Iteration'], single_df['Packet_Loss_Ratio'], label='1 User', marker='o', markersize=4)
pkt_loss_2 = (multi2_df['Packet_Loss_Ratio_User_0'] + multi2_df['Packet_Loss_Ratio_User_1']) / 2
plt.plot(multi2_df['Iteration'], pkt_loss_2, label='2 Users (avg)', marker='s', markersize=4)
pkt_loss_4 = (
    multi4_df['Packet_Loss_Ratio_User_0'] + multi4_df['Packet_Loss_Ratio_User_1'] +
    multi4_df['Packet_Loss_Ratio_User_2'] + multi4_df['Packet_Loss_Ratio_User_3']
) / 4
plt.plot(multi4_df['Iteration'], pkt_loss_4, label='4 Users (avg)', marker='^', markersize=4)
pkt_loss_6 = multi6_df[[f'Packet_Loss_Ratio_User_{i}' for i in range(6)]].mean(axis=1)
plt.plot(multi6_df['Iteration'], pkt_loss_6, label='6 Users (avg)', marker='v', markersize=4)
pkt_loss_8 = multi8_df[[f'Packet_Loss_Ratio_User_{i}' for i in range(8)]].mean(axis=1)
plt.plot(multi8_df['Iteration'], pkt_loss_8, label='8 Users (avg)', marker='<', markersize=4)
# pkt_loss_10 = multi10_df[[f'Packet_Loss_Ratio_User_{i}' for i in range(10)]].mean(axis=1)
# plt.plot(multi10_df['Iteration'], pkt_loss_10, label='10 Users (avg)', marker='D', markersize=4)

plt.xlabel('Iteration')
plt.ylabel('Average Packet Loss Ratio')
plt.title('Packet Loss Ratio vs Iteration for Different Numbers of Users')
# plt.xlim(0, 100000)  # Limit X-axis to 100k
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
