# environment_doc1.py
# Environment for Document 1 (Yao and Jia, 2019)

from parameters import *
import numpy as np

class environment_doc1:
    def __init__(self, num_users=num_users, num_channels=num_channels):
        self.num_users = num_users
        self.num_channels = num_channels
        self.jammer_channel = 0  # Start at channel 0
        self.user_channels = [0] * num_users  # Initial channel for each user
        self.time_slot = 0

    def step(self, actions):
        # Update jammer channel (sweep jammer)
        if self.time_slot * slot_duration >= dwell_time:
            self.jammer_channel = (self.jammer_channel + 1) % self.num_channels
            self.time_slot = 0
        self.time_slot += 1

        # Update user channels based on actions
        self.user_channels = actions

        # Calculate reward for each user
        rewards = [0] * self.num_users
        packets_transmitted = [0] * self.num_users
        for user in range(self.num_users):
            user_channel = self.user_channels[user]
            # Check if user avoids jammer
            if user_channel != self.jammer_channel:
                # Check for mutual interference
                collision = False
                for other_user in range(self.num_users):
                    if other_user != user and self.user_channels[other_user] == user_channel:
                        collision = True
                        break
                if not collision:
                    rewards[user] = 1  # Successful transmission
                    packets_transmitted[user] = 1  # Assume 1 packet per successful transmission

        # State: jammer channel and joint actions
        state = (self.jammer_channel, tuple(self.user_channels))
        return state, rewards, packets_transmitted

    def get_state(self):
        return (self.jammer_channel, tuple(self.user_channels))