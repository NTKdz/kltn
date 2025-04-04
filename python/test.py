import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import time
from tqdm import tqdm

class Environment:
    """Environment class that simulates the wireless network with jamming"""
    
    def __init__(self, num_users, num_channels, jammer_type='sweep'):
        self.num_users = num_users
        self.num_channels = num_channels
        self.jammer_type = jammer_type
        self.jammer_channel = 0  # Initial jamming channel
        self.user_channels = np.zeros(num_users, dtype=int)  # Channels selected by users
        self.dwell_time = 2.28  # ms, dwelling time for sweep jammer
        self.current_time = 0  # Current simulation time in ms
        self.slot_time = 1.18  # ms, total time for one slot (transmission + sensing + ACK + learning)
        self.tx_time = 0.98  # ms, transmission time
        self.overhead_time = 0.2  # ms, sensing + ACK + learning time
        self.sweep_direction = 1  # 1 for increasing, -1 for decreasing
        
    def update_jammer(self):
        """Update jammer's channel based on sweeping pattern"""
        if self.jammer_type == 'sweep':
            # Check if it's time to change the jamming channel
            if self.current_time % self.dwell_time < self.overhead_time:
                # Move to next channel in sweeping pattern
                self.jammer_channel = (self.jammer_channel + self.sweep_direction) % self.num_channels
                if self.jammer_channel == 0 or self.jammer_channel == self.num_channels - 1:
                    self.sweep_direction *= -1  # Change direction
                    
    def step(self, actions):
        """
        Execute one time step in the environment
        
        Args:
            actions: List of channel selections for each user
            
        Returns:
            next_state: The new state after actions
            rewards: Rewards for each user
        """
        # Update time
        self.current_time += self.slot_time
        
        # Update jammer's channel
        self.update_jammer()
        
        # Update user channels based on actions
        self.user_channels = np.array(actions)
        
        # Calculate rewards
        rewards = np.zeros(self.num_users)
        for user_idx in range(self.num_users):
            user_channel = self.user_channels[user_idx]
            
            # Check if jammed
            jammed = (user_channel == self.jammer_channel)
            
            # Check if experiencing mutual interference
            interfered = False
            for other_user in range(self.num_users):
                if other_user != user_idx and self.user_channels[other_user] == user_channel:
                    interfered = True
                    break
            
            # Assign reward according to equation (2) in the paper
            if not jammed and not interfered:
                rewards[user_idx] = 1
            else:
                rewards[user_idx] = 0
                
        # Construct the next state (joint action profile and jammer channel)
        next_state = (tuple(self.user_channels), self.jammer_channel)
        
        return next_state, rewards
    
    def reset(self):
        """Reset the environment to initial state"""
        self.jammer_channel = 0
        self.user_channels = np.zeros(self.num_users, dtype=int)
        self.current_time = 0
        self.sweep_direction = 1
        return (tuple(self.user_channels), self.jammer_channel)


class CollaborativeMultiAgentAntiJamming:
    """Implementation of the CMAA algorithm from the paper"""
    
    def __init__(self, num_users, num_channels, learning_rate=0.8, discount_factor=0.6, exploration_rate=0.2):
        self.num_users = num_users
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Initialize Q-values for each user
        # Q-values are stored as dictionaries where keys are (state, action) tuples
        self.q_values = [{} for _ in range(num_users)]
        
        # Generate all possible joint actions
        self.joint_actions = self._generate_joint_actions()
        
    def _generate_joint_actions(self):
        """Generate all possible joint actions for all users"""
        if self.num_users == 1:
            return [[i] for i in range(self.num_channels)]
        
        actions = []
        for action in range(self.num_channels):
            sub_actions = self._generate_joint_actions_recursive(self.num_users - 1)
            for sub_action in sub_actions:
                actions.append([action] + sub_action)
        return actions
    
    def _generate_joint_actions_recursive(self, users_left):
        """Recursively generate joint actions for remaining users"""
        if users_left == 1:
            return [[i] for i in range(self.num_channels)]
        
        actions = []
        for action in range(self.num_channels):
            sub_actions = self._generate_joint_actions_recursive(users_left - 1)
            for sub_action in sub_actions:
                actions.append([action] + sub_action)
        return actions
        
    def get_q_value(self, user_idx, state, action):
        """Get Q-value for a user-state-action combination, initialize if needed"""
        state_action = (state, tuple(action))
        if state_action not in self.q_values[user_idx]:
            self.q_values[user_idx][state_action] = 0.0
        return self.q_values[user_idx][state_action]
    
    def update_q_value(self, user_idx, state, action, reward, next_state):
        """Update Q-value for a user according to equation (3)"""
        state_action = (state, tuple(action))
        
        # Get current Q-value
        current_q = self.get_q_value(user_idx, state, action)
        
        # Find optimal joint action for next state
        optimal_next_action = self.get_optimal_joint_action(next_state)
        
        # Get next state value according to equation (4)
        next_state_value = self.get_q_value(user_idx, next_state, optimal_next_action)
        
        # Update Q-value using equation (3)
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * next_state_value)
        
        # Store updated Q-value
        self.q_values[user_idx][state_action] = new_q
    
    def get_optimal_joint_action(self, state):
        """Find optimal joint action using equation (5)"""
        best_value = float('-inf')
        best_action = None
        
        for action in self.joint_actions:
            # Sum Q-values of all users for this state-action pair
            total_q = sum(self.get_q_value(user_idx, state, action) for user_idx in range(self.num_users))
            
            if total_q > best_value:
                best_value = total_q
                best_action = action
                
        return best_action
    
    def select_action(self, state):
        """Select action based on ε-greedy policy"""
        if random.random() < self.exploration_rate:
            # Explore: choose random joint action
            return random.choice(self.joint_actions)
        else:
            # Exploit: choose optimal joint action
            return self.get_optimal_joint_action(state)


def run_simulation(num_episodes=1000, num_steps=100):
    """Run CMAA simulation for specified number of episodes"""
    # Initialize environment and agent
    num_users = 2
    num_channels = 5
    env = Environment(num_users=num_users, num_channels=num_channels)
    agent = CollaborativeMultiAgentAntiJamming(
        num_users=num_users, 
        num_channels=num_channels,
        learning_rate=0.8,
        discount_factor=0.6,
        exploration_rate=0.2
    )
    
    # Statistics tracking
    normalized_rates = []
    avg_rewards_history = []
    window_size = 20  # Calculate rate over this many packets
    success_counts = np.zeros(window_size)
    
    # Run simulation with progress bar
    for episode in tqdm(range(num_episodes), desc="Training CMAA"):
        state = env.reset()
        episode_rewards = np.zeros(num_users)
        
        for step in range(num_steps):
            # Select action according to the CMAA algorithm
            joint_action = agent.select_action(state)
            
            # Take action and observe next state and rewards
            next_state, rewards = env.step(joint_action)
            
            # Update Q-values for all users
            for user_idx in range(num_users):
                agent.update_q_value(user_idx, state, joint_action, rewards[user_idx], next_state)
            
            # Accumulate rewards
            episode_rewards += rewards
            
            # Update success counts for normalized rate calculation
            success_counts = np.roll(success_counts, -1)
            success_counts[-1] = np.mean(rewards)
            
            # Move to next state
            state = next_state
        
        # Track average reward per episode
        avg_reward = np.mean(episode_rewards) / num_steps
        avg_rewards_history.append(avg_reward)
        
        # Calculate normalized rate (successful transmissions / total packets)
        if episode % 20 == 0:
            normalized_rate = np.mean(success_counts)
            normalized_rates.append(normalized_rate)
            
        # Decay exploration rate over time
        if episode % 100 == 0 and episode > 0:
            agent.exploration_rate = max(0.01, agent.exploration_rate * 0.9)
    
    return normalized_rates, avg_rewards_history, agent, env


def visualize_time_frequency(agent, env, state="initial"):
    """Visualize the time-frequency information as shown in the paper"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up plot parameters
    channels = range(env.num_channels)
    time_slots = range(10)  # Show 10 time slots
    
    # Plot grid
    for ch in channels:
        ax.axhline(y=ch, color='gray', linestyle='-', alpha=0.3)
    for t in time_slots:
        ax.axvline(x=t, color='gray', linestyle='-', alpha=0.3)
    
    if state == "initial":
        # Random actions for initial state
        for t in time_slots:
            # Jammer position (assuming sweep pattern)
            jammer_channel = t % env.num_channels
            ax.add_patch(Rectangle((t, jammer_channel), 0.9, 0.9, color='red', alpha=0.5))
            
            # User positions (random)
            for user_idx in range(env.num_users):
                user_channel = np.random.randint(0, env.num_channels)
                ax.add_patch(Rectangle((t, user_channel), 0.9, 0.9, color=f'C{user_idx}', alpha=0.5))
    else:  # Convergent state
        current_state = env.reset()
        
        for t in time_slots:
            # Get optimal action from trained agent
            joint_action = agent.get_optimal_joint_action(current_state)
            
            # Simulate environment step
            next_state, _ = env.step(joint_action)
            
            # Jammer position
            jammer_channel = env.jammer_channel
            ax.add_patch(Rectangle((t, jammer_channel), 0.9, 0.9, color='red', alpha=0.5, label='Jammer' if t == 0 else ""))
            
            # User positions
            for user_idx in range(env.num_users):
                user_channel = joint_action[user_idx]
                ax.add_patch(Rectangle((t, user_channel), 0.9, 0.9, color=f'C{user_idx}', 
                                        alpha=0.5, label=f'User {user_idx+1}' if t == 0 else ""))
            
            # Update state
            current_state = next_state
    
    # Set labels and title
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('Channel')
    ax.set_yticks(channels)
    ax.set_title(f'Time-Frequency Information at {state.capitalize()} State')
    if state == "convergent":
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.tight_layout()
    
    return fig


def visualize_channel_selection_heatmap(agent, env, num_steps=100):
    """Create a heatmap showing channel selection frequency"""
    # Reset environment
    env.reset()
    
    # Track channel selection frequency for each user
    selection_freq = np.zeros((env.num_users, env.num_channels))
    jammer_freq = np.zeros(env.num_channels)
    
    # Run simulation using trained policy
    state = env.reset()
    
    for _ in range(num_steps):
        # Select action using trained policy
        joint_action = agent.get_optimal_joint_action(state)
        
        # Track selections
        for user_idx, channel in enumerate(joint_action):
            selection_freq[user_idx, channel] += 1
        
        jammer_freq[env.jammer_channel] += 1
        
        # Step environment
        next_state, _ = env.step(joint_action)
        state = next_state
    
    # Normalize frequencies
    selection_freq = selection_freq / num_steps
    jammer_freq = jammer_freq / num_steps
    
    # Create figure with subplots
    fig, axs = plt.subplots(env.num_users + 1, 1, figsize=(10, 8), sharex=True)
    
    # Plot jammer frequency
    axs[0].bar(range(env.num_channels), jammer_freq, color='red', alpha=0.7)
    axs[0].set_title('Jammer Channel Selection Frequency')
    axs[0].set_ylabel('Frequency')
    axs[0].set_ylim(0, 1)
    
    # Plot each user's selection frequencies
    for user_idx in range(env.num_users):
        axs[user_idx+1].bar(range(env.num_channels), selection_freq[user_idx], 
                            color=f'C{user_idx}', alpha=0.7)
        axs[user_idx+1].set_title(f'User {user_idx+1} Channel Selection Frequency')
        axs[user_idx+1].set_ylabel('Frequency')
        axs[user_idx+1].set_ylim(0, 1)
    
    # Set common labels
    axs[-1].set_xlabel('Channel')
    plt.tight_layout()
    
    return fig


def visualize_q_values(agent, env):
    """Create a visualization of Q-values for each user"""
    # Select a specific state for visualization
    state = env.reset()
    
    # Extract Q-values for this state for all actions
    q_values = []
    for user_idx in range(env.num_users):
        user_q_values = np.zeros((env.num_channels, env.num_channels))
        for action in agent.joint_actions:
            q_value = agent.get_q_value(user_idx, state, action)
            user_q_values[action[0], action[1]] = q_value
        q_values.append(user_q_values)
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, env.num_users, figsize=(12, 5))
    
    # Plot Q-values as heatmaps
    for user_idx in range(env.num_users):
        im = axs[user_idx].imshow(q_values[user_idx], cmap='viridis')
        axs[user_idx].set_title(f'User {user_idx+1} Q-Values')
        axs[user_idx].set_xlabel('User 2 Channel')
        axs[user_idx].set_ylabel('User 1 Channel')
        plt.colorbar(im, ax=axs[user_idx])
    
    plt.tight_layout()
    return fig


def compare_methods(num_episodes=1000, num_steps=100):
    """Compare different methods as shown in the paper"""
    # Define parameters
    num_users = 2
    num_channels = 5
    
    # Initialize environment
    env = Environment(num_users=num_users, num_channels=num_channels)
    
    # For tracking results
    cmaa_rates = []
    indep_q_rates = []
    sensing_rates = []
    
    # 1. Run CMAA
    print("Running CMAA...")
    cmaa_agent = CollaborativeMultiAgentAntiJamming(
        num_users=num_users, 
        num_channels=num_channels,
        learning_rate=0.8,
        discount_factor=0.6,
        exploration_rate=0.2
    )
    
    # Initialize metrics
    window_size = 20
    success_counts = np.zeros(window_size)
    
    # Train CMAA
    for episode in tqdm(range(num_episodes), desc="CMAA Training"):
        state = env.reset()
        
        for step in range(num_steps):
            # Select action
            joint_action = cmaa_agent.select_action(state)
            
            # Take action
            next_state, rewards = env.step(joint_action)
            
            # Update Q-values
            for user_idx in range(num_users):
                cmaa_agent.update_q_value(user_idx, state, joint_action, rewards[user_idx], next_state)
            
            # Update success counts
            success_counts = np.roll(success_counts, -1)
            success_counts[-1] = np.mean(rewards)
            
            # Move to next state
            state = next_state
        
        # Record normalized rate
        if episode % 20 == 0:
            normalized_rate = np.mean(success_counts)
            cmaa_rates.append(normalized_rate)
        
        # Decay exploration
        if episode % 100 == 0 and episode > 0:
            cmaa_agent.exploration_rate = max(0.01, cmaa_agent.exploration_rate * 0.9)
    
    # 2. Run independent Q-learning (no coordination)
    print("Running Independent Q-learning...")
    # Simulate independent Q-learning with separate Q-tables
    q_tables = [{} for _ in range(num_users)]
    exploration_rate = 0.2
    
    # Reset metrics
    success_counts = np.zeros(window_size)
    
    for episode in tqdm(range(num_episodes), desc="Independent Q-learning Training"):
        state = env.reset()
        
        for step in range(num_steps):
            # Each user selects action independently
            actions = []
            for user_idx in range(num_users):
                if random.random() < exploration_rate:  # Exploration
                    actions.append(random.randint(0, num_channels - 1))
                else:  # Exploitation
                    best_channel = 0
                    best_value = float('-inf')
                    for channel in range(num_channels):
                        state_action = (state, channel)
                        if state_action not in q_tables[user_idx]:
                            q_tables[user_idx][state_action] = 0.0
                        if q_tables[user_idx][state_action] > best_value:
                            best_value = q_tables[user_idx][state_action]
                            best_channel = channel
                    actions.append(best_channel)
            
            # Take action
            next_state, rewards = env.step(actions)
            
            # Update Q-values independently
            for user_idx in range(num_users):
                state_action = (state, actions[user_idx])
                if state_action not in q_tables[user_idx]:
                    q_tables[user_idx][state_action] = 0.0
                
                # Calculate next best action value
                next_best_value = float('-inf')
                for channel in range(num_channels):
                    next_state_action = (next_state, channel)
                    if next_state_action not in q_tables[user_idx]:
                        q_tables[user_idx][next_state_action] = 0.0
                    if q_tables[user_idx][next_state_action] > next_best_value:
                        next_best_value = q_tables[user_idx][next_state_action]
                
                # Update Q-value
                q_tables[user_idx][state_action] = (1 - 0.8) * q_tables[user_idx][state_action] + \
                                                0.8 * (rewards[user_idx] + 0.6 * next_best_value)
            
            # Update success counts
            success_counts = np.roll(success_counts, -1)
            success_counts[-1] = np.mean(rewards)
            
            # Move to next state
            state = next_state
        
        # Record normalized rate
        if episode % 20 == 0:
            normalized_rate = np.mean(success_counts)
            indep_q_rates.append(normalized_rate)
        
        # Decay exploration
        if episode % 100 == 0 and episode > 0:
            exploration_rate = max(0.01, exploration_rate * 0.9)
    
    # 3. Run sensing-based method
    print("Running Sensing-based method...")
    # Reset metrics
    success_counts = np.zeros(window_size)
    
    for episode in tqdm(range(num_episodes), desc="Sensing-based Method"):
        state = env.reset()
        
        for step in range(num_steps):
            # Users select channels based on sensing results
            actions = []
            available_channels = list(range(num_channels))
            
            # Remove jammer's channel if detected
            if env.jammer_channel in available_channels:
                available_channels.remove(env.jammer_channel)
            
            # Users select channels in order
            for user_idx in range(num_users):
                if available_channels:
                    # Choose a random available channel
                    chosen_channel = random.choice(available_channels)
                    actions.append(chosen_channel)
                    available_channels.remove(chosen_channel)
                else:
                    # No available channels, pick a random one
                    actions.append(random.randint(0, num_channels - 1))
            
            # Take action
            next_state, rewards = env.step(actions)
            
            # Update success counts
            success_counts = np.roll(success_counts, -1)
            success_counts[-1] = np.mean(rewards)
            
            # Move to next state
            state = next_state
        
        # Record normalized rate
        if episode % 20 == 0:
            normalized_rate = np.mean(success_counts)
            sensing_rates.append(normalized_rate)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(0, num_episodes, 5)
    
    ax.plot(episodes[:len(cmaa_rates)], cmaa_rates, label='Collaborative Multi-agent (CMAA)', linewidth=2)
    ax.plot(episodes[:len(indep_q_rates)], indep_q_rates, label='Independent Q-learning', linewidth=2)
    ax.plot(episodes[:len(sensing_rates)], sensing_rates, label='Sensing-based Method', linewidth=2)
    
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Normalized Rate')
    ax.set_title('Performance Comparison of Anti-jamming Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation explaining normalized rate
    ax.text(0.02, 0.02, 
             'Normalized Rate = Successful Transmissions / Total Packets',
             transform=ax.transAxes, fontsize=9, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    return fig, cmaa_agent


def main():
    # Set up random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Increase training episodes
    num_episodes = 10000
    num_steps = 100
    
    # Run simulation
    print("Running CMAA simulation...")
    start_time = time.time()
    normalized_rates, avg_rewards, agent, env = run_simulation(num_episodes=num_episodes, num_steps=num_steps)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot learning curve
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(0, num_episodes, 10)
    ax.plot(episodes[:len(normalized_rates)], normalized_rates, linewidth=2)
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Normalized Rate')
    ax.set_title('CMAA Learning Curve')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cmaa_learning_curve.png')
    
    # Plot average rewards
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(num_episodes), avg_rewards, linewidth=2)
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Average Reward per Episode')
    ax.set_title('CMAA Average Reward')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cmaa_average_reward.png')
    
    # Visualize time-frequency information
    print("Generating time-frequency visualizations...")
    initial_fig = visualize_time_frequency(agent, env, state="initial")
    initial_fig.savefig('initial_state.png')
    
    convergent_fig = visualize_time_frequency(agent, env, state="convergent")
    convergent_fig.savefig('convergent_state.png')
    
    # Create additional visualizations
    print("Creating additional visualizations...")
    channel_heatmap = visualize_channel_selection_heatmap(agent, env, num_steps=500)
    channel_heatmap.savefig('channel_selection_heatmap.png')
    
    q_values_fig = visualize_q_values(agent, env)
    q_values_fig.savefig('q_values_visualization.png')
    
    # Compare different methods with fewer episodes for speed
    print("Comparing different methods...")
    comparison_fig, _ = compare_methods(num_episodes=10000, num_steps=100)
    comparison_fig.savefig('method_comparison.png')
    
    print("Simulation complete. Results saved as PNG files:")
    print("1. cmaa_learning_curve.png - Shows the learning curve of CMAA")
    print("2. cmaa_average_reward.png - Shows the average reward over episodes")
    print("3. initial_state.png - Shows the time-frequency information at initial state")
    print("4. convergent_state.png - Shows the time-frequency information at convergent state")
    print("5. channel_selection_heatmap.png - Shows the channel selection frequency")
    print("6. q_values_visualization.png - Shows the Q-values for each user")
    print("7. method_comparison.png - Compares CMAA with other methods")

if __name__ == "__main__":
    main()