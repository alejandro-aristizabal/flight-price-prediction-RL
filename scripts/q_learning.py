import numpy as np
import pandas as pd
from environment import FlightEnv
import argparse
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(
        self,
        num_states,
        num_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        epsilon_decay=0.99,
        min_epsilon=0.01,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Q-Learning agent for flight prices."
    )
    parser.add_argument(
        "--data_path", type=str, help="Path to the preprocessed data CSV file"
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument(
        "--epsilon_decay",
        type=float,
        default=0.99,
        help="Decay rate for exploration rate",
    )
    parser.add_argument(
        "--min_epsilon", type=float, default=0.01, help="Minimum exploration rate"
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum steps per episode"
    )
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    env = FlightEnv(data)
    agent = QLearningAgent(
        env.num_states,
        env.num_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
    )

    rewards = []
    for episode in range(args.episodes):
        state = env.reset()
        state_index = 0
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < args.max_steps:
            action = agent.choose_action(state_index)
            next_state, reward, done = env.step(action)
            next_state_index = state_index + 1
            agent.update_q_table(state_index, action, reward, next_state_index)
            state_index = next_state_index
            total_reward += reward
            steps += 1
        agent.decay_epsilon()
        rewards.append(total_reward)
        print(
            f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}"
        )

    # Get data folder from the file path
    data_folder = args.data_path.split("/")[-2]

    # Save the Q-table and rewards
    np.save(f"models/{data_folder}_q_table.npy", agent.q_table)
    np.save(f"models/{data_folder}_rewards.npy", rewards)

    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward vs Episodes")
    plt.grid(True)
    plt.show()
