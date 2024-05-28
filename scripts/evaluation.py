import numpy as np
import pandas as pd
from environment import FlightEnv
import argparse


def evaluate_policy(q_table, data):
    env = FlightEnv(data)
    state = env.reset()
    state_index = 0
    total_reward = 0
    done = False
    print(f"total states: {env.num_states}")
    while not done:
        action = np.argmax(q_table[state_index])
        next_state, reward, done = env.step(action)
        state_index += 1
        total_reward += reward
        if action == 1 | state_index % 1000 == 0:
            print(
                f"Step {state_index}: Action {action}, Reward {reward}, Total Reward {total_reward}"
            )
    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Q-Learning policy.")
    parser.add_argument(
        "--data_path", type=str, help="Path to the preprocessed data CSV file"
    )
    parser.add_argument("--q_table_path", type=str, help="Path to the saved Q-table")
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    q_table = np.load(args.q_table_path)
    total_reward = evaluate_policy(q_table, data)
    print("Total Reward:", total_reward)
