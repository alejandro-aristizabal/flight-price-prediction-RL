import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_rewards(rewards_path):
    rewards = np.load(rewards_path)
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward vs Episodes")
    plt.grid(True)
    plt.show()

    # Save the plot
    plot_path = rewards_path.replace(".npy", ".png")
    plt.savefig(plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the rewards over episodes.")
    parser.add_argument("rewards_path", type=str, help="Path to the rewards numpy file")
    args = parser.parse_args()

    plot_rewards(args.rewards_path)
