import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run the flight price Q-Learning pipeline."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step to execute (1: Preprocess, 2: Train, 3: Evaluate, 4: Visualize)",
    )
    parser.add_argument("--file_path", type=str, help="Path to the raw data CSV file")
    parser.add_argument(
        "--output_path", type=str, help="Path to save the preprocessed data CSV file"
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument("--q_table_path", type=str, help="Path to save the Q-table")
    parser.add_argument(
        "--rewards_path", type=str, help="Path to save the rewards numpy file"
    )
    parser.add_argument(
        "--epsilon_decay",
        type=float,
        default=0.99,
        help="Decay rate for exploration rate",
    )
    parser.add_argument(
        "--min_epsilon", type=float, default=0.01, help="Minimum exploration rate"
    )

    args = parser.parse_args()

    print(args.step)

    if args.step == 1:
        print("Step 1: Preprocessing data...")
        subprocess.run(
            [
                "python",
                "scripts/preprocess.py",
                "--file_path",
                args.file_path,
                "--output_path",
                args.output_path,
            ]
        )

    if args.step == 2:
        print("Step 2: Training Q-Learning agent...")
        subprocess.run(
            [
                "python",
                "scripts/q_learning.py",
                "--data_path",
                args.file_path,
                "--alpha",
                str(args.alpha),
                "--gamma",
                str(args.gamma),
                "--epsilon",
                str(args.epsilon),
                "--episodes",
                str(args.episodes),
            ]
        )

    if args.step == 3:
        print("Step 3: Evaluating policy...")
        subprocess.run(
            [
                "python",
                "scripts/evaluation.py",
                "--data_path",
                args.file_path,
                "--q_table_path",
                args.q_table_path,
            ]
        )

    if args.step == 4:
        print("Step 4: Visualizing rewards...")
        subprocess.run(["python", "scripts/visualize.py", args.rewards_path])


if __name__ == "__main__":
    main()
