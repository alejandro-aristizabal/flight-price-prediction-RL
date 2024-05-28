import numpy as np
import pandas as pd


class FlightEnv:
    def __init__(self, data):
        self.data = data.reset_index(drop=True)  # Ensure index is reset
        self.num_states = len(data)
        self.num_actions = 2  # Buy or wait
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self.data.iloc[self.current_index].values

    def step(self, action):
        current_state = self.data.iloc[self.current_index].values
        self.current_index += 1
        if self.current_index >= self.num_states:
            done = True
            next_state = current_state  # Use current state as next state if done
        else:
            done = False
            next_state = self.data.iloc[self.current_index].values

        reward = self.calculate_reward(current_state, action)

        return next_state, reward, done

    def calculate_reward(self, state, action):
        current_price = state[1]  # baseFare
        if action == 0:  # Wait
            return -1  # Small penalty for waiting
        elif action == 1:  # Buy
            if self.current_index < self.num_states - 1:
                future_price = self.data.iloc[self.current_index + 1, 1]  # baseFare
                return max(0, current_price - future_price)  # Reward for saving money
            else:
                return 0  # No future price available
