import numpy as np


class HighwayUtils:

    def __init__(self, max_speed_kmh=120):
        self.max_speed = max_speed_kmh  # km/h

    def calculate_reward(self, state, collision_detector):

        if len(state) == 0:
            return 0, 0, False, False

        speed_kmh = state[0] * 3.6  # Convert m/s to km/h

        # Speed reward
        if speed_kmh < self.max_speed - 40:
            reward_speed = 0
        elif speed_kmh <= self.max_speed:
            reward_speed = (speed_kmh - self.max_speed + 40) / 40
        else:
            reward_speed = 0

        # Safety penalty
        done_collision = collision_detector.check_collision()
        reward_safety = -50 if done_collision else 0

        # Time completion check
        done_step = collision_detector.check_time_done()

        return reward_speed + reward_safety, reward_safety, done_collision, done_step

    def normalize_state(self, state):
        return state