import numpy as np


class MetricTracker:
    def __init__(self):
        self.return_list = []
        self.steps_list = []
        self.velocity_episode = []
        self.velocity_step = []
        self.acceleration_episode = []
        self.acceleration_step = []
        self.reward_update = []

    def add_return(self, episode_return):
        """episode reward"""
        self.return_list.append(episode_return)
        self.reward_update.append(episode_return)

    def add_steps(self, step):
        """episode step"""
        self.steps_list.append(step)

    def add_velocity(self, velocity, is_episode=True):
        """add velocity"""
        if is_episode:
            self.velocity_episode.append(velocity)
        else:
            self.velocity_step.append(velocity)

    def add_acceleration(self, acceleration, is_episode=True):
        if is_episode:
            self.acceleration_episode.append(acceleration)
        else:
            self.acceleration_step.append(acceleration)

    def reset_velocity_and_acceleration(self):
        """reset"""
        self.velocity_episode = []
        self.velocity_step = []
        self.acceleration_episode = []
        self.acceleration_step = []

    def get_recent_average_reward(self, window_size=10):
        """window_size episode reward"""
        if len(self.return_list) == 0:
            return 0
        recent_returns = self.return_list[-window_size:]
        return np.mean(recent_returns)

    def get_success_rate(self, max_simulation_step):
        """rate"""
        if len(self.steps_list) == 0:
            return 0
        return np.mean([1 if i >= max_simulation_step else 0 for i in self.steps_list])

    def clear(self):
        """clear buffer"""
        self.steps_list = []
        self.reward_update = []
        self.velocity_step = []
        self.acceleration_step = []