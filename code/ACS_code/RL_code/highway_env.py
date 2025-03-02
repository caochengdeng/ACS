import math
import gymnasium as gym
import traci
from gymnasium import spaces
import numpy as np
import GetTraciData as GetTraciData_im


class HighwayEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.action_space = spaces.MultiDiscrete([3, 7])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 18))
        self.gtd = GetTraciData_im.GetTraciData()
        self.MaxSpeed = 120  # km/h
        self.preheat = 20
        self.simulation_time = 60
        self.run_step = 0

    def step(self, action):
        self.run_step += 1

        action[1] = action[1] - 3
        action[0] = action[0] - 1

        self.gtd.StepSimulation(action)

        state = self.gtd.GetState()

        reward, reward_safety, done_collision, _ = self.Calculate_Reward(state)

        action_next, velocity, acceleration = self.gtd.RuleModel()

        done_step = False
        if self.run_step == self.simulation_time * 10:
            done_step = True

        # Environment termination conditions: 1: Collision, 2: Simulation end
        done = done_collision or done_step
        # print(done_collision, done_step, done)

        if done:
            self.run_step = 0
            self.close()
        if done:
            info = {'collision': int(done_collision), 'steps': done_step, 'velocity': velocity, 'acceleration': acceleration}  # 此处可以存储碰撞次数等信息
        else:
            info = {'velocity': velocity, 'acceleration': acceleration}

        return state, reward, reward_safety, done, False, info, action_next

    def reset(self, seed=24, If_update=False, Up_seed=1):
        # seed
        super().reset(seed=seed)

        self.gtd.StartSimulation(If_update=If_update, seed=Up_seed)
        self.gtd.Preheat(self.preheat)

        state = self.gtd.GetState()
        return state

    def render(self):
        self.gtd.render_cmd = "sumo-gui"

    def close(self):
        self.gtd.CloseSimulation()

    def Calculate_Reward(self, state):
        reward_speed = (0 if (state[0] * 3.6 < self.MaxSpeed - 40) else (state[0] * 3.6 - self.MaxSpeed + 40) / 40) \
            if state[0] * 3.6 <= self.MaxSpeed else 0

        reward_safety = 0

        done_collision = self.gtd.CollisionDetection()
        if done_collision:
            reward_safety = -50

        done_step = self.gtd.GetTimeDone(self.simulation_time)
        reward = reward_speed + reward_safety

        return reward, reward_safety, done_collision, done_step
