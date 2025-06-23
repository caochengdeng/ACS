import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HighwayEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        # Delayed imports to prevent circular references
        from .get_traci_data_modularized import GetTraciData_im

        # Environment configuration
        self.action_space = spaces.MultiDiscrete([3, 7])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 18))
        self.gtd = GetTraciData_im()
        self.MaxSpeed = 120  # km/h
        self.preheat = 20
        self.simulation_time = 60
        self.run_step = 0
        self.env_type = 0

        # Lazy load utility modules
        from ACS.envs.utils.highway_utils import HighwayUtils
        self._utils = HighwayUtils(max_speed_kmh=self.MaxSpeed)

    def step(self, action):
        """
        Execute one step in the environment.
        """
        self.run_step += 1

        # Process action (simple inline processing)
        try:
            action[1] = action[1] - 3
            action[0] = action[0] - 1
        except Exception:
            action = [None, action]

        # Step simulation
        self.gtd.StepSimulation(action)

        # Get new state
        state = self.gtd.GetState()

        # Calculate reward
        reward, reward_safety, done_collision, done_step = self._calculate_reward(state)

        # Get rule-based action
        action_next, velocity, acceleration, road_id, distance = self.gtd.RuleModel()

        # Determine termination condition
        if self.env_type == 3:
            done = True if distance + 15 > 10000000 else done_collision
        else:
            done = done_collision or done_step

        # Reset step counter if episode is done
        if done:
            self.run_step = 0
            self.close()

        # Build info dictionary
        info = {
            'collision': int(done_collision),
            'velocity': velocity,
            'acceleration': acceleration,
            'road_id': road_id,
            'distance': distance,
            'steps': done_step
        }

        return state, reward, reward_safety, done, False, info, action_next

    def _calculate_reward(self, state):
        """
        Internal method to calculate reward using utility class.
        """
        return self.gtd.Calculate_Reward(state)

    def reset(self, seed=24, Ifupdata=False, Up_seed=1, env_type=0, lanechange_model_off=True):
        """
        Reset the environment to an initial state.
        """
        super().reset(seed=seed)
        self.env_type = env_type

        # Adjust settings based on environment type
        if env_type == 1 or env_type == 2:
            self.preheat = 9
            self.simulation_time = 20
        elif env_type == 3:
            self.preheat = 300.0
            self.simulation_time = 9999999

        # Start simulation
        self.gtd.StartSimulation(Ifupdata=Ifupdata, seed=Up_seed, env_type=env_type)
        self.gtd.Preheat(self.preheat, lanechange_model_off=lanechange_model_off)

        # Get initial state
        state = self.gtd.GetState()
        return state

    def render(self):
        """Render the environment."""
        self.gtd.render_cmd = "sumo"

    def close(self):
        """Close the simulation."""
        self.gtd.CloseSimulation()