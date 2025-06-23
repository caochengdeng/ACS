import copy
import torch
import numpy as np

class SafetyBuffer:
    def __init__(self):
        self.safety_list = []
        self.safety_list_spare = []

    def clear(self):
        print("Safety Buffer length:{}".format(len(self.safety_list)))
        del self.safety_list[:]
        self.safety_list = copy.deepcopy(self.safety_list_spare)
        del self.safety_list_spare[:]

    def quantile(self, Q):
        S_safety = torch.tensor(self.safety_list)
        return torch.quantile(S_safety, Q / 10)


class RolloutBuffer:
    """
    Buffer for storing episode data
    """

    def __init__(self):
        self.actions = []  # Actions
        self.states = []  # States
        self.next_states = []
        self.rewards = []  # Rewards
        self.is_terminals = []  # Terminal
        self.rewards_safety = []  # Collision penalties
        self.next_actions_safety = []
        self.actions_safety = []

    def clear(self):
        """
        Clear the buffer
        """
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.rewards_safety[:]
        del self.next_actions_safety[:]
        del self.actions_safety[:]
