import numpy as np
import torch
import os
from ACS_Agent import PolicyNet, ValueNet, SafetyNet


class PPO:
    def __init__(self, n_states, n_actions, device, crt):
        self.actor = PolicyNet(n_states, n_actions).to(device)
        self.critic = ValueNet(n_states).to(device)
        self.safety = SafetyNet(n_states).to(device)

        self.device = device
        self.s_criteria = crt

    # 动作选择
    def take_action(self, state, action_rule):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        probs = self.actor(state)

        actions = []

        for i in range(len(probs)):
            action_list = torch.distributions.Categorical(probs[i])
            action = action_list.sample().item()
            actions.append(action)

        safety_value_rule = self.safety(
            torch.tensor(np.array(action_rule), dtype=torch.float).unsqueeze(0).to(self.device), state)
        safety_value_actor = self.safety(
            torch.tensor(np.array(actions), dtype=torch.float).unsqueeze(0).to(self.device), state)

        action_make = actions if (safety_value_actor > self.s_criteria) else \
            (action_rule if safety_value_rule > safety_value_actor else actions)
        # if safety_value_actor <= self.s_criteria:
        #     if safety_value_rule > safety_value_actor:
        #         print('use actor_rule')
        # action_make = actions
        return action_make, actions

    def load_model(self, path, index):
        path_actor = os.path.join(path, 'actor_model_{}'.format(index))
        path_safety = os.path.join(path, 'safety_model_{}'.format(index))

        self.actor.load_state_dict(torch.load(path_actor, map_location=lambda storage, loc: storage))
        self.safety.load_state_dict(torch.load(path_safety, map_location=lambda storage, loc: storage))
