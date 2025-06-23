import torch
import torch.nn as nn
import torch.nn.functional as F
from ACS.common.initialization import orthogonal_init

# ----------------------------------- #
# Build policy network -- actor
# ----------------------------------- #
class PolicyNet(nn.Module):
    def __init__(self, n_states, nvecs_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.ModuleList([nn.Linear(64, n) for n in nvecs_actions])

        # self._initialize_weights()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        for layer in self.fc3:
            orthogonal_init(layer, gain=0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x1 = F.leaky_relu(x)
        x = [F.softmax(layer(x1), dim=-1) for layer in self.fc3]  # Calculate probability of each action
        return x


# ----------------------------------- #
# Build value network -- critic
# ----------------------------------- #

class ValueNet(nn.Module):
    def __init__(self, n_states):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)
        # self._initialize_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)  # Evaluate current state value
        x = F.leaky_relu(x)
        x = self.fc3(x)
        return x