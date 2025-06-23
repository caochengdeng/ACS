import torch
import torch.nn as nn
import torch.nn.functional as F

class SafetyNet(nn.Module):
    def __init__(self, n_states):
        super(SafetyNet, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 8)

        self.f1 = nn.Linear(n_states, 64)
        self.f2 = nn.Linear(64, 32)
        self.f3 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        x1 = x1.to(dtype=torch.float)
        x1 = self.fc1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.fc2(x1)
        x1 = F.leaky_relu(x1)

        x2 = self.f1(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.f2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.f3(x2)
        x2 = F.leaky_relu(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = torch.clamp(-self.fc4(x), max=0)
        return x