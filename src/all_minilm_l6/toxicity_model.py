import torch.nn as nn
import torch.nn.functional as F

class ToxicityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(384, 128)
        self.norm0 = nn.BatchNorm1d(128)
        self.linear1 = nn.Linear(128, 32)
        self.norm1 = nn.BatchNorm1d(32)
        self.linear_out = nn.Linear(32, 6)

    def forward(self, x):
        x = self.linear0(x)
        x = self.norm0(x)
        x = F.relu(x)

        x = self.linear1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.linear_out(x)

        return x
