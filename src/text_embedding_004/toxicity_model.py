import torch.nn as nn
import torch.nn.functional as F

class ToxicityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(768, 512)
        self.norm0 = nn.BatchNorm1d(512)
        self.linear1 = nn.Linear(512, 128)
        self.norm1 = nn.BatchNorm1d(128)
        self.linear_out = nn.Linear(128, 6)

    def forward(self, x):
        x = self.linear0(x)
        x = self.norm0(x)
        x = F.relu(x)

        x = self.linear1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.linear_out(x)

        return x
