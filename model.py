import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    """
    Basic actor feed forward network policy model
    """

    def __init__(self, state_size, num_actions, h1_size=10, h2_size=10, seed=999):
        """

        :param state_size:
        :param num_actions:
        """

        super(QNet, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.h1 = nn.Linear(state_size, h1_size)
        self.h2 = nn.Linear(h1_size, h2_size)
        self.out = nn.Linear(h2_size, num_actions)

    def forward(self, state):
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        return self.out(x)
