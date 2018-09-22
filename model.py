import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    """
    Basic actor feed forward network policy model
    """

    def __init__(self, state_size, num_actions, h1_size=20, h2_size=15, seed=888):
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


class Dueling_DQN(nn.Module):
    def __init__(self, state_size, num_actions, h1_size=10, h2_size=10):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.h1 = nn.Linear(state_size, h1_size)
        self.h2 = nn.Linear(h1_size, h2_size)

        self.adv = nn.Linear(in_features=h2_size, out_features=h2_size)
        self.val = nn.Linear(in_features=h2_size, out_features=h2_size)

        self.adv2 = nn.Linear(in_features=h2_size, out_features=num_actions)
        self.val2 = nn.Linear(in_features=h2_size, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.h1(x))
        x = self.relu(self.h2(x))

        adv = self.relu(self.adv(x))
        val = self.relu(self.val(x))

        adv = self.adv2(adv)
        val = self.val2(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x
