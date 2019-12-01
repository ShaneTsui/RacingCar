from torch import nn as nn
from torch.nn import functional as F


class ActionNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the network
        param: state_dim : 2N = N + N (Xs + Ys)
        param: action_dim: 3, steer, accel, break
        """
        super(ActionNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, action_dim)

    def forward(self, state):
        """
        Define the forward pass
        param: state: The state of the environment
        """
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return x

    def is_converged(self):
        raise NotImplementedError()