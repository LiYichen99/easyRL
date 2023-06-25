import torch.nn as nn


class LinearPolicyModel(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(LinearPolicyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class ConvPolicyModel(nn.Module):

    def __init__(self, in_channels, in_h, in_w, hidden_channels, output_dim):
        super(ConvPolicyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_channels * in_h * in_w, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)
