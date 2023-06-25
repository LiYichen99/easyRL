import torch
import torch.nn as nn


class LinearQModel(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, dueling=False):
        super(LinearQModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.model_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.model_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.model_q = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.dueling = dueling

    def forward(self, x):
        if self.dueling:
            h = self.model(x)
            v = self.model_v(h)
            a = self.model_a(h)
            return v + a - torch.mean(a, keepdim=True, dim=-1)
        else:
            return self.model_q(self.model(x))


class ConvQModel(nn.Module):

    def __init__(self, in_channels, in_h, in_w, hidden_channels, output_dim, dueling=False):
        super(ConvQModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.model_v = nn.Sequential(
            nn.Linear(hidden_channels * in_h * in_w, 1)
        )
        self.model_a_or_q = nn.Sequential(
            nn.Linear(hidden_channels * in_h * in_w, output_dim)
        )
        self.dueling = dueling

    def forward(self, x):
        if self.dueling:
            h = self.model(x)
            v = self.model_v(h)
            a = self.model_a_or_q(h)
            return v + a - torch.mean(a, keepdim=True, dim=-1)
        else:
            return self.model_a_or_q(self.model(x))
