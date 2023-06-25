import torch.nn as nn
import torch.nn.functional as F


class LinearActor(nn.Module):

    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(LinearActor, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class LinearCritic(nn.Module):

    def __init__(self, obs_dim, hidden_dim):
        super(LinearCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)


class ConvActor(nn.Module):
    def __init__(self, in_channels, in_h, in_w, hidden_channels, action_dim):
        super(ConvActor, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(hidden_channels * in_h * in_w, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class ConvCritic(nn.Module):
    def __init__(self, in_channels, in_h, in_w, hidden_channels):
        super(ConvCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(hidden_channels * in_h * in_w, 1)
        )

    def forward(self, x):
        return self.model(x)


class ActorCritic(nn.Module):

    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.model(x)
        return self.actor(h), self.critic(h)


class A3CNet(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(A3CNet, self).__init__()
        self.critic_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )
        self.actor_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, action_dim)
        )

    def forward(self, x):
        action = F.softmax(self.actor_model(x), dim=-1)
        value = self.critic_model(x)
        return action, value