"""Factorized Gaussian Noisy Nets and mean-centered dueling Q values."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma_init / math.sqrt(in_features))
        nn.init.constant_(self.bias_sigma, sigma_init / math.sqrt(out_features))
        self.reset_noise()

    @torch.no_grad()
    def reset_noise(self):
        def scaled_noise(count):
            x = torch.randn(count, device=self.weight_mu.device, dtype=self.weight_mu.dtype)
            return x.sign() * x.abs().sqrt()
        incoming = scaled_noise(self.weight_mu.shape[1])
        outgoing = scaled_noise(self.weight_mu.shape[0])
        self.weight_epsilon.copy_(outgoing.outer(incoming))
        self.bias_epsilon.copy_(outgoing)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    def __init__(self, observation_size, action_size=3, hidden=128):
        super().__init__()
        self.features = nn.Sequential(NoisyLinear(observation_size, hidden), nn.ReLU())
        self.value = nn.Sequential(NoisyLinear(hidden, hidden), nn.ReLU(), NoisyLinear(hidden, 1))
        self.advantage = nn.Sequential(NoisyLinear(hidden, hidden), nn.ReLU(), NoisyLinear(hidden, action_size))

    def reset_noise(self):
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, states):
        features = self.features(states)
        value, advantage = self.value(features), self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
