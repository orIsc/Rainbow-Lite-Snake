"""CNN, factorized Gaussian NoisyNet, and dueling categorical (C51) heads."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """Learn mu and sigma; factorized noise stays fixed until reset_noise()."""
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
        # Factorized Gaussian noise: f(x)=sign(x)*sqrt(abs(x)); one vector
        # per input/output rather than an independent draw for every weight.
        def scaled_noise(count):
            x = torch.randn(count, device=self.weight_mu.device, dtype=self.weight_mu.dtype)
            return x.sign() * x.abs().sqrt()
        incoming = scaled_noise(self.weight_mu.shape[1])
        outgoing = scaled_noise(self.weight_mu.shape[0])
        self.weight_epsilon.copy_(outgoing.outer(incoming))
        self.bias_epsilon.copy_(outgoing)

    def forward(self, x):
        if self.training:
            # Sigma is learned by backprop alongside mu; epsilon is a buffer.
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Deterministic evaluation uses the learned means only.
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    def __init__(self, size=20, action_size=4, hidden=128,
                 atoms=51, v_min=-10.0, v_max=100.0):
        super().__init__()
        if atoms < 2 or v_min >= v_max:
            raise ValueError("Invalid categorical support")
        self.action_size, self.atoms = action_size, atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            width = self.features(torch.zeros(1, 3, size, size)).shape[1]
        self.value = nn.Sequential(NoisyLinear(width, hidden), nn.ReLU(),
                                   NoisyLinear(hidden, atoms))
        self.advantage = nn.Sequential(NoisyLinear(width, hidden), nn.ReLU(),
                                       NoisyLinear(hidden, action_size * atoms))

    def reset_noise(self):
        for layer in self.modules():
            
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def logits(self, states):
        features = self.features(states.float())
        value = self.value(features).view(-1, 1, self.atoms)
        advantage = self.advantage(features).view(-1, self.action_size, self.atoms)
        # Center across actions independently for every atom.
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def forward(self, states):
        return self.logits(states).softmax(dim=-1)

    def q_values(self, states):
        return (self(states) * self.support).sum(dim=-1)
