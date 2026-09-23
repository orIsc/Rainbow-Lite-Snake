"""Three-step aggregation followed by proportional prioritized replay."""
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool  # True termination only; time limits still bootstrap.
    discount: float  # gamma ** actual number of aggregated steps


class NStepAccumulator:
    def __init__(self, gamma=0.99, n=3):
        if n < 1 or not 0 <= gamma <= 1:
            raise ValueError("Invalid n-step parameters")
        self.gamma = gamma
        self.n = n
        self.waiting_room = deque()

    def _finalize(self):
        reward, discount = 0.0, 1.0
        for _, _, r, next_state, done in self.waiting_room:
            reward += discount * r
            discount *= self.gamma
            if done:
                break
        state, action = self.waiting_room[0][:2]
        return Transition(state.copy(), action, reward, next_state.copy(), done, discount)

    def append(self, state, action, reward, next_state, done, truncated=False):
        self.waiting_room.append((state.copy(), action, reward, next_state.copy(), done))
        finalized = []
        if done or truncated:
            # Flush all suffixes on either boundary. Truncations retain done=False.
            while self.waiting_room:
                finalized.append(self._finalize())
                self.waiting_room.popleft()
        elif len(self.waiting_room) == self.n:
            finalized.append(self._finalize())
            self.waiting_room.popleft()
        return finalized


class PrioritizedReplayBuffer:
    def __init__(self, capacity=50_000, alpha=0.6, seed=0, epsilon=1e-6):
        if capacity <= 0 or not 0 <= alpha <= 1 or epsilon <= 0:
            raise ValueError("Invalid replay parameters")
        self.capacity, self.alpha, self.epsilon = capacity, alpha, epsilon
        self.data = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def state_dict(self):
        # Only tensors and primitive containers, compatible with weights_only=True.
        return {"capacity": self.capacity, "alpha": self.alpha, "epsilon": self.epsilon,
                "position": self.position, "rng": self.rng.bit_generator.state,
                "priorities": torch.from_numpy(self.priorities.copy()),
                "data": [(torch.from_numpy(t.state.copy()), t.action, t.reward,
                          torch.from_numpy(t.next_state.copy()), t.done, t.discount)
                         for t in self.data]}

    def load_state_dict(self, state):
        self.capacity, self.alpha, self.epsilon = state["capacity"], state["alpha"], state["epsilon"]
        self.position = state["position"]
        self.priorities = state["priorities"].cpu().numpy().copy()
        self.data = [Transition(s.cpu().numpy().copy(), a, r, ns.cpu().numpy().copy(), d, g)
                     for s, a, r, ns, d, g in state["data"]]
        self.rng.bit_generator.state = state["rng"]

    def add(self, transition):
        priority = self.priorities[:len(self)].max() if self.data else 1.0
        if len(self) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.position] = transition
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta):
        if not self.data or batch_size <= 0 or not 0 <= beta <= 1:
            raise ValueError("Invalid batch size, beta, or empty replay")
        probabilities = self.priorities[:len(self)] ** self.alpha
        probabilities /= probabilities.sum()
        indices = self.rng.choice(len(self), batch_size, replace=True, p=probabilities)
        # Normalize by the maximum weight over the entire replay distribution.
        weights = (len(self) * probabilities[indices]) ** (-beta)
        weights /= (len(self) * probabilities.min()) ** (-beta)
        return [self.data[i] for i in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        # Sampling with replacement can duplicate indices; keep maximum surprise.
        updates = {}
        for index, error in zip(indices, td_errors):
            priority = abs(float(error)) + self.epsilon
            if not np.isfinite(priority):
                raise ValueError("Non-finite TD error")
            updates[index] = max(updates.get(index, 0), priority)
        for index, priority in updates.items():
            self.priorities[index] = priority
