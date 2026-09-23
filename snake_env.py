"""Gymnasium Snake with binary food, body, and wall planes."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 10}
    DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))
    OPPOSITE = (1, 0, 3, 2)

    def __init__(self, size=10, max_steps=2000, render_mode=None):
        super().__init__()
        if size < 6 or (max_steps is not None and max_steps <= 0):
            raise ValueError("size must be >= 6 and max_steps positive or None")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("Unsupported render mode")
        self.size, self.max_steps, self.render_mode = size, max_steps, render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, (3, size, size), dtype=np.uint8)
        # Border cells are walls: 10x10 by default, with an 8x8 playable interior.
        self.walls = np.zeros((size, size), dtype=np.uint8)
        self.walls[[0, -1], :] = 1
        self.walls[:, [0, -1]] = 1
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        x = y = self.size // 2
        self.body = [(x, y), (x - 1, y), (x - 2, y)]
        self.direction = 3
        self.steps = self.score = 0
        self.done = False
        self._spawn_food()
        if self.render_mode == "human":
            self.render()
        return self._observe(), {"score": self.score}

    def _spawn_food(self):
        occupied = set(self.body)
        free = [(x, y) for y in range(1, self.size - 1)
                for x in range(1, self.size - 1) if (x, y) not in occupied]
        self.food = free[int(self.np_random.integers(len(free)))] if free else None

    def _observe(self):
        grid = np.zeros(self.observation_space.shape, dtype=np.uint8)
        if self.food is not None:
            x, y = self.food
            grid[0, y, x] = 1
        for x, y in self.body:
            grid[1, y, x] = 1
        grid[2] = self.walls
        return grid

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode")
        if not self.action_space.contains(action):
            raise ValueError("action must be 0=up, 1=down, 2=left, or 3=right")
        # Reverse requests continue the current heading.
        if action != self.OPPOSITE[self.direction]:
            self.direction = int(action)
        dx, dy = self.DIRECTIONS[self.direction]
        x, y = self.body[0]
        head = (x + dx, y + dy)
        eating = head == self.food
        occupied = self.body if eating else self.body[:-1]
        x, y = head
        collision = not (0 <= x < self.size and 0 <= y < self.size)
        collision = collision or bool(self.walls[y, x]) or head in occupied
        self.steps += 1
        terminated, reward = collision, -10.0 if collision else -0.01
        if not collision:
            self.body.insert(0, head)
            if eating:
                self.score += 1
                reward = 10.0
                self._spawn_food()
                terminated = self.food is None
            else:
                self.body.pop()
        truncated = bool(not terminated and self.max_steps is not None
                         and self.steps >= self.max_steps)
        self.done = terminated or truncated
        if self.render_mode == "human":
            self.render()
        return self._observe(), reward, terminated, truncated, {"score": self.score}

    def render(self):
        grid = np.where(self.walls, "#", ".")
        for x, y in self.body:
            grid[y, x] = "o"
        x, y = self.body[0]
        grid[y, x] = "H"
        if self.food is not None:
            x, y = self.food
            grid[y, x] = "*"
        frame = "\n".join(" ".join(row) for row in grid)
        if self.render_mode == "human":
            print(frame)
        elif self.render_mode == "ansi":
            return frame
