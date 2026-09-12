"""Grid Snake with relative actions and a fully observable ordered body."""
import numpy as np


class SnakeEnv:
    # Clockwise directions: up, right, down, left.
    DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
    action_size = 3  # straight, turn left, turn right

    def __init__(self, size=8, seed=0, starvation_steps=None, features=True):
        if size < 4:
            raise ValueError("size must be at least 4")
        self.size = size
        self.features = features
        self.limit = starvation_steps or 4 * size * size
        self.rng = np.random.default_rng(seed)
        # Body order, head, food, heading, remaining starvation budget.
        self.observation_size = 3 * size * size + 5 + (7 if features else 0)
        self.reset()

    def reset(self):
        x, y = self.size // 2, self.size // 2
        self.body = [(x, y), (x - 1, y), (x - 2, y)]
        self.direction = 1
        self.idle = self.score = 0
        self.done = False
        self._spawn_food()
        return self._observe()

    def _spawn_food(self):
        occupied = set(self.body)
        free = [(x, y) for y in range(self.size) for x in range(self.size)
                if (x, y) not in occupied]
        self.food = free[int(self.rng.integers(len(free)))] if free else None

    def _observe(self):
        grid = np.zeros((3, self.size, self.size), dtype=np.float32)
        for i, (x, y) in enumerate(self.body):
            grid[0, y, x] = (len(self.body) - i) / (self.size ** 2)
        x, y = self.body[0]
        grid[1, y, x] = 1
        if self.food is not None:
            x, y = self.food
            grid[2, y, x] = 1
        heading = np.eye(4, dtype=np.float32)[self.direction]
        budget = np.array([max(0, 1 - self.idle / self.limit)], dtype=np.float32)
        observation = np.concatenate((grid.ravel(), heading, budget))
        if not self.features:
            return observation
        # Redundant relative features let the MLP generalize across grid positions.
        # They describe the state only: no actions are masked or selected for it.
        x, y = self.body[0]
        danger = []
        for turn in (0, -1, 1):
            dx, dy = self.DIRECTIONS[(self.direction + turn) % 4]
            head = (x + dx, y + dy)
            occupied = self.body if head == self.food else self.body[:-1]
            danger.append(float(not (0 <= head[0] < self.size and 0 <= head[1] < self.size)
                                or head in occupied))
        fx, fy = self.food if self.food is not None else (x, y)
        forward = self.DIRECTIONS[self.direction]
        right = self.DIRECTIONS[(self.direction + 1) % 4]
        relative = np.array([(fx - x) * forward[0] + (fy - y) * forward[1],
                             (fx - x) * right[0] + (fy - y) * right[1]], dtype=np.float32)
        features = np.concatenate((np.asarray(danger, dtype=np.float32),
                                   relative / (self.size - 1), np.sign(relative)))
        return np.concatenate((observation, features))

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() after an episode ends")
        if action not in (0, 1, 2):
            raise ValueError("action must be 0, 1, or 2")
        self.direction = (self.direction + (0, -1, 1)[action]) % 4
        dx, dy = self.DIRECTIONS[self.direction]
        x, y = self.body[0]
        head = (x + dx, y + dy)
        eating = head == self.food
        # Moving into the current tail is legal when it moves away this turn.
        occupied = self.body if eating else self.body[:-1]
        if not (0 <= head[0] < self.size and 0 <= head[1] < self.size) or head in occupied:
            self.done = True
            return self._observe(), -1.0, True, {"score": self.score}
        self.body.insert(0, head)
        self.idle += 1
        reward = -0.01
        if eating:
            self.score += 1
            self.idle = 0
            reward = 1.0
            self._spawn_food()
            self.done = self.food is None
        else:
            self.body.pop()
        # Starvation is a terminal game rule, not an external time-limit truncation.
        if self.idle >= self.limit:
            self.done, reward = True, -1.0
        return self._observe(), reward, self.done, {"score": self.score}

    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        for x, y in self.body:
            grid[y][x] = "o"
        x, y = self.body[0]
        grid[y][x] = "H"
        if self.food is not None:
            x, y = self.food
            grid[y][x] = "*"
        print("\n".join(" ".join(row) for row in grid))
