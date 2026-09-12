import copy
import unittest
import numpy as np
import torch
from network import DuelingDQN, NoisyLinear
from replay import NStepAccumulator, PrioritizedReplayBuffer, Transition
from snake_env import SnakeEnv
from train import learn


class ComponentTests(unittest.TestCase):
    def test_relative_features_match_game_rules(self):
        env = SnakeEnv(4, seed=3)
        rng = np.random.default_rng(1)
        for _ in range(100):
            observation = env._observe()
            self.assertEqual(observation.shape, (env.observation_size,))
            self.assertEqual(observation.dtype, np.float32)
            for action in range(3):
                probe = copy.deepcopy(env)
                _, reward, done, _ = probe.step(action)
                self.assertEqual(bool(observation[-7 + action]), done and reward == -1)
            _, _, done, _ = env.step(int(rng.integers(3)))
            if done:
                env.reset()
        legacy = SnakeEnv(4, features=False)
        self.assertEqual(legacy.reset().shape, (3 * 4 * 4 + 5,))
        env.body = [(1, 1), (0, 1), (0, 2)]
        env.direction = 1
        env.food = (3, 0)
        np.testing.assert_allclose(env._observe()[-4:], [2 / 3, -1 / 3, 1, -1])

    def test_terminal_flush(self):
        queue = NStepAccumulator(gamma=0.5)
        states = [np.array([i], dtype=np.float32) for i in range(4)]
        self.assertEqual(queue.append(states[0], 0, 1, states[1], False), [])
        self.assertEqual(queue.append(states[1], 1, 2, states[2], False), [])
        result = queue.append(states[2], 2, 4, states[3], True)
        self.assertEqual([t.reward for t in result], [3, 4, 4])
        self.assertEqual([t.discount for t in result], [0.125, 0.25, 0.5])
        self.assertTrue(all(t.done for t in result))
        self.assertEqual(len(queue.waiting_room), 0)
        self.assertEqual(queue.append(states[0], 0, 9, states[1], True)[0].reward, 9)

    def test_nonterminal_window_and_short_episode(self):
        queue = NStepAccumulator(0.5)
        s = np.zeros(1, dtype=np.float32)
        queue.append(s, 0, 1, s, False)
        queue.append(s, 0, 2, s, False)
        transition = queue.append(s, 0, 4, s, False)[0]
        self.assertEqual(transition.reward, 3)
        self.assertFalse(transition.done)
        self.assertEqual([t.reward for t in queue.append(s, 0, 8, s, True)], [6, 8, 8])
        queue.append(s, 0, 1, s, False)
        self.assertEqual([t.reward for t in queue.append(s, 0, 2, s, True)], [2, 2])

    def test_noise_and_dueling(self):
        torch.manual_seed(4)
        layer = NoisyLinear(5, 3)
        x = torch.ones(4, 5)
        first = layer(x)
        layer.reset_noise()
        self.assertFalse(torch.equal(first, layer(x)))
        layer(x).sum().backward()
        self.assertGreater(layer.weight_sigma.grad.abs().sum().item(), 0)
        layer.eval()
        first = layer(x)
        layer.reset_noise()
        torch.testing.assert_close(first, layer(x))
        model = DuelingDQN(5, hidden=16).eval()
        torch.testing.assert_close(model(x).mean(-1, keepdim=True), model.value(model.features(x)))

    def test_priorities_and_weights(self):
        buffer = PrioritizedReplayBuffer(2, alpha=1, seed=5)
        s = np.zeros(1, dtype=np.float32)
        for _ in range(2):
            buffer.add(Transition(s, 0, 0, s, False, 0.99))
        buffer.update_priorities([0, 1, 1], [1, 9, 2])
        _, indices, weights = buffer.sample(10_000, beta=1)
        self.assertGreater(np.mean(indices == 1), 0.85)
        np.testing.assert_allclose(weights[indices == 0], 1)
        np.testing.assert_allclose(weights[indices == 1], (1 + 1e-6) / (9 + 1e-6))

    def test_snake_tail_and_terminal(self):
        env = SnakeEnv(4)
        env.body = [(1, 1), (1, 2), (0, 2), (0, 1)]
        env.direction = 3
        env.food = (3, 3)
        _, _, done, _ = env.step(0)
        self.assertFalse(done)
        _, reward, done, _ = env.step(0)
        self.assertTrue(done)
        self.assertEqual(reward, -1)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_terminal_target_and_optimizer(self):
        torch.manual_seed(2)
        online = DuelingDQN(5, hidden=8)
        target = DuelingDQN(5, hidden=8)
        with torch.no_grad():
            for parameter in online.parameters():
                parameter.zero_()
            for parameter in target.parameters():
                parameter.zero_()
            target.value[-1].bias_mu.fill_(100)
        buffer = PrioritizedReplayBuffer(2)
        s = np.ones(5, dtype=np.float32)
        buffer.add(Transition(s, 0, 2, s, True, 0.99 ** 3))
        optimizer = torch.optim.Adam(online.parameters(), lr=0.001)
        loss = learn(online, target, optimizer, buffer, 2, 0.4, torch.device("cpu"))
        self.assertAlmostEqual(loss, 1.5)
        self.assertAlmostEqual(buffer.priorities[0], 2 + 1e-6)
        self.assertGreater(online.value[-1].bias_mu.item(), 0)


if __name__ == "__main__":
    unittest.main()
