import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from gymnasium.utils.env_checker import check_env
from network import DuelingDQN, NoisyLinear
from replay import NStepAccumulator, PrioritizedReplayBuffer, Transition
from snake_env import SnakeEnv
from train import learn, project_distribution


torch.set_num_threads(1)


class ComponentTests(unittest.TestCase):
    def test_gym_contract(self):
        env = SnakeEnv()
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=7)
        self.assertEqual(obs.shape, (3, 10, 10))
        self.assertEqual(obs.dtype, np.uint8)
        self.assertEqual(obs[0].sum(), 1)
        self.assertEqual(obs[1].sum(), 3)
        self.assertEqual(obs[2].sum(), 36)

    def test_food_reverse_wall_and_timeout(self):
        env = SnakeEnv(size=20, max_steps=2)
        env.reset(seed=1)
        env.food = (11, 10)
        _, reward, terminated, truncated, _ = env.step(2)
        self.assertEqual(env.body[0], (11, 10))  # reverse ignored
        self.assertEqual(reward, 10)
        self.assertEqual(len(env.body), 4)
        self.assertFalse(terminated or truncated)
        env.food = (1, 1)
        _, reward, terminated, truncated, _ = env.step(3)
        self.assertEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        env.reset()
        env.body = [(18, 10), (17, 10), (16, 10)]
        _, reward, terminated, truncated, _ = env.step(3)
        self.assertEqual(reward, -10)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        with self.assertRaises(RuntimeError):
            env.step(3)

    def test_tail_self_collision_and_full_board(self):
        env = SnakeEnv(size=6)
        env.reset()
        env.body = [(2, 2), (2, 3), (1, 3), (1, 2)]
        env.direction, env.food = 2, (4, 4)
        self.assertFalse(env.step(2)[2])  # vacating tail is legal
        env.reset()
        env.body = [(2, 2), (2, 3), (3, 3), (3, 2), (3, 1)]
        env.direction = 0
        self.assertTrue(env.step(3)[2])
        env.reset()
        env.body = [(3, 1)] + [(x, y) for y in range(1, 5) for x in range(1, 5)
                               if (x, y) not in ((3, 1), (4, 1))]
        env.direction, env.food = 3, (4, 1)
        _, reward, terminated, _, _ = env.step(3)
        self.assertEqual(reward, 10)
        self.assertTrue(terminated)
        self.assertIsNone(env.food)

    def test_nstep_terminal_and_truncation(self):
        s = np.zeros((3, 10, 10), dtype=np.uint8)
        for terminal in (True, False):
            queue = NStepAccumulator(gamma=0.5, n=3)
            queue.append(s, 0, 1, s, False)
            queue.append(s, 0, 2, s, False)
            result = queue.append(s, 0, 4, s, terminal, not terminal)
            self.assertEqual([t.reward for t in result], [3, 4, 4])
            self.assertEqual([t.discount for t in result], [0.125, 0.25, 0.5])
            self.assertTrue(all(t.done == terminal for t in result))
            self.assertEqual(len(queue.waiting_room), 0)
        queue = NStepAccumulator(n=1)
        self.assertEqual(len(queue.append(s, 0, 1, s, False)), 1)

    def test_noise_and_categorical_dueling(self):
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
        model = DuelingDQN(hidden=16).eval()
        x = torch.zeros(2, 3, 10, 10)
        self.assertEqual(model(x).shape, (2, 4, 51))
        torch.testing.assert_close(model(x).sum(-1), torch.ones(2, 4))
        torch.testing.assert_close(model.logits(x).mean(1), model.value(model.features(x)))

    def test_nstep_nonterminal_window(self):
        queue = NStepAccumulator(gamma=0.5)
        s = np.zeros(1, dtype=np.uint8)
        queue.append(s, 0, 1, s, False)
        queue.append(s, 0, 2, s, False)
        first = queue.append(s, 0, 4, s, False)[0]
        self.assertEqual(first.reward, 3)
        self.assertEqual(first.discount, 0.125)
        self.assertFalse(first.done)
        self.assertEqual([t.reward for t in queue.append(s, 0, 8, s, True)], [6, 8, 8])

    def test_projection_mass_exact_atoms_clipping_and_terminal(self):
        support = torch.linspace(-10, 10, 51)
        probs = torch.rand(5, 51)
        probs /= probs.sum(-1, keepdim=True)
        result = project_distribution(probs, torch.tensor([0., -100., 100., 0., 0.2]),
                                      torch.tensor([False, True, True, True, True]),
                                      torch.ones(5), support)
        torch.testing.assert_close(result.sum(-1), torch.ones(5))
        self.assertTrue((result >= 0).all())
        torch.testing.assert_close(result[0], probs[0], atol=2e-7, rtol=1e-5)
        self.assertAlmostEqual(result[1, 0].item(), 1, places=6)
        self.assertAlmostEqual(result[2, -1].item(), 1, places=6)
        self.assertAlmostEqual(result[3, 25].item(), 1, places=6)
        torch.testing.assert_close(result[4, 25:27], torch.tensor([0.5, 0.5]), atol=2e-6, rtol=1e-5)

    def test_priorities_and_weights(self):
        buffer = PrioritizedReplayBuffer(2, alpha=1, seed=5)
        s = np.zeros(1, dtype=np.uint8)
        for _ in range(2):
            buffer.add(Transition(s, 0, 0., s, False, 0.99))
        buffer.update_priorities([0, 1, 1], [1, 9, 2])
        _, indices, weights = buffer.sample(10000, beta=1)
        self.assertGreater(np.mean(indices == 1), 0.85)
        np.testing.assert_allclose(weights[indices == 0], 1)
        np.testing.assert_allclose(weights[indices == 1], (1 + 1e-6) / (9 + 1e-6))

    def test_optimizer_and_target_isolation(self):
        online = DuelingDQN(hidden=16)
        target = copy.deepcopy(online).requires_grad_(False)
        before = [p.detach().clone() for p in target.parameters()]
        buffer = PrioritizedReplayBuffer(4)
        s = np.zeros((3, 10, 10), dtype=np.uint8)
        for done in (True, False):
            buffer.add(Transition(s, 1, 10., s, done, 0.99 ** 3))
        optimizer = torch.optim.Adam(online.parameters(), lr=0.001)
        loss = learn(online, target, optimizer, buffer, 4, 0.4, torch.device('cpu'))
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 0)
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(online.parameters(), before)))
        for p, q in zip(target.parameters(), before):
            torch.testing.assert_close(p, q)
            self.assertIsNone(p.grad)

    def test_double_dqn_uses_online_action_and_target_distribution(self):
        online = DuelingDQN(hidden=8)
        target = copy.deepcopy(online).requires_grad_(False)
        buffer = PrioritizedReplayBuffer(1)
        s = np.zeros((3, 10, 10), dtype=np.uint8)
        buffer.add(Transition(s, 0, 0., s, False, 1.))
        # Online selects action 2; target's own best action would be 3.
        distributions = torch.zeros(1, 4, 51)
        distributions[:, :, 0] = 1
        distributions[0, 2, 0] = 0
        distributions[0, 2, 25] = 1
        distributions[0, 3, 0] = 0
        distributions[0, 3, 50] = 1
        optimizer = torch.optim.Adam(online.parameters(), lr=0.001)
        with patch.object(online, 'q_values', return_value=torch.tensor([[0., 1., 3., 2.]])), \
             patch.object(target, 'forward', return_value=distributions), \
             patch('train.project_distribution', wraps=project_distribution) as projection:
            learn(online, target, optimizer, buffer, 1, 0.4, torch.device('cpu'))
        torch.testing.assert_close(projection.call_args.args[0], distributions[:, 2])


if __name__ == '__main__':
    unittest.main()
