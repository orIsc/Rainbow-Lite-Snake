"""Run: python train.py --episodes 2000; evaluate with --evaluate snake.pt."""
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from network import DuelingDQN
from replay import NStepAccumulator, PrioritizedReplayBuffer
from snake_env import SnakeEnv


def score_policy(online, size, episodes, seed, device, features=True):
    env = SnakeEnv(size, seed, features=features)
    was_training = online.training
    online.eval()
    scores = []
    with torch.no_grad():
        for _ in range(episodes):
            state, done = env.reset(), False
            while not done:
                action = online(torch.as_tensor(state, device=device).unsqueeze(0)).argmax(1).item()
                state, _, done, info = env.step(action)
            scores.append(info["score"])
    online.train(was_training)
    return float(np.mean(scores))


def learn(online, target, optimizer, replay, batch_size, beta, device):
    batch, indices, weights = replay.sample(batch_size, beta)
    states = torch.as_tensor(np.stack([t.state for t in batch]), device=device)
    next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), device=device)
    actions = torch.tensor([t.action for t in batch], device=device)
    rewards = torch.tensor([t.reward for t in batch], device=device)
    dones = torch.tensor([t.done for t in batch], device=device)
    discounts = torch.tensor([t.discount for t in batch], device=device)
    online.reset_noise()
    target.reset_noise()
    predicted = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        # Double DQN: online selects an action; the target network evaluates it.
        next_actions = online(next_states).argmax(dim=1, keepdim=True)
        next_values = target(next_states).gather(1, next_actions).squeeze(1)
        # Terminal suffixes never bootstrap; shorter suffixes carry gamma ** k.
        expected = rewards + discounts * next_values.masked_fill(dones, 0.0)
    td_errors = expected - predicted
    losses = F.smooth_l1_loss(predicted, expected, reduction="none")
    loss = (torch.as_tensor(weights, device=device) * losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    replay.update_priorities(indices, td_errors.detach().cpu().numpy())
    return loss.item()


def evaluate(args, device):
    saved = torch.load(args.evaluate, map_location=device, weights_only=True)
    # Old checkpoints used only the grid. Preserve their evaluation semantics.
    env = SnakeEnv(saved["size"], args.seed, features=saved.get("features", False))
    online = DuelingDQN(env.observation_size, hidden=saved["hidden"]).to(device)
    online.load_state_dict(saved["model"])
    online.eval()  # Deterministic learned mu parameters, no exploration noise.
    scores = []
    for episode in range(args.episodes):
        state, done = env.reset(), False
        while not done:
            if args.render:
                env.render()
                print()
                time.sleep(0.1)
            with torch.no_grad():
                action = online(torch.as_tensor(state, device=device).unsqueeze(0)).argmax(1).item()
            state, _, done, info = env.step(action)
        scores.append(info["score"])
        print(f"episode={episode + 1} score={scores[-1]}")
    print(f"mean_score={np.mean(scores):.2f}")


def train(args, device):
    env = SnakeEnv(args.size, args.seed)
    online = DuelingDQN(env.observation_size, hidden=args.hidden).to(device)
    target = DuelingDQN(env.observation_size, hidden=args.hidden).to(device)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    # Both stay in training mode to activate NoisyLinear, even inside no_grad().
    online.train()
    target.train()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    replay = PrioritizedReplayBuffer(args.capacity, seed=args.seed)
    accumulator = NStepAccumulator(args.gamma)
    steps = updates = 0
    scores = []
    loss = None
    best_score = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_output = output.with_name(output.stem + ".best" + output.suffix)

    def save_checkpoint(path, episode):
        torch.save({"model": online.state_dict(), "size": args.size,
                    "hidden": args.hidden, "features": True,
                    "episodes": episode, "steps": steps, "updates": updates}, path)

    for episode in range(1, args.episodes + 1):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            # Exploration comes solely from fresh learned parameter noise.
            online.reset_noise()
            with torch.no_grad():
                action = online(torch.as_tensor(state, device=device).unsqueeze(0)).argmax(1).item()
            next_state, reward, done, info = env.step(action)
            for transition in accumulator.append(state, action, reward, next_state, done):
                replay.add(transition)
            state = next_state
            steps += 1
            episode_reward += reward
            if len(replay) >= max(args.warmup, args.batch_size) and steps % args.train_every == 0:
                beta = min(1.0, 0.4 + 0.6 * updates / args.beta_steps)
                loss = learn(online, target, optimizer, replay, args.batch_size, beta, device)
                updates += 1
                if updates % args.target_every == 0:
                    target.load_state_dict(online.state_dict())
        scores.append(info["score"])
        if episode == 1 or episode % args.log_every == 0:
            loss_text = "n/a" if loss is None else f"{loss:.4f}"
            print(f"episode={episode} steps={steps} updates={updates} "
                  f"score={info['score']} mean100={np.mean(scores[-100:]):.2f} "
                  f"reward={episode_reward:.2f} loss={loss_text}", flush=True)
        if episode % args.eval_every == 0 or episode == args.episodes:
            # Fixed validation food sequences are separate from training sequences.
            # fork_rng prevents validation from changing the training noise sequence.
            with torch.random.fork_rng(devices=[]):
                validation = score_policy(online, args.size, args.eval_episodes,
                                          args.seed + 10_000, device)
            if validation > best_score:
                best_score = validation
                save_checkpoint(best_output, episode)
            print(f"validation_mean={validation:.2f} best={best_score:.2f}", flush=True)
    save_checkpoint(output, args.episodes)
    print(f"Saved evaluation checkpoint to {output}")
    print(f"Best validation checkpoint: {best_output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=50_000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-every", type=int, default=500)
    parser.add_argument("--beta-steps", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="snake.pt")
    parser.add_argument("--evaluate", metavar="CHECKPOINT")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    for name in ("episodes", "hidden", "batch_size", "capacity", "train_every",
                 "target_every", "beta_steps", "log_every", "threads", "eval_every", "eval_episodes"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not 0 <= args.gamma <= 1 or args.lr <= 0 or args.warmup < 0:
        parser.error("Invalid gamma, learning rate, or warmup")
    if not args.evaluate and args.capacity < max(args.warmup, args.batch_size):
        parser.error("capacity must accommodate warmup and batch size")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Install CUDA-enabled PyTorch or use --device cpu.")
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    else:
        print(f"Device: {device}", flush=True)
    if args.evaluate:
        evaluate(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
