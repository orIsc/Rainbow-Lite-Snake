"""Run: python train.py --episodes 2000; evaluate with --evaluate rainbow.pt."""
import argparse
import json
import os
import subprocess
import sys
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from network import DuelingDQN
from replay import NStepAccumulator, PrioritizedReplayBuffer
from snake_env import SnakeEnv
from checkpoint_paths import resolve_artifact, training_paths


TRAIN_SETTINGS = ("size", "hidden", "v_min", "v_max", "capacity", "gamma", "n_step",
                  "lr", "batch_size", "warmup", "train_every", "target_every",
                  "beta_steps", "seed", "max_steps", "eval_episodes", "eval_every")


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def launch_viewer(replay_path):
    """Start independent playback so training can exit and release its GPU."""
    executable = Path(sys.executable)
    options = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
               "stderr": subprocess.DEVNULL, "close_fds": True}
    if os.name == "nt":
        windowless = executable.with_name("pythonw.exe")
        if windowless.exists():
            executable = windowless
        options["creationflags"] = (subprocess.DETACHED_PROCESS
                                    | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        options["start_new_session"] = True
    subprocess.Popen([str(executable), str(Path(__file__).with_name("viewer.py").resolve()),
                      "--replay", str(Path(replay_path).resolve())], **options)


def score_policy(online, size, episodes, seed, device, max_steps=2000):
    env = SnakeEnv(size, max_steps=max_steps)
    was_training = online.training
    online.eval()
    scores = []
    with torch.no_grad():
        for episode in range(episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            done = False
            while not done:
                action = online.q_values(torch.as_tensor(state, device=device).unsqueeze(0)).argmax(1).item()
                state, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            scores.append(info["score"])
    online.train(was_training)
    return float(np.mean(scores))


@torch.no_grad()
def project_distribution(next_probs, rewards, terminated, discounts, support):
    """Project R_n + gamma**k * Z onto fixed equally spaced C51 atoms."""
    atoms = support.numel()
    delta = (support[-1] - support[0]) / (atoms - 1)
    shifted = rewards[:, None] + discounts[:, None] * (~terminated[:, None]) * support
    positions = ((shifted.clamp(support[0], support[-1]) - support[0]) / delta)
    positions = positions.clamp(0, atoms - 1)
    lower, upper = positions.floor().long(), positions.ceil().long()
    projected = torch.zeros_like(next_probs)
    # Exact atom hits need weight 1, not two zero interpolation weights.
    projected.scatter_add_(1, lower, next_probs * (upper - positions + (lower == upper)))
    projected.scatter_add_(1, upper, next_probs * (positions - lower))
    return projected


def learn(online, target, optimizer, replay, batch_size, beta, device):
    batch, indices, weights = replay.sample(batch_size, beta)
    states = torch.as_tensor(np.stack([t.state for t in batch]), device=device)
    next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), device=device)
    actions = torch.tensor([t.action for t in batch], device=device)
    rewards = torch.tensor([t.reward for t in batch], device=device, dtype=torch.float32)
    dones = torch.tensor([t.done for t in batch], device=device)
    discounts = torch.tensor([t.discount for t in batch], device=device, dtype=torch.float32)
    online.reset_noise()
    target.reset_noise()
    rows = torch.arange(batch_size, device=device)
    log_probs = F.log_softmax(online.logits(states), dim=-1)[rows, actions]
    with torch.no_grad():
        # Double DQN: online selects an action; the target network evaluates it.
        next_actions = online.q_values(next_states).argmax(dim=1)
        next_probs = target(next_states)[rows, next_actions]
        expected = project_distribution(next_probs, rewards, dones, discounts, online.support)
    # Stable categorical cross-entropy; PER weights only affect the optimizer loss.
    losses = -(expected * log_probs).sum(-1)
    loss = (torch.as_tensor(weights, device=device) * losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    # KL divergence measures distributional surprise without target entropy.
    with torch.no_grad():
        priorities = (losses + (expected * expected.clamp_min(1e-8).log()).sum(-1)).clamp_min(0)
    replay.update_priorities(indices, priorities.cpu().numpy())
    return loss.item()


def evaluate(args, device):
    saved = torch.load(resolve_artifact(args.evaluate), map_location=device, weights_only=True)
    if saved.get("format") != "rainbow-c51-v1":
        raise ValueError("Legacy scalar-Q checkpoints require the old implementation; train a new C51 model")
    env = SnakeEnv(saved["size"], max_steps=args.max_steps,
                   render_mode="ansi" if args.render else None)
    online = DuelingDQN(saved["size"], hidden=saved["hidden"],
                        v_min=saved["v_min"], v_max=saved["v_max"]).to(device)
    online.load_state_dict(saved["model"])
    online.eval()  # Deterministic learned mu parameters, no exploration noise.
    scores = []
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed if episode == 0 else None)
        done = False
        while not done:
            if args.render:
                print(env.render())
                time.sleep(0.1)
            with torch.no_grad():
                action = online.q_values(torch.as_tensor(state, device=device).unsqueeze(0)).argmax(1).item()
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        scores.append(info["score"])
        print(f"episode={episode + 1} score={scores[-1]}")
    print(f"mean_score={np.mean(scores):.2f}")


def train(args, device):
    saved = args.resume_data
    env = SnakeEnv(args.size, max_steps=args.max_steps)
    online = DuelingDQN(args.size, hidden=args.hidden, v_min=args.v_min, v_max=args.v_max).to(device)
    target = DuelingDQN(args.size, hidden=args.hidden, v_min=args.v_min, v_max=args.v_max).to(device)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    # Both stay in training mode to activate NoisyLinear, even inside no_grad().
    online.train()
    target.train()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    replay = PrioritizedReplayBuffer(args.capacity, seed=args.seed)
    accumulator = NStepAccumulator(args.gamma, n=args.n_step)
    steps = updates = 0
    scores = []
    loss = None
    best_score = -1.0
    best_run = None
    start_episode = 0
    if saved:
        online.load_state_dict(saved["model"])
        target.load_state_dict(saved.get("target", saved["model"]))
        start_episode = saved.get("episodes", 0)
        steps, updates = saved.get("steps", 0), saved.get("updates", 0)
        if "training" in saved:
            state = saved["training"]
            optimizer.load_state_dict(state["optimizer"])
            replay.load_state_dict(state["replay"])
            env.reset(seed=args.seed)
            env.np_random.bit_generator.state = state["env_rng"]
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "cuda" and state["cuda_rng"]:
                if len(state["cuda_rng"]) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(state["cuda_rng"])
            random.setstate(state["python_rng"])
            scores = state["scores"]
            best_run = state["best_run"]
            print(f"Restored full training state at episode {start_episode}.", flush=True)
        else:
            env.reset(seed=args.seed + start_episode)
            print("Restored model weights; older checkpoint has no optimizer/replay state. "
                  "Replay will refill before learning.", flush=True)
    output, best_output, replay_output, resume_output = training_paths(args.output)
    if best_run is not None:
        replay_output.write_text(json.dumps(best_run), encoding="utf-8")
    # Import lazily: ordinary training does not require a GUI installation.
    def frame():
        return {"body": [list(cell) for cell in env.body],
                "food": list(env.food) if env.food is not None else None, "score": env.score}

    def save_checkpoint(path, episode, full=False):
        payload = {"format": "rainbow-c51-v1", "model": online.state_dict(), "size": args.size,
                    "hidden": args.hidden, "v_min": args.v_min, "v_max": args.v_max,
                    "episodes": episode, "steps": steps, "updates": updates}
        if full:
            # Saved only at episode boundaries: the n-step queue is empty.
            payload["target"] = target.state_dict()
            payload["settings"] = {key: getattr(args, key) for key in TRAIN_SETTINGS}
            payload["training"] = {"optimizer": optimizer.state_dict(), "replay": replay.state_dict(),
                "env_rng": env.np_random.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "python_rng": random.getstate(), "scores": scores[-100:], "best_run": best_run}
        atomic_save(payload, path)

    if saved:
        # Keep the starting policy as a candidate, even if continued training regresses.
        best_score = score_policy(online, args.size, args.eval_episodes,
                                  args.seed + 10_000, device, args.max_steps)
        save_checkpoint(best_output, start_episode)
        print(f"Starting validation_mean={best_score:.2f}", flush=True)

    final_episode = start_episode + args.episodes
    for episode in range(start_episode + 1, final_episode + 1):
        state, _ = env.reset(seed=args.seed if not saved and episode == 1 else None)
        done = False
        episode_reward = 0.0
        frames = [frame()]
        while not done:
            # Exploration comes solely from fresh learned parameter noise.
            online.reset_noise()
            with torch.no_grad():
                action = online.q_values(torch.as_tensor(state, device=device).unsqueeze(0)).argmax(1).item()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            frames.append(frame())
            for transition in accumulator.append(state, action, reward, next_state, terminated, truncated):
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
        if best_run is None or (info["score"], episode_reward) > (best_run["score"], best_run["reward"]):
            best_run = {"size": args.size, "episode": episode, "score": info["score"],
                        "reward": episode_reward, "frames": frames,
                        "ending": "Time limit" if truncated else "Board cleared" if env.food is None else "Collision"}
            temporary = replay_output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(best_run), encoding="utf-8")
            temporary.replace(replay_output)
        if episode == start_episode + 1 or episode % args.log_every == 0:
            loss_text = "n/a" if loss is None else f"{loss:.4f}"
            print(f"episode={episode} steps={steps} updates={updates} "
                  f"score={info['score']} mean100={np.mean(scores[-100:]):.2f} "
                  f"reward={episode_reward:.2f} loss={loss_text}", flush=True)
        if episode % args.eval_every == 0 or episode == final_episode:
            # Fixed validation food sequences are separate from training sequences.
            # fork_rng prevents validation from changing the training noise sequence.
            with torch.random.fork_rng(devices=[]):
                validation = score_policy(online, args.size, args.eval_episodes,
                                          args.seed + 10_000, device, args.max_steps)
            if validation > best_score:
                best_score = validation
                save_checkpoint(best_output, episode)
            print(f"validation_mean={validation:.2f} best={best_score:.2f}", flush=True)
        if episode % args.save_every == 0 or episode == final_episode:
            save_checkpoint(resume_output, episode, full=True)
    save_checkpoint(output, final_episode)
    print(f"Saved evaluation checkpoint to {output}")
    print(f"Best validation checkpoint: {best_output}")
    print(f"Best training replay: {replay_output}")
    print(f"Resume training checkpoint: {resume_output}")
    if args.watch_best:
        launch_viewer(replay_output)
        print("Viewer opened separately. Training complete.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=100.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=50_000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-every", type=int, default=500)
    parser.add_argument("--beta-steps", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="rainbow.pt")
    parser.add_argument("--evaluate", metavar="CHECKPOINT")
    parser.add_argument("--resume", metavar="CHECKPOINT", help="Continue training; --episodes is additional episodes")
    parser.add_argument("--save-every", type=int, default=100, help="Save full training state every N episodes")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--watch-best", action="store_true", help="Open the best run in a separate process after training")
    args = parser.parse_args()
    if args.resume and args.evaluate:
        parser.error("Use either --resume or --evaluate")
    args.resume_data = None
    if args.resume:
        args.resume_data = torch.load(resolve_artifact(args.resume), map_location="cpu", weights_only=True)
        if args.resume_data.get("format") != "rainbow-c51-v1":
            parser.error("Resume requires a Rainbow C51 checkpoint")
        settings = args.resume_data.get("settings", {})
        for key in TRAIN_SETTINGS:
            if key in settings:
                setattr(args, key, settings[key])
        for key in ("size", "hidden", "v_min", "v_max"):
            setattr(args, key, args.resume_data[key])
        if settings:
            print("Using saved training hyperparameters; --episodes specifies additional episodes.")
    for name in ("episodes", "hidden", "batch_size", "capacity", "train_every",
                 "target_every", "beta_steps", "log_every", "threads", "eval_every", "eval_episodes",
                 "max_steps", "n_step", "save_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not 0 <= args.gamma <= 1 or args.lr <= 0 or args.warmup < 0:
        parser.error("Invalid gamma, learning rate, or warmup")
    if args.size < 6 or not np.isfinite([args.v_min, args.v_max]).all() or args.v_min >= args.v_max:
        parser.error("size must be >= 6 and finite v_min < v_max")
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
