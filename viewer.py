"""Replay a recorded training run, or the best evaluation of a C51 checkpoint."""
import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from checkpoint_paths import resolve_artifact


def capture_frame(env):
    return {"body": [list(cell) for cell in env.body],
            "food": list(env.food) if env.food is not None else None,
            "score": env.score}


def show_replay(replay, title="Snake — best run"):
    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)
    root.configure(bg="#101827")
    size = replay["size"]
    cell = max(8, min(28, 640 // size))
    status = tk.StringVar()
    tk.Label(root, textvariable=status, bg="#101827", fg="white",
             font=("Segoe UI", 12), pady=12).pack()
    canvas = tk.Canvas(root, width=size * cell, height=size * cell,
                       bg="#101827", highlightthickness=0)
    canvas.pack(padx=16)
    controls = ttk.Frame(root, padding=12)
    controls.pack(fill="x")
    frames = replay["frames"]
    index, paused = 0, False
    speed = tk.IntVar(value=10)
    button_text = tk.StringVar(value="Pause")

    def draw():
        frame = frames[index]
        canvas.delete("all")
        for y in range(size):
            for x in range(size):
                color = "#334155" if x in (0, size - 1) or y in (0, size - 1) else "#182335"
                canvas.create_rectangle(x * cell, y * cell, (x + 1) * cell,
                                        (y + 1) * cell, fill=color, outline="#101827")
        for part, (x, y) in enumerate(frame["body"]):
            canvas.create_rectangle(x * cell + 2, y * cell + 2,
                                    (x + 1) * cell - 2, (y + 1) * cell - 2,
                                    fill="#a3e635" if part == 0 else "#22c55e", outline="")
        if frame["food"] is not None:
            x, y = frame["food"]
            canvas.create_oval(x * cell + 4, y * cell + 4,
                               (x + 1) * cell - 4, (y + 1) * cell - 4,
                               fill="#fb7185", outline="")
        ending = f"  •  {replay['ending']}" if index == len(frames) - 1 else ""
        status.set(f"Episode {replay['episode']}  •  Food {frame['score']}  •  "
                   f"Step {index}/{len(frames) - 1}{ending}")

    def toggle():
        nonlocal paused
        paused = not paused
        button_text.set("Play" if paused else "Pause")

    def restart():
        nonlocal index, paused
        index, paused = 0, False
        button_text.set("Pause")
        draw()

    def tick():
        nonlocal index, paused
        if not paused:
            if index < len(frames) - 1:
                index += 1
                draw()
            else:
                paused = True
                button_text.set("Play")
        root.after(max(1, 1000 // max(1, speed.get())), tick)

    ttk.Button(controls, textvariable=button_text, command=toggle).pack(side="left")
    ttk.Button(controls, text="Replay", command=restart).pack(side="left", padx=8)
    ttk.Label(controls, text="Speed (steps/sec)").pack(side="left", padx=8)
    ttk.Spinbox(controls, from_=1, to=60, textvariable=speed, width=5,
                state="readonly").pack(side="left")
    root.bind("<space>", lambda event: toggle())
    root.bind("<Escape>", lambda event: root.destroy())
    draw()
    root.after(100, tick)
    root.mainloop()


def best_checkpoint_run(path, episodes=20, seed=12345, max_steps=2000):
    import torch
    from network import DuelingDQN
    from snake_env import SnakeEnv

    torch.set_num_threads(1)
    saved = torch.load(resolve_artifact(path), map_location="cpu", weights_only=True)
    if saved.get("format") != "rainbow-c51-v1":
        raise ValueError("This viewer needs a Rainbow C51 checkpoint")
    model = DuelingDQN(saved["size"], hidden=saved["hidden"],
                       v_min=saved["v_min"], v_max=saved["v_max"])
    model.load_state_dict(saved["model"])
    model.eval()
    env = SnakeEnv(saved["size"], max_steps=max_steps)
    best = None
    with torch.inference_mode():
        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed + episode - 1)
            frames, reward_sum = [capture_frame(env)], 0.0
            while True:
                action = model.q_values(torch.as_tensor(state).unsqueeze(0)).argmax(1).item()
                state, reward, terminated, truncated, _ = env.step(action)
                reward_sum += reward
                frames.append(capture_frame(env))
                if terminated or truncated:
                    break
            run = {"size": env.size, "episode": episode, "score": env.score,
                   "reward": reward_sum, "frames": frames,
                   "ending": "Time limit" if truncated else "Board cleared" if env.food is None else "Collision"}
            if best is None or (run["score"], run["reward"]) > (best["score"], best["reward"]):
                best = run
            print(f"Evaluation {episode}/{episodes}: food={env.score}", flush=True)
    env.close()
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", default="rainbow-10x10.best-run.json")
    parser.add_argument("--checkpoint", help="Evaluate a checkpoint instead of playing a recorded run")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes and max-steps must be positive")
    if args.checkpoint:
        replay = best_checkpoint_run(args.checkpoint, args.episodes, args.seed, args.max_steps)
        title = "Snake — best checkpoint evaluation"
    else:
        path = resolve_artifact(args.replay)
        if not path.exists():
            parser.error(f"{path} not found. Train with --output rainbow-10x10.pt or specify --replay/--checkpoint")
        replay = json.loads(path.read_text(encoding="utf-8"))
        title = "Snake — best recorded training run"
    show_replay(replay, title)


if __name__ == "__main__":
    main()
