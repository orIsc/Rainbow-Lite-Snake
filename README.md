# Rainbow-Lite Snake

A from-scratch grid environment and PyTorch agent. Use Python 3.12 with the
pinned PyTorch 2.6 dependency. No Gym, rendering library, or external Snake
engine is needed. Virtual environments, caches, and generated checkpoints are
excluded from the repository.

## Run

Clone the repository and install with Python 3.12 (Windows PowerShell):

```powershell
git clone https://github.com/orIsc/Rainbow-Lite-Snake.git
cd Rainbow-Lite-Snake
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

To train a fresh model and evaluate its best validation checkpoint:

```powershell
.\.venv312\Scripts\python.exe train.py --episodes 2000
.\.venv312\Scripts\python.exe train.py --evaluate snake.best.pt --episodes 5 --render
.\.venv312\Scripts\python.exe -m unittest -v
```

Use the explicit environment interpreter instead of bare `python` or `pip` to
avoid accidentally using a different installed Python version. In VS Code,
select `.venv312\Scripts\python.exe` as the interpreter. Python 3.14 is not
compatible with the pinned PyTorch 2.6 package.

For macOS/Linux, create the environment with `python3.12 -m venv .venv312`,
replace `.\.venv312\Scripts\python.exe` with `.venv312/bin/python`, and install
with `.venv312/bin/python -m pip install -r requirements.txt`.
CPU is the default; pass `--device cuda` for a compatible CUDA installation.
PyTorch 2.6 CPU was verified locally; the newer 2.14 build failed to initialize
its native DLLs on this Windows machine.

## Local benchmark

A checkpoint saved after 300 training episodes on the 8x8 board averaged 11.10
food on 100 test games with seed 12345, compared with 0.24 for the original
grid-only model. Test seeds were separate from training and checkpoint-selection
seeds. These are results from one training run, not a guarantee across training
seeds. The verification run was stopped after the 300-episode checkpoint.
The locally trained `snake-improved.best.pt` is not included in this repository;
train a model using the commands above to generate your own checkpoint.

## Components

* `snake_env.py`: grid Snake with straight/left/right actions. Observations encode
  ordered body positions, head, food, heading, and starvation budget, plus seven
  derived features: danger for each action, normalized forward/right food offsets,
  and their signs. These help the MLP learn across positions without masking any
  action or providing a scripted policy. Entering a
  moving tail is legal. Food gives +1, collision/starvation -1, other steps -0.01.
* `network.py`: every fully connected layer is a custom factorized Gaussian
  `NoisyLinear` with learned weight/bias means and standard deviations. Separate
  value and advantage streams combine as `Q = V + A - mean(A)`.
* `replay.py`: `deque(maxlen=3)` aggregates returns, then proportional PER stores
  them. New entries receive the current maximum priority. Sampling uses
  `P(i) = priority(i)**alpha / sum(priority**alpha)` with replacement.
* `train.py`: noise-based action selection, weighted Huber loss, TD-error priority
  updates, target synchronization, checkpoint saving, and deterministic evaluation.
  Defaults use one optimizer update per step after warmup and learning rate 0.0005.

Every 100 episodes, training evaluates the deterministic policy on 20 fixed
validation games using a separate seed. `snake.best.pt` contains the best
validation checkpoint, while `snake.pt` contains the final weights. Evaluate with
a different seed for an independent performance check, for example
`--evaluate snake.best.pt --episodes 100 --seed 12345`.
The reported score counts food eaten, not shaped reward or snake length.
Old grid-only checkpoints still load, but the new features require fresh training.

## How the upgrades interact

1. Fresh network noise drives each greedy training action, including during replay
   warmup. There is no epsilon-greedy action selection. Noise is also independently
   refreshed in online and target networks for each optimization step.
2. The waiting room produces `R = r0 + gamma*r1 + gamma**2*r2`. When an episode
   ends it flushes all remaining suffixes, each with its actual `gamma**k` and
   terminal flag. No transitions span episodes.
3. PER draws these aggregated transitions and supplies normalized importance
   weights `(N*P(i))**(-beta)`. Beta anneals from 0.4 to 1 over optimizer updates.
4. Dueling Q values feed the Double DQN target:
   `R + gamma**k * (1-done) * Q_target(next_state, argmax Q_online(next_state))`.
   The weighted Huber loss trains both streams and the noise parameters. Absolute
   unweighted TD errors plus a small epsilon become the new replay priorities.

The target remains in training mode so its NoisyLinear layers remain stochastic;
`torch.no_grad()` prevents target gradients. Evaluation uses `eval()` to select
actions from learned means alone. Starvation is an actual terminal game rule.

This is a compact reference implementation: PER sampling and maximum-priority
lookup are O(buffer size). A sum tree is a useful later optimization for large
buffers. Checkpoints contain model weights and architecture for evaluation, not
the replay/optimizer/RNG state needed to resume training exactly. Training quality
depends on run length and seed; a smoke test does not establish convergence.

Quick smoke test:

```powershell
.\.venv312\Scripts\python.exe train.py --episodes 10 --size 4 --hidden 32 --batch-size 8 --warmup 8 --train-every 1 --target-every 5 --log-every 1 --output smoke.pt
```
