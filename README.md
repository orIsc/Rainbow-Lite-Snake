# Rainbow DQN Snake

PyTorch implementation of CNN + dueling NoisyNet + C51 + Double DQN +
prioritized replay + configurable n-step returns (default 3).

## Files

- `snake_env.py`: Gymnasium environment with `reset(seed=...)` and the five-value
  `step()` API. Binary uint8 observation `(3, 10, 10)` ordered food, body, walls.
- `network.py`: CNN followed by `nn.Flatten`, separate noisy value and advantage
  streams, and 51 probabilities per action. Comments explain factorized noise
  and per-atom dueling aggregation.
- `replay.py`: proportional PER with importance weights and n-step aggregation.
- `train.py`: C51 projection, Double DQN selection/evaluation, weighted
  cross-entropy, KL-based priorities, target synchronization, training/evaluation.
- `test_components.py`: environment, noise, replay, projection, and optimizer tests.

## Install and run

Use Python 3.12 with the pinned PyTorch version. In an activated virtual environment:

```sh
python -m pip install -r requirements.txt
python -m unittest -v
python train.py --episodes 2000 --output rainbow.pt
python train.py --evaluate rainbow.best.pt --episodes 10 --seed 12345 --render
```

CPU is the default. Use `--device cuda` with a compatible CUDA-enabled PyTorch
installation. `--help` lists all hyperparameters. A quick training smoke test:

```sh
python train.py --episodes 3 --max-steps 30 --hidden 16 --batch-size 4 --warmup 4 --capacity 100 --target-every 2 --eval-episodes 2 --output rainbow-smoke.pt
```

## Gameplay popup

Training automatically records the highest-food episode (ties prefer higher total
reward) as `checkpoints/replays/<output-stem>.best-run.json` for bare output names. This records the actual moves, including
exploration, even though network weights change during training.

```powershell
.\.venv-c51\Scripts\python.exe train.py --episodes 2000 --output rainbow-10x10.pt --watch-best
.\.venv-c51\Scripts\python.exe viewer.py
```

`--watch-best` opens the popup in a separate process when training finishes, so
the training command exits immediately and releases its GPU resources.
The viewer defaults to `rainbow-10x10.best-run.json`.
You can also run `viewer.py`
in a second terminal during training to watch the best episode saved so far.
The viewer loads a snapshot when opened; reopen it to load a newer record.
The window has Pause/Play, Replay, and speed controls. Space pauses; Escape closes.
The bright green cell is the head, green cells the body, and pink circles food.

For existing checkpoints that have no recorded training trajectory:

```powershell
.\.venv-c51\Scripts\python.exe viewer.py --checkpoint rainbow.best.pt --episodes 20
```

This evaluates 20 games and displays the best evaluation, rather than reconstructing
an old training episode. Custom output names use a matching replay name, e.g.
`--output experiment.pt` saves `experiment.best-run.json`; open it with
`viewer.py --replay experiment.best-run.json`. The popup uses Python's Tkinter
(included in the prepared Windows environment), with no extra pip dependency.

## Environment conventions

The default 10x10 grid includes its outer wall border, leaving an 8x8 playable interior.
Actions are `0=up`, `1=down`, `2=left`, `3=right`. A reverse request continues
straight. Moving into the tail is legal when the tail moves away on that step.
Food gives +10; wall/self collision gives -10; other steps give -0.01.
Filling the interior terminates successfully with the final food reward.

Episodes truncate after 2000 steps by default to bound loops; change
`--max-steps` or construct `SnakeEnv(max_steps=None)` to disable the limit.
Time-limit truncations retain bootstrapping. Both truncations and terminations
flush n-step suffixes so replay never mixes episodes. Only true termination
suppresses the bootstrap. This follows the
[Gymnasium environment API](https://gymnasium.farama.org/api/env/).

The exact requested binary body plane includes the head but does not distinguish
it or encode heading/body order. Thus a single observation is partially observable.
This implementation preserves that specification; adding a head/heading channel,
frame stacking, or recurrence would be useful extensions for stronger learning.

## Learning details

NoisyLinear learns mean and noise-scale parameters. Factorized Gaussian noise is
resampled for each training action and optimizer update; there is no epsilon-greedy
policy. Evaluation uses the learned means (`eval()`). Online and target networks
use independent noise during learning; the target has no gradients.

The network returns `(batch, 4, 51)` distributions. Expected Q-values are
`sum(probability * atom_value)`. The online network selects the next action;
the target network supplies that action's categorical distribution. The n-step
Bellman target `R_n + gamma**k * Z` is clipped and linearly projected onto the
fixed support, including exact atom hits without losing probability mass.
Cross-entropy uses stable log-softmax, importance weights, and gradient clipping.
Unweighted KL divergence plus epsilon supplies PER priorities. Beta anneals from
0.4 to 1; target weights synchronize every 500 optimizer updates by default.

The 51-atom support defaults to [-10, 100], configurable with `--v-min/--v-max`.
This is an approximation: returns outside the support are clipped. Monitor and
adjust the support for longer/high-scoring runs. Rewards are not rescaled.

Replay stores uint8 observations (roughly 30 MB for two 10x10x3 arrays per
50,000 transitions, plus Python overhead). This readable implementation samples
PER in O(capacity); a sum/min tree is an optimization for larger replay buffers.

Validation runs every 100 episodes and saves `rainbow.best.pt`; the final model
is `rainbow.pt`. Lightweight model checkpoints support evaluation or continuation
from learned weights; full `.resume.pt` files preserve training state. Old Rainbow-Lite checkpoints are
incompatible with the new architecture and require fresh training. A smoke test
checks execution, not policy convergence or playing strength.

## Continue training

Continue your existing best 10x10 model for 2000 additional episodes, saving under
a new name to preserve the original baseline:

```powershell
python train.py --resume rainbow-10x10.best.pt --device cuda --episodes 2000 --output rainbow-10x10-continued.pt --watch-best
```

Older checkpoints restore model weights and counters, with a fresh optimizer and
empty replay buffer. Warmup refills replay before updates resume. Architecture
and grid size come from the checkpoint automatically.

New training runs save `OUTPUT.resume.pt` every 100 episodes (`--save-every`)
and at completion. These contain online/target networks, optimizer, replay data
and priorities, counters, training settings, environment/replay/Python/PyTorch RNG
states, recent scores, and the best recorded run. Save files are replaced atomically.
Checkpoints are at episode boundaries, so no partial n-step queue is required.
After interruption, resume from the latest saved boundary; unsaved episodes are lost.

```powershell
python train.py --resume rainbow-10x10-continued.resume.pt --device cuda --episodes 2000 --output rainbow-10x10-next.pt --watch-best
```

Full resume restores saved training hyperparameters, overriding their CLI values.
You can choose the device, output, additional episode count, logging interval,
save interval, and viewer option. Same-device CPU continuation was verified
against an uninterrupted run; bitwise equivalence across devices/CUDA is not promised.
The starting policy is evaluated and retained as a candidate for the new run's
best checkpoint, so subsequent regression does not discard it. Historical best
model files from earlier runs remain separate; use a new output name to keep them.
More training can improve or worsen performance; compare evaluation averages.


## Checkpoint folders

Artifacts are organized by file type:

- `checkpoints/best/`: best validation model weights (`*.best.pt`).
- `checkpoints/final/`: final model weights (`*.pt`).
- `checkpoints/resume/`: full training state (`*.resume.pt`).
- `checkpoints/replays/`: recorded best training games (`*.best-run.json`).

Existing artifacts have been moved into these folders. Bare filenames still work
with `--resume`, `--evaluate`, and the viewer; the scripts look in the matching
folder automatically. A bare `--output experiment.pt` writes each artifact into
its corresponding folder. An explicit output path such as `runs/test.pt` keeps
all artifacts together in that explicitly chosen directory.
