"""Shared locations for model weights, training state, and gameplay recordings."""
from pathlib import Path


def artifact_folder(name):
    if name.endswith(".best-run.json"):
        return "replays"
    if name.endswith(".resume.pt"):
        return "resume"
    if name.endswith(".best.pt"):
        return "best"
    return "final"


def resolve_artifact(value):
    """Accept explicit paths and old bare filenames after organization."""
    path = Path(value)
    if path.exists() or path.parent != Path("."):
        return path
    return Path("checkpoints") / artifact_folder(path.name) / path.name


def training_paths(value):
    path = Path(value)
    names = (path.name, path.stem + ".best" + path.suffix,
             path.stem + ".best-run.json", path.stem + ".resume" + path.suffix)
    if path.parent == Path("."):
        paths = tuple(Path("checkpoints") / artifact_folder(name) / name for name in names)
    else:
        # Explicit output directories retain the caller's chosen layout.
        paths = tuple(path.parent / name for name in names)
    for item in paths:
        item.parent.mkdir(parents=True, exist_ok=True)
    return paths
