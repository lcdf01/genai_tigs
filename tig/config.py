"""Paths and hyperparameters for every (model, dataset) combination.

Values are copied from the legacy ``experiments/*/tig_*`` scripts; a
combination that has not been ported yet points at its legacy script.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"

MODELS = ("vae", "gan", "sd")
DATASETS = ("mnist", "svhn", "cifar10", "imagenet")
MUTATIONS = ("low", "high")


@dataclass(frozen=True)
class TigConfig:
    model: str
    dataset: str
    # Initial perturbation per mutation level; None = not documented in the repo.
    perturbation: Dict[str, Optional[float]] = field(default_factory=dict)
    min_val: float = float("-inf")
    max_val: float = float("inf")
    weights: Dict[str, Path] = field(default_factory=dict)
    legacy_script: Optional[Path] = None

    @property
    def ported(self) -> bool:
        return self.legacy_script is None


_MNIST = EXPERIMENTS / "mnist_experiments"

CONFIGS = {
    ("vae", "mnist"): TigConfig(
        model="vae",
        dataset="mnist",
        perturbation={"high": 0.0146425576210022, "low": None},
        min_val=-3.86494088172913,
        max_val=3.45633792877197,
        weights={
            "vae": _MNIST / "mnist_vae/vae/weights/MNIST_EnD.pth",
            "classifier": _MNIST / "mnist_classifier/weight/MNIST_conv_classifier.pth",
        },
    ),
}

_LEGACY = {
    ("vae", "svhn"): "svhn_experiments/svhn_vae/tig_vae_svhn.py",
    ("vae", "cifar10"): "cifar10_experiments/cifar10_vae/tig_vae_cifar10.py",
    ("vae", "imagenet"): "imagenet_experiments/imagenet_vae/tig_vae_imagenet.py",
    ("gan", "mnist"): "mnist_experiments/mnist_gan/conditional_gan/tig_cdcgan_mnist.py",
    ("gan", "svhn"): "svhn_experiments/svhn_gan/conditional_gan/tig_cdcgan_svhn.py",
    ("gan", "cifar10"): "cifar10_experiments/cifar10_gan/conditional_gan/tig_cdcgan_cifar10.py",
    ("gan", "imagenet"): "imagenet_experiments/imagenet_gan/conditional_gan/tig_biggan_imagenet.py",
    ("sd", "mnist"): "mnist_experiments/mnist_sd/conditional_sd/tig_sd_mnist.py",
    ("sd", "svhn"): "svhn_experiments/svhn_sd/conditional_sd/tig_sd_svhn.py",
    ("sd", "cifar10"): "cifar10_experiments/cifar10_sd/conditional_sd/tig_sd_cifar10.py",
    ("sd", "imagenet"): "imagenet_experiments/imagenet_sd/ conditional_sd/tig_sd_imagenet.py",
}
for _key, _path in _LEGACY.items():
    CONFIGS[_key] = TigConfig(model=_key[0], dataset=_key[1], legacy_script=EXPERIMENTS / _path)


def get_config(model: str, dataset: str) -> TigConfig:
    try:
        return CONFIGS[(model, dataset)]
    except KeyError:
        raise ValueError(f"unknown combination: {model} x {dataset}") from None
