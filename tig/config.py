"""Registry of every (model, dataset) combination: networks, weights and GA bounds.

Values are copied from the legacy ``legacy/<dataset>/<model>/tig_*.py`` scripts. Adding a
combination means adding one ``TigConfig`` entry; nothing else changes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# Original scripts of the paper, one folder per dataset and model: legacy/<dataset>/<model>/.
LEGACY = REPO_ROOT / "legacy"
# Checkpoints published on Google Drive (see README) go here, one folder per dataset.
WEIGHTS = REPO_ROOT / "weights"

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
    # Name -> factory for the generator's network and the classifier.
    networks: Dict[str, Callable[[], torch.nn.Module]] = field(default_factory=dict)
    image_shape: Tuple[int, int, int] = (1, 28, 28)
    legacy_script: Optional[Path] = None

    @property
    def ported(self) -> bool:
        return self.legacy_script is None


def _mnist_classifier():
    from tig.models.classifiers import MnistClassifier
    return MnistClassifier()


def _vgg():
    from tig.models.classifiers import VGGNet
    return VGGNet()


def _mnist_vae():
    from tig.models.vae import MlpVAE
    return MlpVAE(img_size=28 * 28, h_dim=1600, z_dim=400)


def _svhn_vae():
    from tig.models.vae import ResConvVAE
    return ResConvVAE(img_size=(32, 32), c_num=3, h_dim=4000, z_dim=800)


def _cifar10_vae():
    from tig.models.vae import StridedConvVAE
    return StridedConvVAE(c_num=3, h_dim=4000, z_dim=1024)


def _mnist_gan():
    from tig.models.gan import mnist_cdcgan
    return mnist_cdcgan(nz=100)


def _rgb32_gan():
    from tig.models.gan import rgb32_cdcgan
    return rgb32_cdcgan(nz=100)


_SVHN_CLASSIFIER = WEIGHTS / "svhn/SVHN_vggnet.pth"
_CIFAR10_CLASSIFIER = WEIGHTS / "cifar10/CIFAR10_cifar10_train.pynet.pth"
_MNIST_CLASSIFIER = LEGACY / "mnist/classifier/weights/mnist_classifier.pth"
_RGB = (3, 32, 32)

CONFIGS = {
    ("vae", "mnist"): TigConfig(
        model="vae", dataset="mnist",
        perturbation={"high": 0.0146425576210022, "low": None},
        min_val=-3.86494088172913, max_val=3.45633792877197,
        weights={"generator": LEGACY / "mnist/vae/weights/mnist_vae.pth", "classifier": _MNIST_CLASSIFIER},
        networks={"generator": _mnist_vae, "classifier": _mnist_classifier},
    ),
    ("vae", "svhn"): TigConfig(
        model="vae", dataset="svhn",
        perturbation={"high": 0.0199879989624023, "low": None},
        min_val=-5.26724052429199, max_val=4.72675895690918,
        weights={"generator": WEIGHTS / "svhn/svhn_convend.pth", "classifier": _SVHN_CLASSIFIER},
        networks={"generator": _svhn_vae, "classifier": _vgg},
        image_shape=_RGB,
    ),
    ("vae", "cifar10"): TigConfig(
        model="vae", dataset="cifar10",
        perturbation={"high": 0.0160218644142151, "low": None},
        min_val=-4.15868806838989, max_val=3.85224413871765,
        weights={"generator": WEIGHTS / "cifar10/cifar10_convend.pth", "classifier": _CIFAR10_CLASSIFIER},
        networks={"generator": _cifar10_vae, "classifier": _vgg},
        image_shape=_RGB,
    ),
    ("gan", "mnist"): TigConfig(
        model="gan", dataset="mnist",
        perturbation={"low": 0.00177406988143921, "high": None},
        min_val=-4.78276300430298, max_val=4.08758640289307,
        weights={
            "generator": LEGACY / "mnist/gan/weights/generator_epoch59.pth",
            "classifier": _MNIST_CLASSIFIER,
        },
        networks={"generator": _mnist_gan, "classifier": _mnist_classifier},
    ),
    ("gan", "svhn"): TigConfig(
        model="gan", dataset="svhn",
        perturbation={"high": 0.0172531118392944, "low": None},
        min_val=-4.07097911834717, max_val=4.55557680130005,
        weights={
            "generator": LEGACY / "svhn/gan/weights/generator_epoch99.pth",
            "classifier": _SVHN_CLASSIFIER,
        },
        networks={"generator": _rgb32_gan, "classifier": _vgg},
        image_shape=_RGB,
    ),
    ("gan", "cifar10"): TigConfig(
        model="gan", dataset="cifar10",
        # The legacy script does not label this value; it matches the MNIST GAN "low" extent.
        perturbation={"low": 0.00182106418609619, "high": None},
        min_val=-4.4389796257019, max_val=4.66634130477905,
        weights={
            "generator": LEGACY / "cifar10/gan/weights/generator_epoch699.pth",
            "classifier": _CIFAR10_CLASSIFIER,
        },
        networks={"generator": _rgb32_gan, "classifier": _vgg},
        image_shape=_RGB,
    ),
}

_LEGACY = {
    ("vae", "imagenet"): "imagenet/vae/tig_vae.py",
    ("gan", "imagenet"): "imagenet/gan/tig_biggan.py",
    ("sd", "mnist"): "mnist/sd/tig_sd.py",
    ("sd", "svhn"): "svhn/sd/tig_sd.py",
    ("sd", "cifar10"): "cifar10/sd/tig_sd.py",
    ("sd", "imagenet"): "imagenet/sd/tig_sd.py",
}
for _key, _path in _LEGACY.items():
    CONFIGS[_key] = TigConfig(model=_key[0], dataset=_key[1], legacy_script=LEGACY / _path)


def get_config(model: str, dataset: str) -> TigConfig:
    try:
        return CONFIGS[(model, dataset)]
    except KeyError:
        raise ValueError(f"unknown combination: {model} x {dataset}") from None
