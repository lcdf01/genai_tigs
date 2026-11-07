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
DATASETS = ("mnist", "svhn", "cifar10", "imagenet_pizza", "imagenet_teddy")
MUTATIONS = ("low", "high")
# Every legacy script sets the initial perturbation to a fixed fraction of the
# latent range (max_val - min_val); the fraction is the mutation extent.
MUTATION_FRACTION = {"low": 0.0002, "high": 0.002}
# ImageNet-1k class indices used by the paper.
PIZZA, TEDDY_BEAR = 963, 850
IMAGENET_CLASSES = {"imagenet_pizza": PIZZA, "imagenet_teddy": TEDDY_BEAR}


@dataclass(frozen=True)
class TigConfig:
    model: str
    dataset: str
    min_val: float = float("-inf")
    max_val: float = float("inf")
    # Checkpoint files; a network missing here downloads its own pretrained weights.
    weights: Dict[str, Path] = field(default_factory=dict)
    # Name -> factory for the generator's network and the classifier.
    networks: Dict[str, Callable[[], torch.nn.Module]] = field(default_factory=dict)
    image_shape: Tuple[int, int, int] = (1, 28, 28)
    # Classes the seeds are drawn from (None = all 10 digits/classes).
    classes: Optional[Tuple[int, ...]] = None
    # Prior seeds are drawn from N(0, I) truncated to [-bound, bound] (None = untruncated).
    prior_bound: Optional[float] = None
    legacy_script: Optional[Path] = None

    @property
    def ported(self) -> bool:
        return self.legacy_script is None

    def perturbation(self, mutation: str) -> float:
        return MUTATION_FRACTION[mutation] * (self.max_val - self.min_val)


def _mnist_classifier():
    from tig.models.classifiers import MnistClassifier
    return MnistClassifier()


def _vgg():
    from tig.models.classifiers import VGGNet
    return VGGNet()


def _vgg19_bn_imagenet():
    from tig.models.classifiers import vgg19_bn_imagenet
    return vgg19_bn_imagenet()


def _mnist_vae():
    from tig.models.vae import MlpVAE
    return MlpVAE(img_size=28 * 28, h_dim=1600, z_dim=400)


def _svhn_vae():
    from tig.models.vae import ResConvVAE
    return ResConvVAE(img_size=(32, 32), c_num=3, h_dim=4000, z_dim=800)


def _cifar10_vae():
    from tig.models.vae import StridedConvVAE
    return StridedConvVAE(c_num=3, h_dim=4000, z_dim=1024)


def _imagenet_vae():
    from tig.models.vae import ImageNetVAE
    return ImageNetVAE(latent_dim=512)


def _mnist_gan():
    from tig.models.gan import mnist_cdcgan
    return mnist_cdcgan(nz=100)


def _rgb32_gan():
    from tig.models.gan import rgb32_cdcgan
    return rgb32_cdcgan(nz=100)


def _biggan():
    from tig.models.gan import biggan_deep_256
    return biggan_deep_256()


_SVHN_CLASSIFIER = WEIGHTS / "svhn/SVHN_vggnet.pth"
_CIFAR10_CLASSIFIER = WEIGHTS / "cifar10/CIFAR10_cifar10_train.pynet.pth"
_MNIST_CLASSIFIER = LEGACY / "mnist/classifier/weights/mnist_classifier.pth"
_RGB = (3, 32, 32)

CONFIGS = {
    ("vae", "mnist"): TigConfig(
        model="vae", dataset="mnist",
        min_val=-3.86494088172913, max_val=3.45633792877197,
        weights={"generator": LEGACY / "mnist/vae/weights/mnist_vae.pth", "classifier": _MNIST_CLASSIFIER},
        networks={"generator": _mnist_vae, "classifier": _mnist_classifier},
    ),
    ("vae", "svhn"): TigConfig(
        model="vae", dataset="svhn",
        min_val=-5.26724052429199, max_val=4.72675895690918,
        weights={"generator": WEIGHTS / "svhn/svhn_convend.pth", "classifier": _SVHN_CLASSIFIER},
        networks={"generator": _svhn_vae, "classifier": _vgg},
        image_shape=_RGB,
    ),
    ("vae", "cifar10"): TigConfig(
        model="vae", dataset="cifar10",
        min_val=-4.15868806838989, max_val=3.85224413871765,
        weights={"generator": WEIGHTS / "cifar10/cifar10_convend.pth", "classifier": _CIFAR10_CLASSIFIER},
        networks={"generator": _cifar10_vae, "classifier": _vgg},
        image_shape=_RGB,
    ),
    ("gan", "mnist"): TigConfig(
        model="gan", dataset="mnist",
        min_val=-4.78276300430298, max_val=4.08758640289307,
        weights={"generator": LEGACY / "mnist/gan/weights/generator_epoch59.pth", "classifier": _MNIST_CLASSIFIER},
        networks={"generator": _mnist_gan, "classifier": _mnist_classifier},
    ),
    ("gan", "svhn"): TigConfig(
        model="gan", dataset="svhn",
        min_val=-4.07097911834717, max_val=4.55557680130005,
        weights={"generator": LEGACY / "svhn/gan/weights/generator_epoch99.pth", "classifier": _SVHN_CLASSIFIER},
        networks={"generator": _rgb32_gan, "classifier": _vgg},
        image_shape=_RGB,
    ),
    ("gan", "cifar10"): TigConfig(
        model="gan", dataset="cifar10",
        min_val=-4.4389796257019, max_val=4.66634130477905,
        weights={
            "generator": LEGACY / "cifar10/gan/weights/generator_epoch699.pth",
            "classifier": _CIFAR10_CLASSIFIER,
        },
        networks={"generator": _rgb32_gan, "classifier": _vgg},
        image_shape=_RGB,
    ),
}

# ImageNet: one fixed class per dataset, torchvision's pretrained VGG19-BN as classifier.
for _dataset, _cls in IMAGENET_CLASSES.items():
    CONFIGS[("vae", _dataset)] = TigConfig(
        model="vae", dataset=_dataset,
        min_val=-5.22431135177612, max_val=5.62481260299683,
        weights={"generator": WEIGHTS / "imagenet/imagenet_vae_model_epoch_41.pth"},
        networks={"generator": _imagenet_vae, "classifier": _vgg19_bn_imagenet},
        image_shape=(3, 128, 128),
        classes=(_cls,),
    )
    CONFIGS[("gan", _dataset)] = TigConfig(
        model="gan", dataset=_dataset,
        min_val=-1.99990940093994, max_val=1.99996650218964,
        networks={"generator": _biggan, "classifier": _vgg19_bn_imagenet},
        image_shape=(3, 224, 224),
        classes=(_cls,),
        prior_bound=2.0,  # truncated_noise_sample(truncation=1) in the legacy script
    )

_LEGACY = {
    ("sd", "mnist"): "mnist/sd/tig_sd.py",
    ("sd", "svhn"): "svhn/sd/tig_sd.py",
    ("sd", "cifar10"): "cifar10/sd/tig_sd.py",
    ("sd", "imagenet_pizza"): "imagenet/sd/tig_sd.py",
    ("sd", "imagenet_teddy"): "imagenet/sd/tig_sd.py",
}
for _key, _path in _LEGACY.items():
    CONFIGS[_key] = TigConfig(model=_key[0], dataset=_key[1], legacy_script=LEGACY / _path)


def get_config(model: str, dataset: str) -> TigConfig:
    try:
        return CONFIGS[(model, dataset)]
    except KeyError:
        raise ValueError(f"unknown combination: {model} x {dataset}") from None
