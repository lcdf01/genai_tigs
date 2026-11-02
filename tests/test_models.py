"""The networks in tig/models must be drop-in replacements for the legacy ones:
same checkpoint keys and shapes, and the same outputs for the same weights.

This is what lets the SVHN/CIFAR-10 checkpoints from Google Drive load without
having them in the repo.
"""

import importlib.util

import pytest
import torch

from tig.config import EXPERIMENTS, get_config
from tig.models.classifiers import MnistClassifier, VGGNet
from tig.models.vae import MlpVAE, ResConvVAE, StridedConvVAE


def _legacy(rel_path, cls_name):
    spec = importlib.util.spec_from_file_location(f"legacy_{cls_name}", EXPERIMENTS / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, cls_name)


CASES = [
    ("mnist_experiments/mnist_vae/vae/model.py", "VAE", dict(img_size=784, h_dim=1600, z_dim=400),
     MlpVAE, dict(img_size=784, h_dim=1600, z_dim=400), (1, 784)),
    ("svhn_experiments/svhn_vae/vae/model.py", "ConvVAE", dict(img_size=(32, 32), c_num=3, h_dim=4000, z_dim=800),
     ResConvVAE, dict(img_size=(32, 32), c_num=3, h_dim=4000, z_dim=800), (1, 3, 32, 32)),
    ("cifar10_experiments/cifar10_vae/vae/vae_model.py", "ConvVAE", dict(c_num=3, h_dim=4000, z_dim=1024),
     StridedConvVAE, dict(c_num=3, h_dim=4000, z_dim=1024), (1, 3, 32, 32)),
]


@pytest.mark.parametrize("path,legacy_cls,legacy_kw,new_cls,new_kw,in_shape", CASES, ids=lambda v: getattr(v, "__name__", None))
def test_vae_matches_legacy(path, legacy_cls, legacy_kw, new_cls, new_kw, in_shape):
    torch.manual_seed(0)
    legacy = _legacy(path, legacy_cls)(**legacy_kw).eval()
    new = new_cls(**new_kw).eval()
    new.load_state_dict(legacy.state_dict())  # raises on any key/shape mismatch

    x = torch.rand(in_shape)
    with torch.no_grad():
        legacy_mu, _ = legacy.encode(x)
        new_mu, _ = new.encode(x)
        torch.testing.assert_close(new_mu, legacy_mu)
        torch.testing.assert_close(new.decode(new_mu), legacy.decode(legacy_mu))


@pytest.mark.parametrize("path,cls,new_cls,in_shape", [
    ("mnist_experiments/mnist_classifier/model_mnist.py", "MnistClassifier", MnistClassifier, (2, 1, 28, 28)),
    ("svhn_experiments/svhn_classifier/model_svhn.py", "VGGNet", VGGNet, (2, 3, 32, 32)),
])
def test_classifier_matches_legacy(path, cls, new_cls, in_shape):
    torch.manual_seed(0)
    legacy = _legacy(path, cls)().eval()
    new = new_cls().eval()
    new.load_state_dict(legacy.state_dict())
    x = torch.rand(in_shape)
    with torch.no_grad():
        torch.testing.assert_close(new(x), legacy(x))


def test_mnist_gan_loads_published_weights():
    from tig.generators import build

    config = get_config("gan", "mnist")
    generator, _ = build(config, torch.device("cpu"))
    z = torch.randn(3, generator.latent_dim)
    images = generator.decode(z, torch.tensor([0, 5, 9]))
    assert images.shape == (3, 1, 28, 28)
    assert images.min() >= -1 and images.max() <= 1  # tanh output
