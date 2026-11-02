"""Generator wrappers: each exposes ``encode(images) -> z`` and ``decode(z) -> images``
with images shaped the way the dataset's classifier expects them."""

import importlib.util
from pathlib import Path

import torch

from tig.config import EXPERIMENTS, TigConfig


def _load_module(path: Path, name: str):
    """Import a legacy model file without turning ``experiments/`` into a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_weights(model: torch.nn.Module, path: Path, device) -> torch.nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"missing weights: {path}")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


class MnistVAE:
    image_shape = (1, 28, 28)

    def __init__(self, config: TigConfig, device):
        vae_mod = _load_module(EXPERIMENTS / "mnist_experiments/mnist_vae/vae/model.py", "mnist_vae_model")
        self.vae = _load_weights(vae_mod.VAE(img_size=28 * 28, h_dim=1600, z_dim=400), config.weights["vae"], device)
        self.latent_dim = 400

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        mu, _ = self.vae.encode(images.view(images.shape[0], -1))
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z).view(-1, *self.image_shape)


def load_mnist_classifier(config: TigConfig, device) -> torch.nn.Module:
    mod = _load_module(EXPERIMENTS / "mnist_experiments/mnist_classifier/model_mnist.py", "mnist_classifier_model")
    return _load_weights(mod.MnistClassifier(), config.weights["classifier"], device)


_BUILDERS = {
    ("vae", "mnist"): (MnistVAE, load_mnist_classifier),
}


def build(config: TigConfig, device):
    """Return ``(generator, classifier)`` for a ported combination."""
    if not config.ported:
        raise NotImplementedError(
            f"{config.model} x {config.dataset} is not ported to the unified framework yet; "
            f"legacy script: {config.legacy_script}"
        )
    make_generator, make_classifier = _BUILDERS[(config.model, config.dataset)]
    return make_generator(config, device), make_classifier(config, device)
