"""Uniform view over generative models for the GA.

Every generator exposes ``latent_dim`` and ``decode(z, labels)``, returning
images shaped as the classifier expects. Generators that can embed a real
image (VAEs) also implement ``encode``; the others draw seeds from the prior.
"""

import torch


class VAEGenerator:
    conditional = False

    def __init__(self, vae, image_shape):
        self.vae = vae
        self.image_shape = image_shape
        self.latent_dim = vae.z_dim

    @torch.no_grad()
    def encode(self, images):
        mu, _ = self.vae.encode(images)
        return mu

    @torch.no_grad()
    def decode(self, z, labels=None):
        return self.vae.decode(z).view(-1, *self.image_shape)


class ConditionalGANGenerator:
    """The class label is part of the generator input, so it is fixed while the GA mutates ``z``."""

    conditional = True

    def __init__(self, generator, latent_dim):
        self.generator = generator
        self.latent_dim = latent_dim

    @torch.no_grad()
    def decode(self, z, labels):
        return self.generator(z, labels)


def can_encode(generator) -> bool:
    return hasattr(generator, "encode")


def _load(config, name, device) -> torch.nn.Module:
    path = config.weights[name]
    if not path.exists():
        raise FileNotFoundError(f"missing {name} weights for {config.model} x {config.dataset}: {path} (see README)")
    net = config.networks[name]()
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return net.to(device).eval()


def build(config, device):
    """Return ``(generator, classifier)`` for a ported combination."""
    if not config.ported:
        raise NotImplementedError(
            f"{config.model} x {config.dataset} is not ported yet; legacy script: {config.legacy_script}"
        )
    net = _load(config, "generator", device)
    classifier = _load(config, "classifier", device)
    if config.model == "vae":
        return VAEGenerator(net, config.image_shape), classifier
    if config.model == "gan":
        return ConditionalGANGenerator(net, latent_dim=net.nz), classifier
    raise NotImplementedError(f"no wrapper for model {config.model}")
