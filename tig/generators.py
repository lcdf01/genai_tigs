"""Uniform view over generative models for the GA.

Every generator exposes ``latent_dim`` and ``decode(z, labels)``, returning
images shaped as the classifier expects. Generators that can embed a real
image (VAEs) also implement ``encode``; the others draw seeds from the prior.
"""

import torch
import torch.nn.functional as F

# torchvision ImageNet normalisation, as used by the pretrained VGG19-BN.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class BigGANGenerator:
    """BigGAN conditioned on a one-hot ImageNet class; output prepared for VGG19-BN."""

    conditional = True
    latent_dim = 128

    def __init__(self, biggan, truncation=1.0, num_classes=1000):
        self.biggan = biggan
        self.truncation = truncation
        self.num_classes = num_classes

    @torch.no_grad()
    def decode(self, z, labels):
        one_hot = F.one_hot(labels, self.num_classes).float()
        return to_vgg_input(self.biggan(z, one_hot, self.truncation))


def to_vgg_input(images):
    """Legacy preprocessing of a [-1, 1] 256x256 batch: 8-bit quantisation (via PIL in the
    original), centre crop to 224 and ImageNet normalisation."""
    pixels = ((images + 1) / 2 * 255).to(torch.uint8).float() / 255
    top = (pixels.shape[-2] - 224) // 2
    left = (pixels.shape[-1] - 224) // 2
    crop = pixels[..., top:top + 224, left:left + 224]
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (crop - mean) / std


def can_encode(generator) -> bool:
    return hasattr(generator, "encode")


def _load(config, name, device) -> torch.nn.Module:
    path = config.weights.get(name)
    if path is None:  # pretrained network that downloads its own weights
        return config.networks[name]().to(device).eval()
    if not path.exists():
        raise FileNotFoundError(f"missing {name} weights for {config.model} x {config.dataset}: {path} (see README)")
    net = config.networks[name]()
    state = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(state.get("model_state_dict", state))  # training checkpoints wrap the weights
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
    if config.model == "gan" and config.dataset.startswith("imagenet"):
        return BigGANGenerator(net), classifier
    if config.model == "gan":
        return ConditionalGANGenerator(net, latent_dim=net.nz), classifier
    raise NotImplementedError(f"no wrapper for model {config.model}")
