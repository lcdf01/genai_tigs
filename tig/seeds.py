"""Where the GA starting points come from.

Each source yields ``(key, z0, label)``: ``key`` identifies the seed in the
results, ``z0`` has shape ``[1, latent_dim]`` and ``label`` is the class the
classifier is expected to predict.
"""

import numpy as np
import torch


def from_test_set(generator, test_set, num, seed, device):
    """Embed real test images (needs a generator with ``encode``, e.g. a VAE)."""
    rng = np.random.default_rng(seed)
    for idx in rng.choice(len(test_set), size=min(num, len(test_set)), replace=False):
        image, label = test_set[int(idx)]
        yield int(idx), generator.encode(image.unsqueeze(0).to(device)), int(label)


def from_prior(generator, num, seed, device, num_classes=10):
    """Sample ``z ~ N(0, I)`` and a random class (conditional generators)."""
    g = torch.Generator().manual_seed(seed)
    for n in range(num):
        z = torch.randn(1, generator.latent_dim, generator=g).to(device)
        label = int(torch.randint(num_classes, (1,), generator=g))
        yield n, z, label
