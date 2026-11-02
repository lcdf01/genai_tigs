"""Regression test on the real MNIST VAE + classifier weights shipped in the repo.

The golden file pins today's output; a refactor that changes it fails here.
After an intentional behaviour change, regenerate it with:

    TIG_UPDATE_GOLDEN=1 python -m pytest tests/test_regression_mnist_vae.py
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tig.config import get_config
from tig.ga import GAParams, run_ga

GOLDEN = Path(__file__).parent / "golden" / "mnist_vae_seed0.npz"
CONFIG = get_config("vae", "mnist")

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in CONFIG.weights.values()), reason="MNIST weights not available"
)


def _run():
    from tig.generators import build

    gen, clf = build(CONFIG, torch.device("cpu"))
    # A fixed starting latent: no dataset download needed.
    z0 = gen.encode(gen.decode(torch.randn(1, gen.latent_dim, generator=torch.Generator().manual_seed(0))))
    label = int(clf(gen.decode(z0)).argmax())
    params = GAParams(
        generations=30, perturbation=CONFIG.perturbation["high"], min_val=CONFIG.min_val, max_val=CONFIG.max_val
    )
    return label, run_ga(gen, clf, z0, label, params, seed=0)


def test_mnist_vae_matches_golden():
    label, result = _run()
    if os.environ.get("TIG_UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            GOLDEN, label=label, image=result.image.numpy(), fitness=result.fitness,
            predicted=result.predicted_label, generations=result.generations,
        )
        pytest.skip("golden file updated")

    golden = np.load(GOLDEN)
    assert label == golden["label"]
    assert result.predicted_label == golden["predicted"]
    assert result.generations == golden["generations"]
    assert result.fitness == pytest.approx(float(golden["fitness"]), abs=1e-4)
    np.testing.assert_allclose(result.image.numpy(), golden["image"], atol=1e-4)
