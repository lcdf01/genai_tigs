"""Regression tests on the real checkpoints shipped in the repo.

Each golden file pins today's output; a refactor that changes it fails here.
After an intentional behaviour change, regenerate them with:

    TIG_UPDATE_GOLDEN=1 python -m pytest tests/test_regression.py
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tig.config import get_config
from tig.ga import GAParams, run_ga
from tig.generators import build, can_encode

GOLDEN_DIR = Path(__file__).parent / "golden"
# (model, dataset, mutation level with a documented perturbation)
CASES = [
    ("vae", "mnist", "high"), ("gan", "mnist", "low"),
    # These need the Google Drive weights in weights/ and are skipped without them.
    ("vae", "svhn", "high"), ("gan", "svhn", "high"), ("vae", "cifar10", "high"), ("gan", "cifar10", "low"),
]


def _start(generator, classifier):
    """A fixed, correctly classified starting point; no dataset download needed."""
    z = torch.randn(1, generator.latent_dim, generator=torch.Generator().manual_seed(0))
    if can_encode(generator):
        z0 = generator.encode(generator.decode(z))
        return z0, int(classifier(generator.decode(z0)).argmax())
    for label in range(10):  # conditional: first class the classifier recognises
        if int(classifier(generator.decode(z, torch.tensor([label]))).argmax()) == label:
            return z, label
    raise AssertionError("no class is recognised from this latent")


@pytest.mark.parametrize("model,dataset,mutation", CASES, ids=[f"{m}-{d}" for m, d, _ in CASES])
def test_matches_golden(model, dataset, mutation):
    config = get_config(model, dataset)
    if not all(p.exists() for p in config.weights.values()):
        pytest.skip("weights not available")

    generator, classifier = build(config, torch.device("cpu"))
    z0, label = _start(generator, classifier)
    params = GAParams(
        generations=30, perturbation=config.perturbation(mutation), min_val=config.min_val, max_val=config.max_val
    )
    result = run_ga(generator, classifier, z0, label, params, seed=0)

    golden_path = GOLDEN_DIR / f"{dataset}_{model}_seed0.npz"
    if os.environ.get("TIG_UPDATE_GOLDEN"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        np.savez_compressed(
            golden_path, label=label, image=result.image.numpy(), fitness=result.fitness,
            predicted=result.predicted_label, generations=result.generations,
        )
        pytest.skip("golden file updated")

    golden = np.load(golden_path)
    assert label == golden["label"]
    assert result.predicted_label == golden["predicted"]
    assert result.generations == golden["generations"]
    assert result.fitness == pytest.approx(float(golden["fitness"]), abs=1e-4)
    np.testing.assert_allclose(result.image.numpy(), golden["image"], atol=1e-4)
