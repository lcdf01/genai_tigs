import csv

import pytest
import torch

from tig import cli
from tig.ga import GAParams
from tig import seeds
from tig.runner import run


def test_unported_combination_exits_with_legacy_hint(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--model", "gan", "--dataset", "svhn"])
    assert exc.value.code == 2
    assert "tig_cdcgan_svhn.py" in capsys.readouterr().err


def test_undocumented_mutation_level_requires_override(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--model", "vae", "--dataset", "mnist", "--mutation", "low"])
    assert "--perturbation" in capsys.readouterr().err


def test_unknown_dataset_rejected():
    with pytest.raises(SystemExit):
        cli.main(["--model", "vae", "--dataset", "fashion"])


def test_runner_writes_summary_and_skips_misclassified(toy, tmp_path):
    gen, clf = toy
    images = [gen.decode(torch.full((1, gen.latent_dim), v)).squeeze(0) for v in (-1.0, 0.0, 1.0)]
    labels = [int(clf(img.unsqueeze(0)).argmax()) for img in images]
    labels[0] = (labels[0] + 1) % 3  # force one wrong label so it is skipped
    test_set = list(zip(images, labels))

    params = GAParams(generations=50, pop_size=10, best_left=4, perturbation=0.5)
    seed_inputs = seeds.from_test_set(gen, test_set, num=3, seed=0, device="cpu")
    rows = run(gen, clf, seed_inputs, params, seed=0, out_dir=tmp_path)

    assert len(rows) == 3
    assert sum(r["status"] == "skipped" for r in rows) == 1
    with open(tmp_path / "summary.csv") as f:
        assert len(list(csv.DictReader(f))) == 3
    assert len(list(tmp_path.glob("*_original.npy"))) == 2


def test_missing_weights_exit_cleanly(capsys):
    from tig.config import get_config

    if get_config("vae", "svhn").weights["generator"].exists():
        pytest.skip("SVHN weights are present")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--model", "vae", "--dataset", "svhn"])
    assert exc.value.code == 1
    assert "missing generator weights" in capsys.readouterr().err
