import csv
from pathlib import Path

import numpy as np
import torch

from tig.ga import GAParams, run_ga


def load_test_set(dataset: str, root: Path):
    import torchvision
    from torchvision import transforms

    if dataset != "mnist":
        raise NotImplementedError(f"test set loader for {dataset} not ported yet")
    return torchvision.datasets.MNIST(root=str(root), train=False, transform=transforms.ToTensor(), download=True)


def run(generator, classifier, test_set, params: GAParams, num_images: int, seed: int, out_dir: Path, device):
    """Draw ``num_images`` distinct test inputs and run the GA on each one the
    classifier gets right. Returns the rows written to ``summary.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_set), size=min(num_images, len(test_set)), replace=False)
    rows = []

    for n, idx in enumerate(indices):
        image, label = test_set[int(idx)]
        z0 = generator.encode(image.unsqueeze(0).to(device))
        with torch.no_grad():
            reconstructed = generator.decode(z0)
            original_pred = int(classifier(reconstructed).argmax(dim=1).item())
        row = {"index": int(idx), "label": int(label), "original_pred": original_pred}

        if original_pred != label:
            # The reconstruction is already misclassified: not a valid seed input.
            rows.append({**row, "status": "skipped"})
            continue

        result = run_ga(generator, classifier, z0, int(label), params, seed=seed + n)
        name = f"image_{n}_X{label}_Y{result.predicted_label}"
        np.save(out_dir / f"{name}_original.npy", reconstructed.squeeze(0).cpu().numpy())
        np.save(out_dir / f"{name}.npy", result.image.numpy())
        rows.append({
            **row,
            "status": "misbehaviour" if result.misbehaviour else "not_found",
            "predicted": result.predicted_label,
            "fitness": result.fitness,
            "generations": result.generations,
        })

    fields = ["index", "label", "original_pred", "status", "predicted", "fitness", "generations"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows
