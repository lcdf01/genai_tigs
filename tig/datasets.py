from pathlib import Path


def load_test_set(dataset: str, root: Path):
    """Torchvision test split as [0, 1] tensors; downloaded into ``root`` on first use."""
    import torchvision
    from torchvision import transforms

    to_tensor = transforms.ToTensor()
    if dataset == "mnist":
        return torchvision.datasets.MNIST(root=str(root), train=False, transform=to_tensor, download=True)
    if dataset == "svhn":
        return torchvision.datasets.SVHN(root=str(root), split="test", transform=to_tensor, download=True)
    if dataset == "cifar10":
        return torchvision.datasets.CIFAR10(root=str(root), train=False, transform=to_tensor, download=True)
    raise NotImplementedError(f"test set loader for {dataset} not ported yet")
