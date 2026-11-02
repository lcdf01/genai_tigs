import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every global RNG used by the generators so a run can be replayed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
