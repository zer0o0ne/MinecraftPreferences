"""One intra-op torch thread per pytest-xdist worker (as in tests/training/pretrain/conftest.py)."""

import torch

torch.set_num_threads(1)
