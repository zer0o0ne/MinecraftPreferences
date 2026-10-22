"""One intra-op torch thread per pytest-xdist worker (as in tests/pretrain/conftest.py); the
approved action masks come from tests/conftest.py (`masks`)."""

import torch

torch.set_num_threads(1)
