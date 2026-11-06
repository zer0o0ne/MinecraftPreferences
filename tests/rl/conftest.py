"""One intra-op torch thread per pytest-xdist worker."""

import torch

torch.set_num_threads(1)
