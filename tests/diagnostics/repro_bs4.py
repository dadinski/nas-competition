"""Reproduce the bs=4 pathology: _choose_batch_size reads driver-free GPU
memory, but the previous dataset's caching-allocator pool is still reserved,
so it sees ~0 free and floors at 4.

Mimics main.py's structure: run_submission() is a function whose locals die on
return, but the allocator pool survives.
"""
import sys, os
sys.path.insert(0, os.path.abspath(sys.argv[1]))

import numpy as np
import torch

from data_processor import DataProcessor
from model import build_skeleton


class FakeClock:
    def check(self):
        return 3600.0


def report(tag):
    free, total = torch.cuda.mem_get_info()
    print("%-28s driver_free=%7.0f MB  reserved=%7.0f MB  allocated=%7.0f MB"
          % (tag, free / 1024**2, torch.cuda.memory_reserved() / 1024**2,
             torch.cuda.memory_allocated() / 1024**2))
    return free / 1024**2


def choose_for(shape, n_train=50000):
    """Call the real _choose_batch_size on a dummy processor."""
    c, h, w = shape
    dp = DataProcessor(None, None, None, None, None, {}, FakeClock())
    return dp._choose_batch_size(c, h, w, n_train)


def simulate_big_dataset():
    """Stand-in for the Myopia run: big model at 128x128, a few real steps."""
    dev = torch.device('cuda')
    m = build_skeleton(3, 2, 32, 2, 3, genotype=['conv3x3'] * 6).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(64, 3, 128, 128, device=dev)
    y = torch.randint(0, 2, (64,), device=dev)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.CrossEntropyLoss()(m(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    # locals die when this function returns, exactly like run_submission's do


print("=== Sokoto (1x9x9) batch size in different memory states ===")
report("clean process")
print("  bs =", choose_for((1, 9, 9)))

simulate_big_dataset()
report("after big dataset returns")
print("  bs =", choose_for((1, 9, 9)), "   <-- what dataset N+1 actually gets")

torch.cuda.empty_cache()
report("after empty_cache()")
print("  bs =", choose_for((1, 9, 9)), "   <-- with the proposed fix")
