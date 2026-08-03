"""Verification for the 2026-07-28 changes.

Each test REPRODUCES THE OLD BUG FIRST where that is possible, then shows the
fix - the habit CLAUDE.md 7g asks for, because every bug found in this project
so far was invisible in review and obvious in a run.

usage: test_changes.py <submission_dir>
"""
import sys, os, math
sys.path.insert(0, os.path.abspath(sys.argv[1]))

import numpy as np
import torch
import torch.nn as nn

FAILED = []


def check(cond, msg):
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    if not cond:
        FAILED.append(msg)


# ---------------------------------------------------------------- macro sizing
print('\n== derive_macro / stem stride ==')
from model import derive_macro, build_skeleton

EXPECT = {   # (h, w): (d, stem_stride, final_map)
    (8, 8): (1, 1, 4), (9, 9): (1, 1, 4), (10, 10): (1, 1, 5),
    (24, 24): (2, 1, 6), (28, 28): (2, 1, 7), (32, 32): (3, 1, 4),
    (64, 64): (3, 2, 4), (128, 128): (3, 4, 4), (256, 256): (3, 8, 4),
}
for (h, w), (ed, es, efm) in EXPECT.items():
    c0, n, d, stem = derive_macro(h, w)
    fm = h // (stem * 2 ** d)
    check((d, stem, fm) == (ed, es, efm),
          '%3dx%-3d -> d=%d stem=%d final=%dx%d (expected d=%d stem=%d final=%d)'
          % (h, w, d, stem, fm, fm, ed, es, efm))

# small inputs must be BYTE-IDENTICAL to the old behaviour (stem_stride == 1),
# otherwise this change could regress the datasets it was never meant to touch
print('\n== small inputs unaffected (stem_stride must be 1) ==')
for hw in (8, 9, 10, 20, 24, 27, 28, 32):
    _, _, _, stem = derive_macro(hw, hw)
    check(stem == 1, 'input %dx%d keeps stem_stride=1' % (hw, hw))

print('\n== stem downsamples without aliasing (only stride-2 convs) ==')
for hw, exp_stride_convs in ((128, 2), (64, 1), (28, 0)):
    _, _, _, stem = derive_macro(hw, hw)
    m = build_skeleton(3, 5, 32, 2, 3, genotype=['conv3x3'] * 6, stem_stride=stem)
    strided = [l for l in m.stem if isinstance(l, nn.Conv2d) and l.stride == (2, 2)]
    big = [l for l in m.stem if isinstance(l, nn.Conv2d) and max(l.stride) > 2]
    check(len(strided) == exp_stride_convs and not big,
          '%dx%d stem: %d stride-2 convs, no stride>2 kernel (aliasing)' % (hw, hw, len(strided)))

print('\n== forward shape + cost reduction ==')
x128 = torch.randn(2, 3, 128, 128)
c0, n, d, stem = derive_macro(128, 128)
new = build_skeleton(3, 7, c0, n, d, genotype=['conv3x3'] * 6, stem_stride=stem)
old = build_skeleton(3, 7, c0, n, d, genotype=['conv3x3'] * 6, stem_stride=1)
check(tuple(new(x128).shape) == (2, 7), 'new 128x128 model outputs [2, num_classes]')
p_new = sum(p.numel() for p in new.parameters())
p_old = sum(p.numel() for p in old.parameters())
check(p_new < p_old * 1.05, 'params not inflated (%d vs %d)' % (p_new, p_old))


def conv_flops(model, x):
    """Measured, not analytic: hook every conv and sum its real output cost."""
    tot = [0]

    def hook(mod, inp, out):
        tot[0] += (mod.in_channels * mod.out_channels
                   * mod.kernel_size[0] * mod.kernel_size[1]
                   * out.shape[2] * out.shape[3])
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return tot[0] / x.shape[0]


f_old, f_new = conv_flops(old, x128), conv_flops(new, x128)
check(f_new * 8 < f_old, '128x128 conv cost cut >8x (%.0f -> %.0f MFLOPs/sample, %.1fx)'
      % (f_old / 1e6, f_new / 1e6, f_old / f_new))

x28 = torch.randn(2, 3, 28, 28)
c0b, nb, db, stemb = derive_macro(28, 28)
m28 = build_skeleton(3, 7, c0b, nb, db, genotype=['conv3x3'] * 6, stem_stride=stemb)
m28o = build_skeleton(3, 7, c0b, nb, db, genotype=['conv3x3'] * 6, stem_stride=1)
check(abs(conv_flops(m28, x28) - conv_flops(m28o, x28)) < 1,
      '28x28 cost is unchanged (regression guard for the untouched datasets)')

# ------------------------------------------------------------- _to_4d_float
print('\n== _to_4d_float: layouts that used to escape as -10 ==')
from data_processor import _to_4d_float

neg = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)[:, :, ::-1, :]
check(neg.strides[2] < 0, 'constructed array really does have a negative stride')
try:
    torch.as_tensor(neg)
    old_raised = False
except Exception:
    old_raised = True
check(old_raised, 'old path (bare torch.as_tensor) raises on it - bug reproduced')
try:
    t = _to_4d_float(neg)
    check(tuple(t.shape) == (2, 3, 4, 4), 'negative-stride array now converts')
except Exception as e:
    check(False, 'negative-stride array now converts (raised %r)' % (e,))

for label, arr in (('float64', np.zeros((4, 3, 5, 5), dtype=np.float64)),
                   ('uint8', np.zeros((4, 3, 5, 5), dtype=np.uint8)),
                   ('big-endian', np.zeros((4, 3, 5, 5), dtype='>f4')),
                   ('3-D [N,H,W]', np.zeros((4, 5, 5), dtype=np.float32)),
                   ('2-D [N,F]', np.zeros((4, 20), dtype=np.float32))):
    try:
        t = _to_4d_float(arr)
        check(t.dim() == 4 and t.dtype == torch.float32, '%s -> 4-D float32 %s' % (label, tuple(t.shape)))
    except Exception as e:
        check(False, '%s -> 4-D float32 (raised %r)' % (label, e))

# ------------------------------------------------------- batch size / memory
print('\n== _choose_batch_size ==')
from data_processor import DataProcessor


class FakeClock:
    def check(self):
        return 3600.0


dp = DataProcessor(None, None, None, None, None, {}, FakeClock())
check(dp._choose_batch_size(1, 9, 9, 2) >= 2, 'n_train=2 never yields batch_size 1')
check(dp._choose_batch_size(1, 9, 9, 3) >= 2, 'n_train=3 never yields batch_size 1')

if torch.cuda.is_available():
    clean = dp._choose_batch_size(1, 9, 9, 50000)

    def burn():
        """Stand-in for a memory-heavy previous dataset. batch 64 at 128x128 with
        stem_stride=1 is what the real Myofibre run looked like; batch 32 leaves
        several GB driver-free and does NOT reproduce the collapse."""
        dev = torch.device('cuda')
        m = build_skeleton(3, 2, 32, 2, 3, genotype=['conv3x3'] * 6, stem_stride=1).to(dev)
        o = torch.optim.SGD(m.parameters(), lr=.01, momentum=.9)
        xx = torch.randn(64, 3, 128, 128, device=dev)
        yy = torch.randint(0, 2, (64,), device=dev)
        for _ in range(3):
            o.zero_grad(set_to_none=True)
            nn.CrossEntropyLoss()(m(xx), yy).backward()
            o.step()
        torch.cuda.synchronize()

    burn()
    resid = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    # ORDER MATTERS: the old formula must be evaluated BEFORE the new one, because
    # the new _choose_batch_size calls empty_cache() and so destroys the very state
    # the bug needs. Measuring it afterwards made this test pass the old formula
    # 512 and "fail" against correct code - a faulty test, not a faulty fix.
    free_only, _ = torch.cuda.mem_get_info()
    old_bs = max(4, min(int(free_only * 0.2 / (81 * 4 * 40)), 512))
    print('     (driver-free %.0f MB, reserved-but-free %.0f MB)'
          % (free_only / 1024 ** 2, resid / 1024 ** 2))
    check(old_bs < clean, 'old formula collapses here (%d -> %d) - bug reproduced'
          % (clean, old_bs))
    after = dp._choose_batch_size(1, 9, 9, 50000)
    check(after == clean, 'new formula survives the previous dataset (%d -> %d)' % (clean, after))
    torch.cuda.empty_cache()
else:
    print('  SKIP  GPU tests (no CUDA)')

# ------------------------------------------------------------- process() guard
print('\n== process() never raises ==')


n = 40
good_x = np.random.rand(n, 3, 8, 8).astype(np.float32)
good_y = np.random.randint(0, 3, n)

# What the wrapper is actually FOR: a failure in one of the optional steps
# (stats, cardinality probe, memory query, augmentation build) must degrade to
# minimal loaders rather than fail the dataset.
for name in ('_estimate_value_cardinality', '_choose_batch_size', '_build_train_transform'):
    dpx = DataProcessor(good_x, good_y, good_x, good_y, good_x, {'num_classes': 3}, FakeClock())

    def boom(*a, **k):
        raise RuntimeError('injected failure in ' + name)
    setattr(dpx, name, boom)
    try:
        tr, va, te = dpx.process()
        xb = next(iter(tr))[0]
        check(xb.dim() == 4, '%s raising -> minimal loaders returned, not a -10' % name)
    except Exception as e:
        check(False, '%s raising -> minimal loaders returned (raised %r)' % (name, e))

# DOCUMENTED LIMIT (see _minimal_process): an array that cannot become a tensor
# at all cannot yield a loader by any route, and both rungs call _to_4d_float.
# Asserted explicitly so nobody later "fixes" this into a false sense of safety.
class Boom:
    def __array__(self, *a, **k):
        raise ValueError('unconvertible')


try:
    DataProcessor(Boom(), good_y, good_x, good_y, good_x, {'num_classes': 3},
                  FakeClock()).process()
    check(False, 'truly unconvertible train_x is expected to raise (documented limit)')
except Exception:
    check(True, 'truly unconvertible train_x raises - documented limit, not a regression')

try:
    dp3 = DataProcessor(good_x, good_y, good_x, good_y, good_x, {'num_classes': 3}, FakeClock())
    tr, va, te = dp3.process()
    xb, yb = next(iter(tr))
    check(xb.dim() == 4 and not isinstance(te.sampler, torch.utils.data.RandomSampler)
          and not te.drop_last, 'normal path still returns valid, harness-legal loaders')
except Exception as e:
    check(False, 'normal path still works (raised %r)' % (e,))

print('\n' + ('ALL PASSED' if not FAILED else 'FAILURES:\n  - ' + '\n  - '.join(FAILED)))
sys.exit(1 if FAILED else 0)
