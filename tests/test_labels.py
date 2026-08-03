"""Non-0-based labels must still train. Reproduces the old failure first.

usage: test_labels.py <submission_dir>
"""
import sys, os
sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as np, torch, torch.nn as nn
from data_processor import DataProcessor
from nas import NAS
from model import build_skeleton

FAILED=[]
def check(c,m):
    print(('  PASS  ' if c else '  FAIL  ')+m)
    if not c: FAILED.append(m)

class Clock:
    def check(self): return 3600.0

N, K = 400, 5
x = np.random.rand(N,3,8,8).astype(np.float32)

def build(labels, meta):
    dp = DataProcessor(x, labels, x, labels, x, meta, Clock())
    return dp.process(), meta

# ---- 1. the bug, reproduced: labels 1..K with a CORRECT num_classes=K --------
print('\n== labels 1..%d, metadata num_classes=%d (metadata is correct) ==' % (K,K))
y1 = np.random.randint(1, K+1, N)
meta = {'num_classes': K}
(tr,va,te), meta = build(y1, meta)
xb, yb = next(iter(tr))
old_width = K                       # what the pipeline used before the fix
m_old = build_skeleton(3, old_width, 32, 2, 1, genotype=['conv3x3']*6, stem_stride=1)
try:
    nn.CrossEntropyLoss()(m_old(xb), yb); check(False,'old head width should have raised')
except IndexError as e:
    check(True, 'old head (width=%d) raises IndexError - bug reproduced: %s' % (old_width, str(e)[:40]))

check(meta.get('n_outputs')==K+1, 'n_outputs widened to %s (expected %d)'%(meta.get('n_outputs'),K+1))
m_new = build_skeleton(3, int(meta['n_outputs']), 32, 2, 1, genotype=['conv3x3']*6, stem_stride=1)
try:
    loss = nn.CrossEntropyLoss()(m_new(xb), yb)
    check(torch.isfinite(loss), 'new head (width=%d) trains on 1-based labels'%meta['n_outputs'])
except Exception as e:
    check(False, 'new head trains on 1-based labels (raised %r)'%(e,))
check(meta['fallback_label'] in range(1,K+1), 'fallback_label %s is a real label value'%meta['fallback_label'])

# NAS must pick the widened value up end-to-end
model = NAS(tr, va, meta, Clock())._safe_fallback(3, int(meta.get('n_outputs')), 8, 8, xb, yb,
                                                  torch.device('cpu'))
check(model(xb).shape[1] == K+1, 'NAS fallback path also emits %d outputs'%(K+1))

# ---- 2. ordinary 0-based labels must be COMPLETELY unaffected ----------------
print('\n== labels 0..%d, metadata num_classes=%d (the normal case) ==' % (K-1,K))
y0 = np.random.randint(0, K, N)
meta0 = {'num_classes': K}
(tr0,_,_), meta0 = build(y0, meta0)
check(meta0.get('n_outputs')==K, 'n_outputs == num_classes == %d (no-op)'%K)

# ---- 3. non-contiguous labels ------------------------------------------------
print('\n== non-contiguous labels {0, 3, 9}, metadata num_classes=3 ==')
y2 = np.random.choice([0,3,9], N)
meta2 = {'num_classes': 3}
(tr2,_,_), meta2 = build(y2, meta2)
check(meta2.get('n_outputs')==10, 'n_outputs widened to %s (expected 10)'%meta2.get('n_outputs'))
xb2, yb2 = next(iter(tr2))
m2 = build_skeleton(3, int(meta2['n_outputs']), 32, 2, 1, genotype=['conv3x3']*6, stem_stride=1)
try:
    check(torch.isfinite(nn.CrossEntropyLoss()(m2(xb2), yb2)), 'trains on non-contiguous labels')
except Exception as e:
    check(False, 'trains on non-contiguous labels (raised %r)'%(e,))

print('\n'+('ALL PASSED' if not FAILED else 'FAILURES:\n  - '+'\n  - '.join(FAILED)))
sys.exit(1 if FAILED else 0)
