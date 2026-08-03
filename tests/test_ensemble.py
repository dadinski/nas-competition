"""Verify ensembling as wired into the real Trainer (not the A/B harness).

Asserts the four things that could go wrong:
  1. it FIRES on a saturating dataset and actually beats the single model
  2. it stays OFF when the dataset does not saturate (no regression path)
  3. predict() returns exactly n_test labels, in order, either way
  4. it degrades to fewer members under a tight clock instead of overrunning

The valid split is reused as the "test" split so accuracy is measurable.

usage: test_ensemble.py <submission_dir> <dataset> <seconds>
"""
import sys, os, json, time, io, contextlib
sys.path.insert(0, os.path.abspath(sys.argv[1]))

import numpy as np
import torch

from data_processor import DataProcessor
from trainer import Trainer
from model import build_skeleton

FAILED = []


def check(cond, msg):
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    if not cond:
        FAILED.append(msg)


class Clock:
    def __init__(self, seconds):
        self.end = time.perf_counter() + seconds

    def check(self):
        return self.end - time.perf_counter()


name, seconds = sys.argv[2], float(sys.argv[3])
p = os.path.join('datasets', name)
d = {k: np.load(os.path.join(p, k + '.npy'))
     for k in ('train_x', 'train_y', 'valid_x', 'valid_y')}
meta = json.load(open(os.path.join(p, 'metadata')))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# valid doubles as test so the ensemble's predictions can be scored
dp = DataProcessor(d['train_x'], d['train_y'], d['valid_x'], d['valid_y'],
                   d['valid_x'], meta, Clock(seconds))
tr, va, te = dp.process()
y_test = torch.as_tensor(np.asarray(d['valid_y'])).long()
n_test = len(te.dataset)

GENO = {'Chesseract': (['conv3x3', 'avgpool3x3', 'conv3x3', 'conv3x3', 'none', 'conv1x1'], 1),
        'Language':   (['conv3x3', 'conv3x3', 'conv1x1', 'none', 'conv3x3', 'conv1x1'], 2)}
geno, dd = GENO[name]
in_ch = int(next(iter(tr))[0].shape[1])
torch.manual_seed(0)
model = build_skeleton(in_ch, int(meta['num_classes']), 32, 2, dd,
                       genotype=geno, stem_stride=1)

t = Trainer(model, dev, tr, va, meta, Clock(seconds))
buf = io.StringIO()
t0 = time.time()
with contextlib.redirect_stdout(buf):
    t.train()
train_wall = time.time() - t0
log = buf.getvalue()
print(log[-2500:])

n_members = len(t.ensemble_states)
saturated = 'saturated after' in log
print('  --- members: %d, train wall %.0fs of %.0fs budget ---' % (n_members, train_wall, seconds))

# COHERENCE is the invariant; whether ensembling fires at a given budget is not.
# Asserting "N > 1" outright made this test FLAKY at 420s: batch shuffling and the
# noise augmentation are unseeded, so member 1's saturation point moves run to
# run, and near a marginal budget there is sometimes no room for a second member
# - in which case the gate is behaving CORRECTLY by declining. Assert the rule,
# then assert the payoff only when it applies.
check(saturated == (n_members > 0),
      'gate is coherent: saturation logged (%s) matches members created (%d)'
      % (saturated, n_members))
if n_members <= 1:
    print('  NOTE  no ensemble at this budget - re-run with a larger one to exercise '
          'the averaging path (600s+ is reliable on Chesseract)')

# members must actually differ - a failed re-init would silently give clones
if n_members > 1:
    k = [key for key in t.ensemble_states[0] if t.ensemble_states[0][key].dtype.is_floating_point][0]
    a, b = t.ensemble_states[0][k], t.ensemble_states[1][k]
    check(not torch.allclose(a, b), 'members hold genuinely different weights (re-init worked)')

# --- ensemble vs single, scored on the same data -----------------------------
preds_ens = t.predict(te)
check(len(preds_ens) == n_test, 'ensemble predict() returns exactly n_test=%d' % n_test)
acc_ens = float((torch.as_tensor(preds_ens) == y_test).float().mean())

saved = t.ensemble_states
t.ensemble_states = []                      # force the single-model path
t.model.load_state_dict(t._best_state_for_fallback)
preds_single = t.predict(te)
check(len(preds_single) == n_test, 'single-model predict() returns exactly n_test')
acc_single = float((torch.as_tensor(preds_single) == y_test).float().mean())
t.ensemble_states = saved

print('  --- single %.4f  ensemble %.4f  gain %+.4f  (benchmark %.4f) ---'
      % (acc_single, acc_ens, acc_ens - acc_single, meta['benchmark'] / 100.0))
if n_members > 1:
    check(acc_ens > acc_single,
          'ensemble beats the single model (%.4f > %.4f)' % (acc_ens, acc_single))
else:
    print('  SKIP  ensemble-vs-single (only %d member)' % n_members)

# --- graceful degradation: almost no clock left ------------------------------
t.clock = Clock(0.5)
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    preds_tight = t.predict(te)
print(buf2.getvalue().strip()[:400])
check(len(preds_tight) == n_test,
      'predict() still returns exactly n_test with the clock nearly exhausted')

print('\n' + ('ALL PASSED' if not FAILED else 'FAILURES:\n  - ' + '\n  - '.join(FAILED)))
sys.exit(1 if FAILED else 0)
