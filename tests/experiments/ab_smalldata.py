"""Does ensembling help in the SMALL-DATA regime the competition actually uses?

The organiser's test datasets are ~4,500-5,300 training samples at a 6-minute
budget and memorise completely (100% train). Ganges is a Gutenberg variant -
identical 1x27x18 shape and 6 classes - so subsampling Gutenberg to 4,500 train
/ 1,500 valid gives a near-replica we own the labels for.

Arms differ ONLY in PATIENCE_FRACTION. The genotype and macro are FIXED across
arms, because NAS search variance is worth +-2 adj on a single run (conway,
CLAUDE.md 7m) and would swamp the effect being measured.

Scored on the real 6,000-sample test split.

usage: ab_smalldata.py <submission_dir> <seconds> <seed> <frac> [frac...]
"""
import sys, os, json, time, io, contextlib, re
sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as np, torch
import trainer as T
from data_processor import DataProcessor
from model import build_skeleton, derive_macro

SEC  = float(sys.argv[2])
SEED = int(sys.argv[3])
FRACS = [float(f) for f in sys.argv[4:]]

GENO = ['conv3x3', 'avgpool3x3', 'conv3x3', 'skip', 'none', 'conv3x3']   # fixed
N_TRAIN, N_VALID = 4500, 1500      # Ganges-scale (Gutenberg / 10)

class Clock:
    def __init__(s, sec): s.end = time.perf_counter() + sec
    def check(s): return s.end - time.perf_counter()

p = 'datasets/Gutenberg/'
rng = np.random.RandomState(0)                      # same subsample for every arm
tr_i = rng.choice(45000, N_TRAIN, replace=False)
va_i = rng.choice(15000, N_VALID, replace=False)
train_x = np.load(p+'train_x.npy')[tr_i]; train_y = np.load(p+'train_y.npy')[tr_i]
valid_x = np.load(p+'valid_x.npy')[va_i]; valid_y = np.load(p+'valid_y.npy')[va_i]
test_x  = np.load(p+'test_x.npy');        test_y  = np.load(p+'test_y.npy')
BENCH = 40.98

print('# Ganges replica: %d train / %d valid / %d test, 1x27x18, 6 classes, %.0fs budget, seed %d'
      % (N_TRAIN, N_VALID, len(test_y), SEC, SEED))
print('# genotype FIXED across arms: %s' % GENO)
print('%-8s %8s %10s %10s %10s %9s' % ('frac', 'members', 'best_val', 'test_acc', 'adj_score', 'epochs'))

for frac in FRACS:
    T.PATIENCE_FRACTION = frac                      # the only thing that varies
    meta = {'num_classes': 6, 'benchmark': BENCH, 'codename': 'GangesReplica'}
    dp = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, meta, Clock(SEC))
    tr, va, te = dp.process()
    in_ch = int(next(iter(tr))[0].shape[1])
    c0, n, d, stem = derive_macro(27, 18)
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = build_skeleton(in_ch, int(meta['n_outputs']), c0, n, d,
                           genotype=GENO, stem_stride=stem)
    t = T.Trainer(model, torch.device('cuda'), tr, va, meta, Clock(SEC))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t.train()
        preds = t.predict(te)
    log = buf.getvalue()
    acc = 100.0 * float((np.asarray(preds) == test_y).mean())
    adj = (acc - BENCH) * 10.0 / (100.0 - BENCH)
    m = re.search(r'summary: (\d+) epochs, best val ([\d.]+)%', log)
    eps, bv = (m.group(1), float(m.group(2))) if m else ('?', float('nan'))
    print('%-8.2f %8d %9.2f%% %9.2f%% %+10.3f %9s'
          % (frac, len(t.ensemble_states), bv, acc, adj, eps), flush=True)
