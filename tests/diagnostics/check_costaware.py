"""Quick check (no training): on GeoClassing's shape, what does the OLD
proxy-only fallback pick, and what does the NEW cost-aware pick choose?

Reports model size, measured step time, and the epochs each would afford.
"""
import sys, os, time, random
sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as np, torch
from data_processor import DataProcessor
from model import build_skeleton, derive_macro, all_genotypes, is_degenerate
from proxies import naswot_score, synflow_score
from nas import NAS

class Clock:
    def __init__(s, sec): s.end = time.perf_counter() + sec
    def check(s): return s.end - time.perf_counter()

# real GeoClassing loaders, so batch size and shapes match production
p = 'datasets/GeoClassing/'
d_ = {k: np.load(p+k+'.npy') for k in ('train_x','train_y','valid_x','valid_y')}
import json; meta = json.load(open(p+'metadata'))
dp = DataProcessor(d_['train_x'], d_['train_y'], d_['valid_x'], d_['valid_y'], d_['valid_x'], meta, Clock(3600))
tr, va, te = dp.process()
xb, yb = next(iter(tr))
in_ch, H, W = int(xb.shape[1]), int(xb.shape[2]), int(xb.shape[3])
ncls = int(meta['n_outputs'])
c0, n, d, stem = derive_macro(H, W)
dev = torch.device('cuda')
print('GeoClassing: %dx%dx%d, batch %d, macro c0=%d n=%d d=%d stem=%d'
      % (in_ch, H, W, xb.shape[0], c0, n, d, stem))

# reproduce the scoring phase
random.seed(0); torch.manual_seed(0)
gs = [g for g in all_genotypes() if not is_degenerate(g)]
random.shuffle(gs); gs = gs[:250]
xb_s = xb[:min(32, xb.shape[0])].to(dev)
scored = []
for g in gs:
    try:
        m = build_skeleton(in_ch, ncls, c0, n, d, genotype=g, stem_stride=stem).to(dev)
        scored.append((naswot_score(m, xb_s), synflow_score(m, in_ch, H, W, dev), g))
        del m; torch.cuda.empty_cache()
    except RuntimeError:
        torch.cuda.empty_cache()
byn = sorted(range(len(scored)), key=lambda i: -scored[i][0])
bys = sorted(range(len(scored)), key=lambda i: -scored[i][1])
rn = {j:r for r,j in enumerate(byn)}; rs = {j:r for r,j in enumerate(bys)}
order = sorted(range(len(scored)), key=lambda i: rn[i]+rs[i])
print('scored %d genotypes' % len(scored))

nas = NAS(tr, va, meta, Clock(3600))
def stats(geno):
    m = build_skeleton(in_ch, ncls, c0, n, d, genotype=geno, stem_stride=stem)
    par = sum(q.numel() for q in m.parameters())
    t = nas._measure_step_time(m, xb, yb, dev, n_iters=3)
    del m; torch.cuda.empty_cache()
    return par, t

old_geno = scored[order[0]][2]
new_geno = nas._cost_aware_pick(scored, order, in_ch, ncls, c0, n, d, stem, dev,
                                deadline=time.perf_counter()+600)

print()
print('NOTE: _measure_step_time excludes the DataLoader and AMP (CLAUDE.md 4), so these')
print('      are NOT wall-clock projections. Only the RATIO between rows is meaningful.')
print()
print('%-26s %10s %12s %10s' % ('', 'params', 's/step', 'rel cost'))
base = None
for label, g in (('OLD  proxy leader', old_geno), ('NEW  cost-aware pick', new_geno)):
    par, t = stats(g)
    if base is None: base = t
    print('%-26s %10d %11.3fs %9.2fx' % (label, par, t, t / base))
print()
print('same genotype chosen?', old_geno == new_geno)
