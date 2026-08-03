"""Run the real Trainer and dump the COMPLETE val curve, so the gate's behaviour
can be judged on the whole run rather than a truncated tail."""
import sys, os, json, time, io, contextlib, re
sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as np, torch
from data_processor import DataProcessor
from trainer import Trainer, _saturation_patience
from model import build_skeleton

class Clock:
    def __init__(s, sec): s.end = time.perf_counter() + sec
    def check(s): return s.end - time.perf_counter()

name, sec = sys.argv[2], float(sys.argv[3])
p = os.path.join('datasets', name)
d = {k: np.load(os.path.join(p, k+'.npy')) for k in ('train_x','train_y','valid_x','valid_y')}
meta = json.load(open(os.path.join(p, 'metadata')))
dp = DataProcessor(d['train_x'], d['train_y'], d['valid_x'], d['valid_y'], d['valid_x'], meta, Clock(sec))
tr, va, te = dp.process()
GENO={'Chesseract':(['conv3x3','avgpool3x3','conv3x3','conv3x3','none','conv1x1'],1),
      'Language':(['conv3x3','conv3x3','conv1x1','none','conv3x3','conv1x1'],2)}
g,dd=GENO[name]
torch.manual_seed(0)
m = build_skeleton(int(next(iter(tr))[0].shape[1]), int(meta['num_classes']), 32, 2, dd,
                   genotype=g, stem_stride=1)
t = Trainer(m, torch.device('cuda'), tr, va, meta, Clock(sec))
buf=io.StringIO()
with contextlib.redirect_stdout(buf): t.train()
log=buf.getvalue()
rows=re.findall(r'Epoch\s+(\d+)(?: m(\d+))? \| train=\s*[\d.]+% val=\s*([\d.]+)%', log)
m1=[float(v) for e,mem,v in rows if not mem]
print('%s: %d total epoch lines, member-1 had %d epochs, members created = %d'
      % (name, len(rows), len(m1), len(t.ensemble_states)))
best=-1; since=0; fired=None; maxfrac=0
for i,x in enumerate(m1):
    if x>best: best,since=x,0
    else: since+=1
    maxfrac=max(maxfrac, since/float(i+1))
    if fired is None and since>=_saturation_patience(i+1): fired=i+1
print('  member-1 best %.2f%% at epoch %d; longest drought %d; max drought fraction %.2f'
      % (max(m1), m1.index(max(m1))+1, since, maxfrac))
print('  NEW rule fires at: %s' % (('epoch %d'%fired) if fired else 'NEVER within this budget'))
for line in log.splitlines():
    if 'saturated after' in line or 'summary:' in line: print(' ', line.strip())
