"""Are there unused speed levers? Measure a real mini-epoch through the REAL
DataLoader (so augmentation and host->device transfer are included, which
_measure_step_time deliberately excludes).

Arms: baseline / cudnn.benchmark / num_workers>0 / both.
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as np, torch, torch.nn as nn
from data_processor import DataProcessor
from model import build_skeleton, derive_macro
from trainer import autocast, GradScaler

def main():
  class Clock:
    def check(self): return 3600.0

  name = sys.argv[2]; NB = int(sys.argv[3])
  p = 'datasets/%s/' % name
  d = {k: np.load(p+k+'.npy') for k in ('train_x','train_y','valid_x','valid_y')}
  meta = json.load(open(p+'metadata'))
  dp = DataProcessor(d['train_x'], d['train_y'], d['valid_x'], d['valid_y'], d['valid_x'], meta, Clock())
  tr, va, te = dp.process()
  xb, _ = next(iter(tr))
  in_ch, H, W = int(xb.shape[1]), int(xb.shape[2]), int(xb.shape[3])
  c0, n, dd, stem = derive_macro(H, W)
  dev = torch.device('cuda')
  GENO = ['conv3x3','conv1x1','conv3x3','skip','conv3x3','conv1x1']

  def mini_epoch(loader, nb):
    torch.manual_seed(0)
    m = build_skeleton(in_ch, int(meta['n_outputs']), c0, n, dd, genotype=GENO, stem_stride=stem).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss(); scaler = GradScaler(enabled=True)
    m.train()
    it = iter(loader)
    for _ in range(3):                     # warm-up, untimed
        try: x,y = next(it)
        except StopIteration: it = iter(loader); x,y = next(it)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=True): loss = crit(m(x.to(dev)), y.to(dev))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize(); t0 = time.perf_counter(); seen = 0
    while seen < nb:
        try: x,y = next(it)
        except StopIteration: it = iter(loader); x,y = next(it)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=True): loss = crit(m(x.to(dev)), y.to(dev))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        seen += 1
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    del m, opt; torch.cuda.empty_cache()
    return el

  def rebuild(nw):
    return torch.utils.data.DataLoader(tr.dataset, batch_size=tr.batch_size,
                                       shuffle=True, drop_last=tr.drop_last,
                                       num_workers=nw, persistent_workers=(nw>0))

  print('%s: %dx%dx%d batch=%d, %d batches/epoch, timing %d batches per arm'
      % (name, in_ch, H, W, tr.batch_size, len(tr), NB))
  base = None
  for label, nw, bench in (('baseline (as shipped)',0,False),
                         ('cudnn.benchmark',0,True),
                         ('num_workers=4',4,False),
                         ('both',4,True)):
    torch.backends.cudnn.benchmark = bench
    ldr = tr if nw == 0 else rebuild(nw)
    el = mini_epoch(ldr, NB)
    if base is None: base = el
    print('  %-24s %7.2fs for %d batches   %5.2fs/epoch-equiv   %s'
          % (label, el, NB, el/NB*len(tr), '%.2fx' % (base/el)))


if __name__ == '__main__':
    main()
