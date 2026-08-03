"""Diagnostic: for the datasets that saturate within a few epochs and then
waste 90% of the budget, is the model OVERfitting (train >> val -> need
regularisation) or UNDERfitting (train ~ val -> need capacity/optimisation)?

The production logs only print validation accuracy, so this cannot be answered
from them. Reproduces the exact macro + genotype the real run selected.

usage: diag_fit.py <submission_dir> <dataset> <epochs>
"""
import sys, os, time, json
sys.path.insert(0, os.path.abspath(sys.argv[1]))
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import torch
import torch.nn as nn

from data_processor import DataProcessor
from model import build_skeleton

# what the 27/07 run actually selected, per log
RUNS = {
    'Chesseract': dict(macro=(32, 2, 1),
                       geno=['conv3x3', 'avgpool3x3', 'conv3x3', 'conv3x3', 'none', 'conv1x1']),
    'Language':   dict(macro=(32, 2, 2),
                       geno=['conv3x3', 'conv3x3', 'conv1x1', 'none', 'conv3x3', 'conv1x1']),
    'Gutenberg':  dict(macro=(32, 2, 2),
                       geno=['conv3x3', 'avgpool3x3', 'conv3x3', 'skip', 'none', 'conv3x3']),
}


class FakeClock:
    def check(self):
        return 3600.0


def load(name):
    p = os.path.join('datasets', name)
    d = {k: np.load(os.path.join(p, k + '.npy'))
         for k in ('train_x', 'train_y', 'valid_x', 'valid_y')}
    meta = json.load(open(os.path.join(p, 'metadata')))
    return d, meta


def accuracy(model, loader, dev):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(dev)).argmax(1).cpu()
            c += int((p == y).sum()); t += y.numel()
    return c / max(1, t)


def main():
    name, n_epochs = sys.argv[2], int(sys.argv[3])
    cfg = RUNS[name]
    d, meta = load(name)
    dev = torch.device('cuda')

    dp = DataProcessor(d['train_x'], d['train_y'], d['valid_x'], d['valid_y'],
                       d['valid_x'], meta, FakeClock())
    train_loader, valid_loader, _ = dp.process()

    # a second, non-augmented loader over the TRAIN split, to measure train
    # accuracy under the same eval conditions as validation
    from data_processor import _ArrayDataset
    import torchvision.transforms as T
    norm = [s for s in train_loader.dataset.transform.transforms
            if isinstance(s, T.Normalize)][0]
    train_eval_ds = _ArrayDataset(d['train_x'][:10000], d['train_y'][:10000], transform=norm)
    train_eval = torch.utils.data.DataLoader(train_eval_ds, batch_size=512, shuffle=False)

    c0, n, dd = cfg['macro']
    xb, _ = next(iter(train_loader))
    model = build_skeleton(int(xb.shape[1]), int(meta['num_classes']), c0, n, dd,
                           genotype=cfg['geno']).to(dev)
    nparam = sum(p.numel() for p in model.parameters())

    # same optimiser policy as trainer.py
    decay = [p for p in model.parameters() if p.dim() > 1]
    nodecay = [p for p in model.parameters() if p.dim() <= 1]
    opt = torch.optim.SGD([{'params': decay, 'weight_decay': 3e-4},
                           {'params': nodecay, 'weight_decay': 0.0}],
                          lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    print('# %s  params=%d  batch=%d  benchmark=%.2f' % (
        name, nparam, train_loader.batch_size, meta['benchmark']))
    print('# epoch  train_acc  val_acc   gap   train_loss  t/ep')
    for ep in range(1, n_epochs + 1):
        model.train()
        t0 = time.time(); tot = nb = 0
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x.to(dev)), y.to(dev))
            loss.backward(); opt.step()
            tot += float(loss); nb += 1
        dt = time.time() - t0
        va = accuracy(model, valid_loader, dev)
        ta = accuracy(model, train_eval, dev)
        print('%6d  %8.4f  %7.4f  %+.4f  %10.4f  %.1fs'
              % (ep, ta, va, ta - va, tot / max(1, nb), dt), flush=True)


main()
