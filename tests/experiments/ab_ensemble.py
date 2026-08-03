"""A/B: can surplus budget be converted into accuracy by ENSEMBLING?

The saturating datasets reach ~100% train accuracy in a few epochs and then gain
nothing from the remaining 90% of the budget. More capacity would make that worse
(they overfit, not underfit) and more epochs demonstrably do nothing. Ensembling
is the one mechanism that turns arbitrary extra wall-clock into accuracy without
assuming anything about the data.

Every arm gets the SAME total epoch budget, so this measures how the budget is
SPENT, not how much of it there is. That is the whole question.

  single      - today's pipeline: one cosine BASE_LR -> 0, keep best-val state
  snap-warm   - K cycles, cosine restarts, weights CARRIED OVER (SGDR / the
                "Snapshot Ensembles" recipe). Cheap, but on a dataset that has
                memorised its training set the restart may just re-memorise and
                give near-identical members.
  snap-reinit - K cycles, weights RE-INITIALISED each cycle. Maximum diversity,
                but each member must converge from scratch.

For each ensemble arm we report BOTH the ensemble accuracy and the best single
member, so the ensembling gain is separated from any effect of the cyclic LR
itself. Members are averaged as softmax probabilities.

usage: ab_ensemble.py <submission_dir> <dataset> <total_epochs> <cycles> [seed]
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(sys.argv[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_processor import DataProcessor
from model import build_skeleton

RUNS = {
    'Chesseract': dict(macro=(32, 2, 1),
                       geno=['conv3x3', 'avgpool3x3', 'conv3x3', 'conv3x3', 'none', 'conv1x1']),
    'Language':   dict(macro=(32, 2, 2),
                       geno=['conv3x3', 'conv3x3', 'conv1x1', 'none', 'conv3x3', 'conv1x1']),
    'Gutenberg':  dict(macro=(32, 2, 2),
                       geno=['conv3x3', 'avgpool3x3', 'conv3x3', 'skip', 'none', 'conv3x3']),
}
BASE_LR = 0.01


class FakeClock:
    def check(self):
        return 3600.0


def load(name):
    p = os.path.join('datasets', name)
    d = {k: np.load(os.path.join(p, k + '.npy'))
         for k in ('train_x', 'train_y', 'valid_x', 'valid_y')}
    return d, json.load(open(os.path.join(p, 'metadata')))


def make_model(cfg, meta, in_ch, dev, seed):
    torch.manual_seed(seed)
    c0, n, dd = cfg['macro']
    return build_skeleton(in_ch, int(meta['num_classes']), c0, n, dd,
                          genotype=cfg['geno']).to(dev)


def make_opt(model):
    decay = [p for p in model.parameters() if p.dim() > 1]
    nodecay = [p for p in model.parameters() if p.dim() <= 1]
    return torch.optim.SGD([{'params': decay, 'weight_decay': 3e-4},
                            {'params': nodecay, 'weight_decay': 0.0}],
                           lr=BASE_LR, momentum=0.9)


@torch.no_grad()
def probs_and_labels(model, loader, dev):
    """Softmax probabilities over the whole split, plus the labels (once)."""
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        ps.append(F.softmax(model(x.to(dev)).float(), dim=1).cpu())
        ys.append(y)
    return torch.cat(ps), torch.cat(ys)


def acc_from_probs(p, y):
    return float((p.argmax(1) == y).float().mean())


def run_arm(label, cfg, meta, train_loader, valid_loader, in_ch, dev,
            total_epochs, cycles, seed):
    """cycles == 1 reproduces the current pipeline exactly."""
    reinit = label.endswith('reinit')
    model = make_model(cfg, meta, in_ch, dev, seed)
    opt = make_opt(model)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    per_cycle = max(1, total_epochs // cycles)
    members = []            # (val_acc, probs) for the best epoch of each cycle
    t0 = time.time()

    for c in range(cycles):
        if c > 0 and reinit:
            model = make_model(cfg, meta, in_ch, dev, seed + 1000 * c)
            opt = make_opt(model)
        elif c > 0:
            opt = make_opt(model)        # fresh momentum, weights carried over

        best_in_cycle = (-1.0, None)
        for e in range(per_cycle):
            # cosine WITHIN the cycle: BASE_LR -> ~0, then the next cycle
            # restarts at BASE_LR. This is what decorrelates the snapshots.
            lr = 0.5 * BASE_LR * (1 + np.cos(np.pi * e / per_cycle))
            for g in opt.param_groups:
                g['lr'] = lr
            model.train()
            for x, y in train_loader:
                opt.zero_grad(set_to_none=True)
                crit(model(x.to(dev)), y.to(dev)).backward()
                opt.step()
            p, yv = probs_and_labels(model, valid_loader, dev)
            a = acc_from_probs(p, yv)
            if a > best_in_cycle[0]:
                best_in_cycle = (a, p)
        members.append(best_in_cycle)

    y = yv
    singles = [m[0] for m in members]
    ens = torch.stack([m[1] for m in members]).mean(0)
    ens_acc = acc_from_probs(ens, y)
    print('  %-13s ensemble=%.4f  best_single=%.4f  gain=%+.4f  members=[%s]  (%.0fs)'
          % (label, ens_acc, max(singles), ens_acc - max(singles),
             ' '.join('%.3f' % s for s in singles), time.time() - t0), flush=True)
    return ens_acc


def main():
    name = sys.argv[2]
    total_epochs = int(sys.argv[3])
    cycles = int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    cfg = RUNS[name]
    d, meta = load(name)
    dev = torch.device('cuda')

    dp = DataProcessor(d['train_x'], d['train_y'], d['valid_x'], d['valid_y'],
                       d['valid_x'], meta, FakeClock())
    train_loader, valid_loader, _ = dp.process()
    in_ch = int(next(iter(train_loader))[0].shape[1])

    print('# %s seed=%d  total_epochs=%d (identical for every arm)  cycles=%d  benchmark=%.2f'
          % (name, seed, total_epochs, cycles, meta['benchmark']))
    run_arm('single', cfg, meta, train_loader, valid_loader, in_ch, dev,
            total_epochs, 1, seed)
    run_arm('snap-warm', cfg, meta, train_loader, valid_loader, in_ch, dev,
            total_epochs, cycles, seed)
    run_arm('snap-reinit', cfg, meta, train_loader, valid_loader, in_ch, dev,
            total_epochs, cycles, seed)


main()
