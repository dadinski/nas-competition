"""A/B: does mixup fix the overfitting seen on the quantized/categorical
datasets (Chesseract: 99.99% train vs 53% val)?

Mixup is the one strong augmentation that assumes NOTHING about what an axis
means - it only convex-combines whole samples and their labels, so unlike
flip/crop it is safe on a one-hot board, a spectrogram or a photo alike. That
property is what makes it a candidate for an unseen-data competition.

Each arm is trained from the SAME initial weights (same seed) so the arms are
comparable. Reports best val over the run, which is what the Trainer's
checkpointing actually keeps.

usage: ab_mixup.py <submission_dir> <dataset> <epochs> [seed]
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(sys.argv[1]))

import numpy as np
import torch
import torch.nn as nn

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


class FakeClock:
    def check(self):
        return 3600.0


def load(name):
    p = os.path.join('datasets', name)
    d = {k: np.load(os.path.join(p, k + '.npy'))
         for k in ('train_x', 'train_y', 'valid_x', 'valid_y')}
    return d, json.load(open(os.path.join(p, 'metadata')))


def evaluate(model, loader, dev):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(dev)).argmax(1).cpu()
            c += int((p == y).sum()); t += y.numel()
    return c / max(1, t)


def run_arm(label, alpha, seed, cfg, meta, train_loader, valid_loader, train_eval,
            in_ch, n_epochs, dev):
    torch.manual_seed(seed)
    c0, n, dd = cfg['macro']
    model = build_skeleton(in_ch, int(meta['num_classes']), c0, n, dd,
                           genotype=cfg['geno']).to(dev)
    decay = [p for p in model.parameters() if p.dim() > 1]
    nodecay = [p for p in model.parameters() if p.dim() <= 1]
    opt = torch.optim.SGD([{'params': decay, 'weight_decay': 3e-4},
                           {'params': nodecay, 'weight_decay': 0.0}],
                          lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0; best_ep = 0; final_train = 0.0
    for ep in range(1, n_epochs + 1):
        model.train()
        # cosine over the fixed epoch count, mirroring the production anneal
        lr = 0.5 * 0.01 * (1 + np.cos(np.pi * (ep - 1) / n_epochs))
        for g in opt.param_groups:
            g['lr'] = lr
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            if alpha > 0:
                lam = float(np.random.beta(alpha, alpha))
                perm = torch.randperm(x.shape[0], device=dev)
                x = lam * x + (1 - lam) * x[perm]
                out = model(x)          # ONE forward pass - calling model(x)
                                        # twice doubles the cost and runs the
                                        # BN running-stat update twice
                loss = lam * crit(out, y) + (1 - lam) * crit(out, y[perm])
            else:
                loss = crit(model(x), y)
            loss.backward(); opt.step()
        va = evaluate(model, valid_loader, dev)
        if va > best:
            best, best_ep = va, ep
        if ep == n_epochs:
            final_train = evaluate(model, train_eval, dev)
    print('  %-18s best_val=%.4f @ep%-3d  final_train=%.4f  gap=%+.4f'
          % (label, best, best_ep, final_train, final_train - best), flush=True)
    return best


def main():
    name, n_epochs = sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    cfg = RUNS[name]
    d, meta = load(name)
    dev = torch.device('cuda')

    dp = DataProcessor(d['train_x'], d['train_y'], d['valid_x'], d['valid_y'],
                       d['valid_x'], meta, FakeClock())
    train_loader, valid_loader, _ = dp.process()

    from data_processor import _ArrayDataset
    import torchvision.transforms as T
    norm = [s for s in train_loader.dataset.transform.transforms
            if isinstance(s, T.Normalize)][0]
    train_eval = torch.utils.data.DataLoader(
        _ArrayDataset(d['train_x'][:10000], d['train_y'][:10000], transform=norm),
        batch_size=512, shuffle=False)

    in_ch = int(next(iter(train_loader))[0].shape[1])
    print('# %s  seed=%d  epochs=%d  benchmark=%.2f  (production best_val was the '
          'number that got checkpointed)' % (name, seed, n_epochs, meta['benchmark']))
    for label, alpha in (('baseline', 0.0), ('mixup a=0.2', 0.2), ('mixup a=0.4', 0.4)):
        t0 = time.time()
        run_arm(label, alpha, seed, cfg, meta, train_loader, valid_loader, train_eval,
                in_ch, n_epochs, dev)
        print('     (%.0fs)' % (time.time() - t0), flush=True)


main()
