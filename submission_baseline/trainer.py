"""
Trainer - Step 1 (fallback-first baseline) + Step 2 (OOM-safe runtime)

Guiding principle: NEVER trigger a -10 outcome (timeout / crash / missing
predictions). Therefore:
  * Training is clock-budgeted: a reserve for predict() is held back and
    training stops in time
  * The best validation checkpoint is kept and restored at the end
  * OOM batches are caught (clear cache, skip batch); persistent OOM shrinks
    the batch size and rebuilds the loader instead of crashing
  * predict() ALWAYS returns exactly n_test predictions in order - filled
    with the most frequent training class as a last resort, and halves the
    batch recursively on OOM
  * The cosine LR schedule is driven by the CLOCK, not by an estimated epoch
    count, so a cut-short epoch can no longer corrupt it (see train())
  * The batch size hint recorded by NAS is applied up front, so the first
    epoch doesn't have to discover the same OOM all over again
  * No training batch of size 1 ever reaches BatchNorm (see helpers.safe_drop_last)

Training policy: label smoothing, and weight decay on matrix-valued
parameters only.
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from helpers import safe_drop_last


def _select_amp():
    """Bind the AMP API this runtime actually supports, warning-free.

    `torch.cuda.amp.autocast/GradScaler` work everywhere we might land but emit
    a FutureWarning on torch >= 2.4 (several lines per epoch - it swamps the
    training log). The modern spelling, `torch.amp.*` with an explicit device
    string, does not exist on older torch: `torch.amp.GradScaler(device)`
    arrived well after 1.10, which the starter kit pins and which the
    evaluation server may still run (see CLAUDE.md 3 - requirements.txt is NOT
    shipped, so we cannot assume a version).

    So probe once at import: actually construct both objects and enter the
    context manager, with warnings escalated to errors. Only if that is clean
    do we adopt the modern spelling; anything at all wrong falls back to the
    legacy one. This keeps a torch-1.10 server working while giving a modern
    one a clean log.

    Returns (autocast, GradScaler) adapters taking a single `enabled` kwarg,
    so the call sites stay identical either way.
    """
    try:
        import warnings
        from torch.amp import autocast as _ac, GradScaler as _gs
        with warnings.catch_warnings():
            warnings.simplefilter('error')      # any deprecation -> fall back
            _gs('cuda', enabled=False)
            with _ac('cuda', enabled=False):
                pass

        def autocast(enabled=True):
            return _ac('cuda', enabled=enabled)

        def GradScaler(enabled=True):
            return _gs('cuda', enabled=enabled)

        return autocast, GradScaler
    except Exception:
        from torch.cuda.amp import autocast as _ac, GradScaler as _gs
        return _ac, _gs


autocast, GradScaler = _select_amp()

try:
    from sklearn.metrics import accuracy_score
    def _acc(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
except Exception:
    def _acc(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float((y_true == y_pred).mean()) if len(y_true) else 0.0


MAX_EPOCHS = 5000

BASE_LR = 0.01
WEIGHT_DECAY = 3e-4
LABEL_SMOOTHING = 0.1


def _is_oom(err):
    return isinstance(err, RuntimeError) and 'out of memory' in str(err).lower()


class Trainer:
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock

        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer(model)
        self.fallback_label = int(metadata.get('fallback_label', 0))

        # AMP: FP16 compute + GradScaler. GradScaler(enabled=False) and
        # autocast(enabled=False) are documented no-ops, so this same code
        # runs unchanged on CPU-only boxes or GPUs without AMP benefit.
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        # NAS records nas_batch_size_hint when the searched cell only fits at
        # a smaller batch than the loader's default (see nas.py). Ignoring it
        # meant the first epoch was GUARANTEED to walk into the persistent-OOM
        # guard below, which aborts that epoch part-way through - wasting the
        # epoch and (before the clock-driven schedule) corrupting the LR
        # schedule's epoch estimate. Apply it before training starts instead.
        self._apply_nas_batch_size_hint()

    def _build_criterion(self):
        """CrossEntropyLoss with label smoothing where the runtime supports it.
        `label_smoothing` arrived in torch 1.10 and the evaluation server's
        version is unknown to us (see CLAUDE.md 3), so degrade rather than
        crash on an older one."""
        try:
            return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        except TypeError:
            print('[Trainer] label_smoothing unsupported by this torch, '
                  'falling back to plain CrossEntropyLoss')
            return nn.CrossEntropyLoss()

    def _build_optimizer(self, model):
        """SGD with weight decay applied to matrix-valued parameters only.
        BatchNorm affine terms and biases are 1-D; decaying them pulls the
        normalisation's learned scale and shift toward zero, which costs
        accuracy while regularising essentially nothing. Splitting on
        dimensionality needs no module introspection and so cannot miss a
        custom block in the searched cell."""
        decay, no_decay = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() > 1 else no_decay).append(p)
        groups = [{'params': decay, 'weight_decay': WEIGHT_DECAY},
                  {'params': no_decay, 'weight_decay': 0.0}]
        print('[Trainer] weight decay on {} tensors, none on {} (BN/bias)'.format(
            len(decay), len(no_decay)))
        return optim.SGD(groups, lr=BASE_LR, momentum=0.9)

    def _apply_nas_batch_size_hint(self):
        """Shrink the train loader up front if NAS found the model only fits
        at a smaller batch size than the DataProcessor chose."""
        try:
            hint = self.metadata.get('nas_batch_size_hint')
            current = self.train_dataloader.batch_size
            if hint is None or current is None or int(hint) >= int(current):
                return
            new_bs = self._rebuild_train_loader(int(hint))
            print("[Trainer] applying nas_batch_size_hint: train batch size {} -> {}".format(
                int(current), new_bs))
        except Exception as e:
            # a bad hint must never stop training - the runtime OOM guard
            # in train() remains as the safety net
            print("[Trainer] could not apply nas_batch_size_hint:", repr(e))

    def _remaining(self):
        try:
            return float(self.clock.check())
        except Exception:
            return 1e9   # clock unavailable -> don't abort artificially

    def _rebuild_train_loader(self, bs):
        """Rebuild the train loader at batch size `bs`, keeping the
        shuffle/drop_last semantics the DataProcessor used."""
        # floor of 2, not 1: a full loader of single-sample batches would make
        # BatchNorm raise on every batch (see helpers.safe_drop_last)
        bs = max(2, int(bs))
        ds = self.train_dataloader.dataset
        drop_last = safe_drop_last(len(ds), bs)
        self.train_dataloader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, drop_last=drop_last)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return bs

    def _shrink_train_loader(self):
        """Halve the batch size and rebuild the train loader (runtime OOM guard)."""
        return self._rebuild_train_loader(self.train_dataloader.batch_size // 2)

    def train(self):
        try:
            self.model.to(self.device)
        except Exception:
            pass

        best_state = copy.deepcopy(self.model.state_dict())
        best_val = -1.0

        budget = self._remaining()
        # Reserve for predict() + overhead, proportional to the budget.
        # IMPORTANT: no large fixed floor - under very short budgets (<=60s)
        # a fixed 20s floor used to eat the ENTIRE budget, so not a single
        # epoch ever ran (bug found via a 20s stress test).
        margin = max(2.0, min(0.15 * budget, 60.0))

        # Cosine LR driven by the CLOCK rather than by an estimated epoch count.
        # The old version froze a `planned` total epoch count from epoch 1's
        # duration - but epoch 1 is precisely the epoch that can be cut short
        # (persistent-OOM guard below, cudnn autotuning, allocator warm-up).
        # An aborted 0.9s epoch 1 followed by real 31.7s epochs yielded
        # planned~3500 while only ~93 epochs actually ran, so
        # cos(pi*93/3500) ~ 1.0 and the LR never annealed at all - observed on
        # 4 of 9 datasets. Deriving progress from the budget actually consumed
        # is immune to partial epochs, OOM restarts and varying epoch cost, and
        # reaches lr~0 exactly as the training budget runs out.
        usable = max(1e-6, budget - margin)   # wall-clock actually available for training
        # _remaining() returns 1e9 when the clock is unavailable. Without this
        # check that sentinel makes progress ~0.0 forever, pinning the LR at
        # base_lr for every one of MAX_EPOCHS epochs with no annealing at all.
        clock_ok = budget < 1e8
        if not clock_ok:
            print('[Trainer] clock unavailable - holding LR at base_lr, no annealing')

        epoch_time = 0.0     # duration of the last COMPLETED epoch (see below)
        lr = BASE_LR

        try:
            for epoch in range(MAX_EPOCHS):
                # only start if the estimated epoch + reserve still fits
                if self._remaining() - epoch_time < margin:
                    break

                if clock_ok:
                    progress = 1.0 - (self._remaining() - margin) / usable
                    progress = min(1.0, max(0.0, progress))
                    lr = 0.5 * BASE_LR * (1.0 + math.cos(math.pi * progress))
                    for g in self.optimizer.param_groups:
                        g['lr'] = lr

                t0 = time.time()
                self.model.train()
                oom_batches = 0
                epoch_complete = True    # cleared if this epoch is cut short
                for data, target in self.train_dataloader:
                    # Belt-and-braces against a single-sample batch: BatchNorm
                    # raises on one sample in train mode, and that is not an
                    # OOM, so it would propagate and end training entirely.
                    if data.shape[0] < 2:
                        continue
                    try:
                        data = data.to(self.device)
                        target = target.to(self.device)
                        self.optimizer.zero_grad(set_to_none=True)
                        with autocast(enabled=self.use_amp):
                            out = self.model(data)
                            loss = self.criterion(out, target)
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    except RuntimeError as e:
                        if _is_oom(e):
                            oom_batches += 1
                            self.optimizer.zero_grad(set_to_none=True)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if oom_batches > 5:
                                new_bs = self._shrink_train_loader()
                                print("[Trainer] persistent OOM -> batch size reduced to {}".format(new_bs))
                                epoch_complete = False
                                break   # abort this epoch, next one uses the smaller loader
                            continue
                        raise
                    if self._remaining() < margin:  # avoid a hard timeout
                        epoch_complete = False
                        break

                # Only a COMPLETED epoch says anything about what the next one
                # will cost; a partial epoch under-estimates it badly, which is
                # what used to poison the schedule. Keep the previous estimate
                # in that case (0.0 on the first epoch -> just try again).
                this_epoch_time = time.time() - t0
                if epoch_complete:
                    epoch_time = this_epoch_time

                # Never start a validation pass that could eat into the
                # margin reserved for predict() - a slow/large valid set
                # would otherwise risk pushing the whole dataset past its
                # time limit (-> instant -10 for this dataset).
                remaining = self._remaining()
                if remaining <= margin:
                    print("  [Trainer] Epoch {:>2} | skipping validation - "
                          "no time left beyond the {:.0f}s margin".format(epoch + 1, margin))
                    break

                eval_budget = remaining - margin
                t_eval = time.time()
                val = self._evaluate(time_budget=eval_budget)
                if val >= best_val:
                    best_val = val
                    best_state = copy.deepcopy(self.model.state_dict())

                print("  [Trainer] Epoch {:>2} | val={:5.2f}% | lr={:.5f} | t/ep={:5.1f}s | t/eval={:5.1f}s | rem={:6.0f}s".format(
                    epoch + 1, val * 100, lr, this_epoch_time, time.time() - t_eval, self._remaining()))
        except Exception as e:
            print("[Trainer] training ended early:", repr(e))

        # restore the best model seen so far
        try:
            self.model.load_state_dict(best_state)
        except Exception:
            pass
        return self.model

    def _evaluate(self, time_budget=None):
        """Validation pass. If time_budget (seconds) is given, the pass
        stops early once it's exceeded, so a slow or unexpectedly large
        validation set can never run past it - it just scores on however
        many batches it got through instead of blowing the time budget."""
        self.model.eval()
        y_true, y_pred = [], []
        t0 = time.time()
        with torch.no_grad():
            for data, target in self.valid_dataloader:
                try:
                    data = data.to(self.device)
                    with autocast(enabled=self.use_amp):
                        out = self.model(data)
                    y_pred += torch.argmax(out, 1).cpu().tolist()
                    y_true += target.tolist()
                except RuntimeError as e:
                    if _is_oom(e) and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                if time_budget is not None and (time.time() - t0) > time_budget:
                    print("  [Trainer] validation time budget ({:.0f}s) reached, "
                          "scoring on {}/{} valid samples".format(
                              time_budget, len(y_true), len(self.valid_dataloader.dataset)))
                    break
        return _acc(y_true, y_pred) if y_true else 0.0

    def _predict_batch(self, data):
        """Predict one batch; halve recursively on OOM (order is preserved)."""
        try:
            with autocast(enabled=self.use_amp):
                out = self.model(data.to(self.device))
            return torch.argmax(out, 1).cpu().tolist()
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if data.shape[0] > 1:
                mid = data.shape[0] // 2
                return self._predict_batch(data[:mid]) + self._predict_batch(data[mid:])
            return [self.fallback_label]   # a single sample still doesn't fit -> fallback

    def predict(self, test_loader):
        n_test = len(test_loader.dataset)
        preds = []
        try:
            self.model.to(self.device)
        except Exception:
            pass
        self.model.eval()

        try:
            with torch.no_grad():
                for data in test_loader:
                    preds += self._predict_batch(data)
        except Exception as e:
            print("[Trainer] predict() emergency handling:", repr(e))

        # force the length to exactly n_test - never shorter, never longer
        if len(preds) < n_test:
            preds += [self.fallback_label] * (n_test - len(preds))
        elif len(preds) > n_test:
            preds = preds[:n_test]
        return preds
