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

Better training policy (LR schedule tuning, AMP, augmentation) comes in
Step 4.
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

try:
    from sklearn.metrics import accuracy_score
    def _acc(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
except Exception:
    def _acc(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float((y_true == y_pred).mean()) if len(y_true) else 0.0


MAX_EPOCHS = 5000


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

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=3e-4)
        self.fallback_label = int(metadata.get('fallback_label', 0))

    def _remaining(self):
        try:
            return float(self.clock.check())
        except Exception:
            return 1e9   # clock unavailable -> don't abort artificially

    def _shrink_train_loader(self):
        """Halve the batch size and rebuild the train loader (runtime OOM guard)."""
        bs = max(1, self.train_dataloader.batch_size // 2)
        ds = self.train_dataloader.dataset
        drop_last = len(ds) > 2 * bs
        self.train_dataloader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, drop_last=drop_last)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return bs

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

        base_lr = 0.01
        planned = None          # estimated total epoch count (after epoch 1)
        epoch_time = 0.0

        try:
            for epoch in range(MAX_EPOCHS):
                # only start if the estimated epoch + reserve still fits
                if self._remaining() - epoch_time < margin:
                    break

                # manual cosine LR once the epoch count is estimated
                if planned is not None:
                    lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * min(epoch, planned) / planned))
                    for g in self.optimizer.param_groups:
                        g['lr'] = lr

                t0 = time.time()
                self.model.train()
                oom_batches = 0
                for data, target in self.train_dataloader:
                    try:
                        data = data.to(self.device)
                        target = target.to(self.device)
                        self.optimizer.zero_grad(set_to_none=True)
                        out = self.model(data)
                        loss = self.criterion(out, target)
                        loss.backward()
                        self.optimizer.step()
                    except RuntimeError as e:
                        if _is_oom(e):
                            oom_batches += 1
                            self.optimizer.zero_grad(set_to_none=True)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if oom_batches > 5:
                                new_bs = self._shrink_train_loader()
                                print("[Trainer] persistent OOM -> batch size reduced to {}".format(new_bs))
                                break   # abort this epoch, next one uses the smaller loader
                            continue
                        raise
                    if self._remaining() < margin:  # avoid a hard timeout
                        break

                epoch_time = time.time() - t0

                # after epoch 1, convert remaining budget into a total epoch estimate
                if planned is None and epoch_time > 0:
                    extra = int((self._remaining() - margin) / epoch_time)
                    planned = max(1, min(MAX_EPOCHS, (epoch + 1) + extra))

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

                print("  [Trainer] Epoch {:>2} | val={:5.2f}% | t/ep={:5.1f}s | t/eval={:5.1f}s | rem={:6.0f}s".format(
                    epoch + 1, val * 100, epoch_time, time.time() - t_eval, self._remaining()))
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
