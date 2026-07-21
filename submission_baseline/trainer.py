"""
Trainer - Schritt 1 (Fallback-first Baseline)

Leitgedanke: NIE einen -10-Fall ausloesen (Timeout / Crash / fehlende
Vorhersagen). Daher:
  * Training laeuft clock-budgetiert: es wird eine Reserve fuer predict
    zurueckgehalten und rechtzeitig abgebrochen
  * Bestes Val-Modell wird als Checkpoint gehalten und am Ende wiederhergestellt
  * OOM-Batches werden abgefangen (Cache leeren, Batch ueberspringen);
    anhaltendes OOM beendet das Training sauber statt zu crashen
  * predict() gibt IMMER exakt n_test Vorhersagen in Reihenfolge zurueck -
    im Notfall mit der haeufigsten Trainingsklasse aufgefuellt

Feineres Training (LR-Schedule, AMP, Augmentation) kommt in Schritt 4.
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


MAX_EPOCHS = 50


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
            return 1e9   # Uhr nicht abfragbar -> nicht kuenstlich abbrechen

    def _shrink_train_loader(self):
        """Batch-Size halbieren und Train-Loader neu bauen (Laufzeit-OOM-Schutz)."""
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
        # Reserve fuer predict + Overhead: 10% des Budgets, gedeckelt auf [20s, 120s]
        margin = max(20.0, min(0.10 * budget, 120.0))

        base_lr = 0.01
        planned = None          # geschaetzte Gesamt-Epochenzahl (nach 1. Epoche)
        epoch_time = 0.0

        try:
            for epoch in range(MAX_EPOCHS):
                # Nur starten, wenn geschaetzte Epoche + Reserve noch reinpasst
                if self._remaining() - epoch_time < margin:
                    break

                # Manuelle Cosine-LR, sobald die Epochenzahl geschaetzt ist
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
                                print("[Trainer] Anhaltendes OOM -> Batch-Size auf {} reduziert".format(new_bs))
                                break   # Epoche abbrechen, naechste laeuft mit kleinerem Loader
                            continue
                        raise
                    if self._remaining() < margin:  # harten Timeout vermeiden
                        break

                epoch_time = time.time() - t0

                # Nach der 1. Epoche das Restbudget in eine Gesamt-Epochenzahl umrechnen
                if planned is None and epoch_time > 0:
                    extra = int((self._remaining() - margin) / epoch_time)
                    planned = max(1, min(MAX_EPOCHS, (epoch + 1) + extra))

                val = self._evaluate()
                if val >= best_val:
                    best_val = val
                    best_state = copy.deepcopy(self.model.state_dict())

                print("  [Trainer] Epoch {:>2} | val={:5.2f}% | t/ep={:5.1f}s | rem={:6.0f}s".format(
                    epoch + 1, val * 100, epoch_time, self._remaining()))
        except Exception as e:
            print("[Trainer] Training vorzeitig beendet:", repr(e))

        # Bestes gesehenes Modell wiederherstellen
        try:
            self.model.load_state_dict(best_state)
        except Exception:
            pass
        return self.model

    def _evaluate(self):
        self.model.eval()
        y_true, y_pred = [], []
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
        return _acc(y_true, y_pred) if y_true else 0.0

    def _predict_batch(self, data):
        """Vorhersage fuer einen Batch; bei OOM rekursiv halbieren (Reihenfolge bleibt)."""
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
            return [self.fallback_label]   # einzelnes Sample passt nicht -> Fallback

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
            print("[Trainer] predict-Notfall:", repr(e))

        # Laenge exakt auf n_test bringen - nie kuerzer, nie laenger
        if len(preds) < n_test:
            preds += [self.fallback_label] * (n_test - len(preds))
        elif len(preds) > n_test:
            preds = preds[:n_test]
        return preds
