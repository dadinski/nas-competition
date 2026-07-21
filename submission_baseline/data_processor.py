"""
DataProcessor - Schritt 1 (Fallback-first Baseline)

Aufgaben:
  * Rohdaten (numpy [N, C, H, W]) in float-Tensoren wandeln, 3D-Inputs -> 4D
  * Per-Channel-Normalisierung (Statistik aus dem Trainingssatz)
  * Adaptive, konservative Batch-Size (senkt OOM-Risiko schon vor dem Training)
  * Drei Dataloader zurueckgeben; Test-Loader OHNE Shuffle und OHNE drop_last
    (sonst schlaegt ein assert im Harness fehl)
  * Haeufigste Trainingsklasse als 'fallback_label' in die Metadata legen
    (fuer eine bombensichere predict-Notloesung im Trainer)
"""

import numpy as np
import torch
import torchvision.transforms as transforms


def _to_4d_float(x):
    """numpy/array -> float32-Tensor der Form [N, C, H, W]."""
    t = torch.as_tensor(np.asarray(x)).float()
    if t.dim() == 3:                       # [N, H, W] -> [N, 1, H, W]
        t = t.unsqueeze(1)
    return t


class _ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = _to_4d_float(x)
        self.y = None if y is None else torch.as_tensor(np.asarray(y)).long()
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        im = self.x[idx]
        if self.transform is not None:
            im = self.transform(im)
        if self.y is None:                 # Test-Split: nur das Bild
            return im
        return im, self.y[idx]


class DataProcessor:
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock

    def _choose_batch_size(self, c, h, w, n_train):
        """Grobe, konservative Heuristik nach Elementen pro Sample."""
        elems = int(c) * int(h) * int(w)
        if elems <= 1024:
            bs = 256
        elif elems <= 4096:
            bs = 128
        elif elems <= 16384:
            bs = 64
        elif elems <= 65536:
            bs = 32
        else:
            bs = 16
        # nie groesser als der halbe Trainingssatz
        if n_train >= 2:
            bs = min(bs, n_train // 2)
        return max(1, bs)

    def process(self):
        train_t = _to_4d_float(self.train_x)
        n_train, c, h, w = train_t.shape

        # Per-Channel-Statistik; std==0 abfangen (konstante Kanaele)
        mean = train_t.mean(dim=[0, 2, 3])
        std = train_t.std(dim=[0, 2, 3])
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
        normalize = transforms.Normalize(mean.tolist(), std.tolist())

        train_ds = _ArrayDataset(self.train_x, self.train_y, transform=normalize)
        valid_ds = _ArrayDataset(self.valid_x, self.valid_y, transform=normalize)
        test_ds = _ArrayDataset(self.test_x, None, transform=normalize)

        bs = self._choose_batch_size(c, h, w, n_train)
        drop_last = n_train > 2 * bs       # sonst koennten (fast) alle Daten wegfallen

        # Nachrichten an spaetere Klassen ueber die Metadata (offiziell erlaubt)
        try:
            labels = np.asarray(self.train_y).reshape(-1)
            self.metadata['fallback_label'] = int(np.bincount(labels).argmax())
        except Exception:
            self.metadata['fallback_label'] = 0
        self.metadata['batch_size'] = int(bs)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=bs, shuffle=True, drop_last=drop_last)
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=bs, shuffle=False, drop_last=False)
        # WICHTIG: kein Shuffle, kein drop_last -> Reihenfolge bleibt, Harness-assert ok
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=bs, shuffle=False, drop_last=False)

        return train_loader, valid_loader, test_loader
