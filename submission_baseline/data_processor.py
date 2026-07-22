"""
DataProcessor - Step 1 (fallback-first baseline)

Responsibilities:
  * Convert raw numpy arrays [N, C, H, W] to float tensors; 3D inputs -> 4D
  * Per-channel normalization (stats from the training split)
  * Adaptive, conservative batch size (lowers OOM risk before training starts)
  * Return three dataloaders; test loader has NO shuffle and NO drop_last
    (otherwise an assert in the harness fails)
  * Store the most frequent training label as 'fallback_label' in metadata
    (used for a bulletproof predict() fallback in the Trainer)
"""

import numpy as np
import torch
import torchvision.transforms as transforms


def _to_4d_float(x):
    """numpy/array -> float32 tensor of shape [N, C, H, W]."""
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
        if self.y is None:                 # test split: image only
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
        """Rough, conservative heuristic based on elements per sample."""
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
        if n_train >= 2:
            bs = min(bs, n_train // 2)
        return max(1, bs)

    def process(self):
        train_t = _to_4d_float(self.train_x)
        n_train, c, h, w = train_t.shape

        mean = train_t.mean(dim=[0, 2, 3])
        std = train_t.std(dim=[0, 2, 3])
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
        normalize = transforms.Normalize(mean.tolist(), std.tolist())

        train_ds = _ArrayDataset(self.train_x, self.train_y, transform=normalize)
        valid_ds = _ArrayDataset(self.valid_x, self.valid_y, transform=normalize)
        test_ds = _ArrayDataset(self.test_x, None, transform=normalize)

        bs = self._choose_batch_size(c, h, w, n_train)
        drop_last = n_train > 2 * bs

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
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=bs, shuffle=False, drop_last=False)

        return train_loader, valid_loader, test_loader
