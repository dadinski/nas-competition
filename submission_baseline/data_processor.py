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


class _GaussianNoise:
    """Adds small Gaussian noise, scaled relative to the (already
    per-channel-normalized) data. This is the one augmentation that needs
    NO assumption about what an axis or value means - it works identically
    whether the array is a photo, a one-hot board encoding, or something
    we've never seen - so it's always applied, independent of any
    'photo vs symbolic' guess. Must run AFTER Normalize."""
    def __init__(self, std=0.03):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std


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

    def _query_free_gpu_mem_mb(self):
        """Best-effort query of currently-free GPU memory in MB.
        Returns None if there is no GPU or the query fails for any reason
        (e.g. older CUDA driver) - callers must treat None as 'unknown'."""
        if not torch.cuda.is_available():
            return None
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            return free_bytes / (1024 ** 2)
        except Exception:
            return None

    def _choose_batch_size(self, c, h, w, n_train):
        """Batch size from the GPU memory actually free right now. The
        NAS/model architecture isn't known yet at this stage, so we can't
        compute an exact per-sample memory cost - instead we budget a
        conservative slice of free memory and divide by the raw sample
        size times a generous overhead factor (activations + gradients +
        optimizer state of a small-to-medium CNN). If no GPU is available
        (or the query fails), we fall back to the old fixed heuristic."""
        elems = int(c) * int(h) * int(w)
        bytes_per_sample = max(1, elems * 4)  # float32

        free_mb = self._query_free_gpu_mem_mb()
        if free_mb is not None:
            usable_fraction = 0.2      # leave the rest for model + activations/gradients
            overhead_factor = 40       # rough multiplier for a training step, not just storage
            budget_bytes = free_mb * (1024 ** 2) * usable_fraction
            bs = int(budget_bytes / (bytes_per_sample * overhead_factor))
            bs = max(4, min(bs, 512))
        else:
            # CPU-only fallback: previous fixed, element-count based heuristic
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

    # A handful of exactly-repeated values (one-hot 0/1, digit codes like
    # 0.1..1.0, class indices...) signals a quantized/categorical encoding;
    # many distinct values signals continuous data (pixel intensities,
    # sensor floats...). This threshold is a rough cutoff, not a precise
    # boundary - see _build_train_transform for why we check this at all.
    _CARDINALITY_THRESHOLD = 32

    def _estimate_value_cardinality(self, train_t, max_check=20000):
        """Rough count of distinct values in a sample of the training
        tensor, used as a DATA-DRIVEN (not shape-based) signal for whether
        an axis-flip is even plausible to be safe. Sampling keeps this
        cheap on large datasets."""
        flat = train_t.reshape(-1)
        if flat.numel() > max_check:
            idx = torch.randperm(flat.numel())[:max_check]
            flat = flat[idx]
        rounded = torch.round(flat * 1000) / 1000  # avoid float noise inflating the count
        return int(torch.unique(rounded).numel()), int(flat.numel())

    def _build_train_transform(self, train_t):
        """Conservative, TRAIN-ONLY augmentation - this is an UNSEEN-DATA
        competition, so nothing here may assume a specific dataset. Earlier
        we branched on input SHAPE (channels/spatial size) and justified a
        flip using structural facts we happened to know about two example
        datasets (Sudoku, Chesseract). That reasoning does not generalize:
        another unseen grid-shaped dataset could encode something where an
        axis-flip is meaningless or actively wrong (e.g. an axis that
        indexes "which letter of the alphabet" rather than a spatial
        position - flipping it just swaps letter identities). Shape alone
        cannot tell us that.

        Instead we look at what the DATA actually looks like:

          * Many distinct values (continuous, e.g. pixel intensities) is
            the closest thing to a dataset-agnostic photo signal - most
            continuous-valued spatial data tolerates small translations
            and a left-right mirror. We add reflect-pad + random crop +
            horizontal flip on top of noise.
          * Few distinct values (quantized/categorical - one-hot, digit
            codes, class maps, ...) means we don't know what a spatial
            transform would do to the encoding, so we skip flip/crop
            entirely and rely only on the noise below.

        This still isn't a guarantee (a continuous-valued dataset could
        still have a directional axis, e.g. a spectrogram's time axis) -
        it is a best-effort default, not a certainty. Set
        metadata['use_augmentation'] = False to disable all of this if a
        run suggests it's hurting on a particular dataset.

        A small amount of Gaussian noise (see _GaussianNoise) is applied
        regardless of the above, since it needs no assumption about axis
        meaning at all.
        """
        if not self.metadata.get('use_augmentation', True):
            self.metadata['augmentation_policy'] = 'disabled'
            print("[DataProcessor] augmentation disabled via metadata")
            return None, None

        _n, c, h, w = train_t.shape
        n_unique, n_checked = self._estimate_value_cardinality(train_t)
        is_continuous = n_unique > self._CARDINALITY_THRESHOLD
        can_crop = h >= 8 and w >= 8   # crop is meaningless on a tiny grid regardless

        noise = _GaussianNoise(std=0.03)

        if is_continuous and can_crop:
            policy = 'continuous (pad+crop+flip+noise)'
            pad = max(2, min(h, w) // 8)
            geo = transforms.Compose([
                transforms.Pad(pad, padding_mode='reflect'),
                transforms.RandomCrop((h, w)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        elif is_continuous:
            policy = 'continuous but too small to crop safely (noise only)'
            geo = None
        else:
            policy = 'quantized/categorical (noise only)'
            geo = None

        self.metadata['augmentation_policy'] = policy
        self.metadata['augmentation_stats'] = {'n_unique_sampled': n_unique, 'n_checked': n_checked}
        print("[DataProcessor] augmentation: {} - shape {}x{}x{}, "
              "{} unique values seen in {} sampled".format(policy, c, h, w, n_unique, n_checked))
        return geo, noise

    def process(self):
        train_t = _to_4d_float(self.train_x)
        n_train, c, h, w = train_t.shape

        mean = train_t.mean(dim=[0, 2, 3])
        std = train_t.std(dim=[0, 2, 3])
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
        normalize = transforms.Normalize(mean.tolist(), std.tolist())

        # Augmentation is TRAIN-ONLY: valid/test must stay a clean,
        # deterministic measure of generalization, and the harness expects
        # test predictions in a fixed 1:1 order with the (unshuffled) test
        # loader, so test data must never be randomized.
        # Order matters: geometric transforms run on raw values (scale-
        # independent), noise runs AFTER Normalize so its magnitude means
        # the same thing (~3% of one std) on every dataset.
        geo_aug, noise_aug = self._build_train_transform(train_t)
        steps = []
        if geo_aug is not None:
            steps.append(geo_aug)
        steps.append(normalize)
        if noise_aug is not None:
            steps.append(noise_aug)
        train_transform = transforms.Compose(steps) if len(steps) > 1 else normalize

        train_ds = _ArrayDataset(self.train_x, self.train_y, transform=train_transform)
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
