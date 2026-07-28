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

from helpers import safe_drop_last

# Fixed seed for the validation permutation (see process()), so repeated runs
# on the same dataset stay comparable.
VALID_PERM_SEED = 1234


def _to_4d_float(x):
    """numpy/array -> float32 tensor of shape [N, C, H, W].

    `torch.as_tensor` refuses several array layouts that are perfectly normal
    on disk and would otherwise propagate straight out of process() to
    main.py, failing the dataset for a score of -10:
      * negative strides - any array saved from a flipped or transposed view
        without .copy() (`ValueError: at least one stride is negative`)
      * non-native byte order, or a dtype torch has no equivalent for
    Going through np.ascontiguousarray + astype(float32) normalises all of
    those in one step, and costs nothing when the input is already a
    contiguous float32 array (both are no-ops then). The local datasets are
    all contiguous float32, so this path is invisible here - it exists purely
    for the unseen ones.
    """
    a = np.asarray(x)
    try:
        t = torch.as_tensor(a).float()
    except (ValueError, TypeError):
        t = torch.as_tensor(np.ascontiguousarray(a).astype(np.float32))
    if t.dim() == 3:                       # [N, H, W] -> [N, 1, H, W]
        t = t.unsqueeze(1)
    elif t.dim() == 2:                     # [N, F] -> [N, 1, 1, F]
        t = t.unsqueeze(1).unsqueeze(1)
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
    """Dataset over an in-memory array, with an optional fixed index order.

    `order` exists so the validation split can be served through one fixed
    seeded permutation without copying the array - see process() for why a
    truncated validation pass over an unpermuted split is actively misleading.
    `y=None` marks the test split, where __getitem__ yields the image alone.
    """

    def __init__(self, x, y, transform=None, order=None):
        self.x = _to_4d_float(x)
        self.y = None if y is None else torch.as_tensor(np.asarray(y)).long()
        self.transform = transform
        # Optional fixed index permutation (see process()). Applied lazily so
        # we get a reordered view without copying the underlying array.
        self.order = order

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.order is not None:
            idx = int(self.order[idx])
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
        """Best-effort query of GPU memory this dataset can actually use, in MB.
        Returns None if there is no GPU or the query fails for any reason
        (e.g. older CUDA driver) - callers must treat None as 'unknown'.

        The harness runs EVERY dataset inside one process (main.py loops over
        them), so when dataset 2 or 3 is sized, dataset 1's model, optimizer
        state and activations are long dead in Python but still held by
        PyTorch's caching allocator. `torch.cuda.mem_get_info()` reports
        DRIVER-free memory, which does not count that pool as free - measured
        on this box: 0 MB "free" with 7878 MB reserved and only 16 MB actually
        live. The batch size then collapsed to the `max(4, ...)` floor.

        That is not hypothetical: in the 2026-07-27 13-dataset run it hit both
        datasets that followed a memory-heavy one (Cryptic after Caitie,
        Sokoto after Myopia), which got batch_size=4 and therefore 20 and 17
        epochs instead of several hundred. Phase 2/3 run three datasets in one
        process, so two of the three are exposed.

        So: release the cache first, and count anything still reserved-but-
        unallocated as available, since the allocator will reuse it."""
        if not torch.cuda.is_available():
            return None
        try:
            torch.cuda.empty_cache()
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            reserved_unused = (torch.cuda.memory_reserved()
                               - torch.cuda.memory_allocated())
            return (free_bytes + max(0, reserved_unused)) / (1024 ** 2)
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
        # Floor of 2, never 1: BatchNorm raises on a single sample in train
        # mode, so the Trainer skips every batch shorter than 2 - a loader of
        # 1-sample batches would therefore take ZERO gradient steps and train
        # nothing, silently. Reachable via n_train // 2 for n_train in {2, 3}.
        # (Trainer._rebuild_train_loader already applies the same floor.)
        return max(2, bs)

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
            # randint, not randperm: randperm materialises a permutation of the
            # WHOLE tensor to draw max_check samples from it - measured 3.1s and
            # ~847 MB on AddNIST (about 84% of process() runtime), scaling
            # linearly with dataset size. Sampling with replacement gives the
            # same cardinality signal at essentially zero cost.
            idx = torch.randint(0, flat.numel(), (max_check,))
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
        """Build the three dataloaders.

        Wrapped so that NOTHING here can propagate to main.py: process() is the
        only pipeline stage with no harness-side protection, and any exception
        it raises fails the dataset outright for -10. _process() holds the real
        logic; _minimal_process() is a no-augmentation, no-normalisation last
        resort that only needs the arrays to be convertible at all."""
        try:
            return self._process()
        except Exception as e:
            print('[DataProcessor] process() failed ({}) - falling back to '
                  'minimal loaders'.format(repr(e)))
            return self._minimal_process()

    def _minimal_process(self):
        """Last-resort loaders: no normalisation stats, no augmentation, no
        cardinality probe, no memory query, fixed small batch. Those are the
        steps this rung actually removes, and any failure in them is what it
        rescues.

        HONEST LIMIT: it still calls _to_4d_float, so a genuinely
        unconvertible array fails here too - exactly the flaw that sank the
        reverted DataProcessor fallback ladder (CLAUDE.md 7b), where every
        rung called the same conversion. That case is addressed at the
        conversion site instead (see _to_4d_float); this wrapper is not a
        second line of defence for it and must not be mistaken for one. If an
        array cannot become a tensor at all, no loader can be built from it."""
        train_ds = _ArrayDataset(self.train_x, self.train_y)
        # The validation permutation is NOT an optimisation, it is a
        # correctness property (see _process): a deadline-truncated validation
        # pass otherwise scores a fixed PREFIX, which is meaningless on a split
        # that happens to be ordered by class. It applies on this rung too - the
        # Trainer and NAS truncate validation regardless of how the loaders were
        # built. Guarded because this rung must not fail.
        valid_order = None
        try:
            # shape[0] only - do NOT build an _ArrayDataset just to read a
            # length, that materialises the whole split as float32
            valid_order = torch.randperm(
                int(np.asarray(self.valid_x).shape[0]),
                generator=torch.Generator().manual_seed(VALID_PERM_SEED))
        except Exception:
            pass
        valid_ds = _ArrayDataset(self.valid_x, self.valid_y, order=valid_order)
        test_ds = _ArrayDataset(self.test_x, None)
        bs = 32
        # Best-effort majority label; predict() pads with it as a last resort.
        try:
            labels = np.asarray(self.train_y).reshape(-1)
            self.metadata['fallback_label'] = int(np.bincount(labels).argmax())
        except Exception:
            pass
        self.metadata.setdefault('fallback_label', 0)
        self.metadata['batch_size'] = bs
        mk = torch.utils.data.DataLoader
        return (mk(train_ds, batch_size=bs, shuffle=True,
                   drop_last=safe_drop_last(len(train_ds), bs)),
                mk(valid_ds, batch_size=bs, shuffle=False, drop_last=False),
                mk(test_ds, batch_size=bs, shuffle=False, drop_last=False))

    def _process(self):
        """The real loader construction; process() wraps this so nothing escapes.

        Order matters here: normalisation statistics come from the TRAINING split
        only, geometric augmentation runs on raw values (scale-independent) while
        Gaussian noise runs after Normalize (so its magnitude means the same thing
        on every dataset), and augmentation is applied to the train split alone -
        valid and test must stay a clean, deterministic measure.
        """
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

        # The valid loader stays shuffle=False so epoch-to-epoch numbers remain
        # comparable - but both Trainer._evaluate and NAS._quick_val_acc stop a
        # validation pass on a deadline, and an unshuffled loader means they
        # then score a fixed PREFIX of the split. If an unseen dataset ships its
        # validation set sorted by class, that prefix is one or two classes and
        # the accuracy is meaningless: it would pick the wrong checkpoint in the
        # Trainer and the wrong genotype in NAS. Serving the split through one
        # fixed random permutation makes any prefix a uniform sample instead.
        # The TEST split is deliberately NOT permuted - the harness matches
        # predictions to labels by position.
        n_valid = int(np.asarray(self.valid_x).shape[0])
        valid_order = torch.randperm(
            n_valid, generator=torch.Generator().manual_seed(VALID_PERM_SEED))

        # train_t is a full float32 copy of the training split and is no longer
        # needed once the stats and the transform are decided. _ArrayDataset
        # below builds its own copy, so leaving this one alive means holding
        # two concurrently: free for float32 input, but a 4x double-copy for
        # uint8, which is the normal on-disk format for images.
        del train_t

        train_ds = _ArrayDataset(self.train_x, self.train_y, transform=train_transform)
        valid_ds = _ArrayDataset(self.valid_x, self.valid_y, transform=normalize,
                                 order=valid_order)
        test_ds = _ArrayDataset(self.test_x, None, transform=normalize)

        bs = self._choose_batch_size(c, h, w, n_train)
        # never leave a final training batch of one sample - BatchNorm would
        # raise and silently end training for this dataset
        drop_last = safe_drop_last(n_train, bs)

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
