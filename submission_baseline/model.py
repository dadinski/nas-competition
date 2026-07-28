"""
model.py - Step 2 (adaptive skeleton) + Step 3 (searchable cell)

Two building blocks:
  * Skeleton: adaptive macro-architecture, structure derived from input size.
    Accepts any block class that maps (cin, cout, stride) -> nn.Module.
  * Cell: a NAS-Bench-201-style searchable block, built from a genotype
    (one op choice per edge in a small DAG). Used by Skeleton once search
    has picked a genotype; ResidualBlock remains as a simple fixed fallback
    block.

TinyNet is the ultimate fallback network (works for any C/H/W).
"""

import itertools
import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Macro-architecture sizing
# ---------------------------------------------------------------------------

def derive_macro(h, w, s_min=4, d_max=3, c0=32, n=2):
    """
    Derive the macro structure from the input size.
      D_full = floor(log2(min(H,W)/s_min))        # reduction needed to reach s_min
      D      = clamp(D_full, 0, d_max)            # how much the STAGES provide
      stem   = 2 ** (D_full - D)                  # the rest, taken in the stem

    Returns (c0, n, d, stem_stride).

    Why the stem stride exists: d alone is capped at d_max, so the total
    downsampling used to be at most 8x no matter how large the input was. A
    128x128 dataset therefore ran its entire network at 128/64/32/16
    resolution and cost ~28x as much per sample as a 28x28 one (analytic
    conv FLOPs: 7287 vs 262 MFLOPs/sample for the heaviest cell). In the
    2026-07-27 run that showed up exactly as predicted - Myofibre took 525s
    per epoch versus AddNIST's 14.6s, completed FIVE epochs in its hour, was
    still improving steeply when the clock ran out, and scored -3.96.

    Taking the surplus reduction in the stem instead makes per-sample cost
    roughly independent of input resolution (all inputs land at ~1.7-1.8x
    the 28x28 cost) and is simply what standard CNNs already do - ResNet
    drops 224x224 to 56x56 before its first stage. Note this makes us *more*
    conventional, not less: total reduction is now 32x, the ImageNet norm,
    where before it was 8x.

    Inputs at or below 32x32 get stem_stride == 1, i.e. they are completely
    unaffected by this - which covers most of the local datasets and is why
    this change cannot regress them.
    """
    m = max(1, min(int(h), int(w)))
    d_full = int(math.floor(math.log2(m / s_min))) if m > s_min else 0
    d = max(0, min(d_max, d_full))
    stem_stride = 2 ** max(0, d_full - d)
    return c0, n, d, stem_stride


# ---------------------------------------------------------------------------
# NAS-Bench-201-style cell (Step 3)
# ---------------------------------------------------------------------------

# 4 nodes, 6 edges (fully connected DAG on 4 nodes), 5 candidate ops per edge.
NUM_NODES = 4
EDGES = [(i, j) for j in range(1, NUM_NODES) for i in range(j)]   # 6 edges
OPS = ['conv3x3', 'conv1x1', 'avgpool3x3', 'skip', 'none']


def _make_op(name, c):
    """Build a single-edge operation; stride is always 1 (Cell is stride-1),
    spatial downsampling happens in the stage's first block via a separate
    stride-conv wrapper (see Cell.__init__)."""
    if name == 'conv3x3':
        return nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
    if name == 'conv1x1':
        return nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
    if name == 'avgpool3x3':
        return nn.AvgPool2d(3, stride=1, padding=1)
    if name == 'skip':
        return nn.Identity()
    if name == 'none':
        return _Zero()
    raise ValueError('unknown op: ' + name)


class _Zero(nn.Module):
    """The 'none' op: outputs zeros (edge effectively disconnected)."""
    def forward(self, x):
        return x * 0.0


class Cell(nn.Module):
    """
    Searchable NAS-Bench-201 style cell.
    genotype: list of op names, one per edge in EDGES (len == 6).
    Optionally changes channels/stride once at the cell input (stem-like proj),
    all internal edges then operate at (cout, stride=1).
    """
    def __init__(self, cin, cout, stride, genotype):
        super().__init__()
        assert len(genotype) == len(EDGES)
        self.pre = None
        if stride != 1 or cin != cout:
            self.pre = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout))
        self.edge_ops = nn.ModuleDict({
            '{}_{}'.format(i, j): _make_op(op, cout)
            for (i, j), op in zip(EDGES, genotype)
        })

    def forward(self, x):
        x0 = self.pre(x) if self.pre is not None else x
        nodes = [x0]
        for j in range(1, NUM_NODES):
            acc = None
            for i in range(j):
                e = self.edge_ops['{}_{}'.format(i, j)](nodes[i])
                acc = e if acc is None else acc + e
            nodes.append(acc)
        return nodes[-1]


def all_genotypes():
    """Every possible genotype: one op per edge, all combinations.
    len(OPS) ** len(EDGES) total (5**6 = 15625 with the current op/edge counts)."""
    return [list(combo) for combo in itertools.product(OPS, repeat=len(EDGES))]


def is_degenerate(genotype):
    """A cell that is (almost) only 'none'/'skip' carries no real signal."""
    useful = sum(1 for op in genotype if op not in ('none', 'skip'))
    return useful == 0


# ---------------------------------------------------------------------------
# Fixed fallback block (used before search / if search is skipped)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Fixed residual block, used as a safe default block for the Skeleton."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.proj = None
        if stride != 1 or cin != cout:               # match dims for the skip path
            self.proj = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.proj is None else self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


# ---------------------------------------------------------------------------
# Macro skeleton
# ---------------------------------------------------------------------------

class Skeleton(nn.Module):
    """Stem -> (d+1) stages of n blocks -> global pool -> linear.

    block_fn(cin, cout, stride) -> nn.Module must be supplied; this lets the
    same skeleton be built either with the fixed ResidualBlock or with a
    searched Cell genotype (see nas.py).
    """
    def __init__(self, in_ch, num_classes, c0, n, d, block_fn, stem_stride=1):
        super().__init__()
        # The stem carries whatever downsampling the d stages cannot (see
        # derive_macro). It is built as a stack of stride-2 3x3 convs rather
        # than one big-stride conv: a 3x3 kernel at stride 4 only looks at 9
        # of every 16 input pixels, which is aliasing - it throws information
        # away rather than pooling it. Successive stride-2 convs see every
        # pixel. The cost is negligible because the channel count is still
        # small here (for 128x128: ~13 MFLOPs against the ~455 of the body).
        stem_layers = []
        cin = in_ch
        n_halvings = max(0, int(math.log2(max(1, int(stem_stride)))))
        for _ in range(n_halvings):
            stem_layers += [nn.Conv2d(cin, c0, 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c0), nn.ReLU(inplace=True)]
            cin = c0
        stem_layers += [nn.Conv2d(cin, c0, 3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(c0), nn.ReLU(inplace=True)]
        self.stem = nn.Sequential(*stem_layers)

        blocks = []
        cin = c0
        for i in range(d + 1):                        # i=0 has no downsample
            cout = c0 * (2 ** i)
            for b in range(n):
                stride = 2 if (b == 0 and i > 0) else 1   # downsample at stage start
                blocks.append(block_fn(cin, cout, stride))
                cin = cout
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_skeleton(in_ch, num_classes, c0, n, d, genotype=None, stem_stride=1):
    """Convenience factory: fixed ResidualBlock if genotype is None,
    else a Skeleton made of searched Cells sharing that genotype.

    stem_stride must be the value derive_macro returned for this input size -
    passing the default 1 for a large input silently restores the old
    resolution-blind cost profile."""
    if genotype is None:
        block_fn = lambda cin, cout, stride: ResidualBlock(cin, cout, stride)
    else:
        block_fn = lambda cin, cout, stride: Cell(cin, cout, stride, genotype)
    return Skeleton(in_ch, num_classes, c0, n, d, block_fn, stem_stride=stem_stride)


class TinyNet(nn.Module):
    """Fallback network, used when the searched/fixed Skeleton doesn't fit
    in memory even at its smallest size.

    The old version ran two conv layers at FULL input resolution with no
    downsampling - on a large image that can need more memory than the
    already-minimal Skeleton it was replacing, defeating the point of a
    'safe' fallback. This version adds up to two stride-2 downsampling
    steps (reusing the same s_min=4 floor as derive_macro) so its memory
    footprint actually shrinks with a shrinking Skeleton instead of
    staying flat, while still working for any C/H/W (h/w are optional -
    without them it behaves like the old flat, no-downsample version)."""
    def __init__(self, in_ch, num_classes, h=None, w=None, channels=32, s_min=4):
        super().__init__()
        n_down = 0
        if h is not None and w is not None:
            m = max(1, min(int(h), int(w)))
            if m > s_min:
                n_down = min(2, int(math.floor(math.log2(m / s_min))))

        layers = []
        cin, cout = in_ch, channels
        n_layers = max(1, n_down + 1)
        for i in range(n_layers):
            stride = 2 if i < n_down else 1
            layers += [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                       nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            cin = cout
        layers += [nn.Conv2d(cin, cout * 2, 3, padding=1, bias=False),
                   nn.BatchNorm2d(cout * 2), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cout * 2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class MinimalNet(nn.Module):
    """Absolute last-resort fallback, used only if even TinyNet doesn't
    fit in memory. No convolutions at all: the input is pooled straight
    down to one value per channel and fed to a linear classifier, so its
    memory footprint is essentially just the input batch itself,
    independent of H/W. This should fit under virtually any circumstance
    a single sample could be loaded at all."""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)
