"""
model.py - Schritt 2

Adaptives Makro-Geruest, dessen Struktur aus der Eingabegroesse (H, W) abgeleitet
wird. Der Block ('ResidualBlock') ist vorerst fest - in Schritt 3 wird genau
dieser Block durch die gesuchte Zelle ersetzt, das Geruest bleibt gleich.

TinyNet dient weiterhin als ultimativer Fallback.
"""

import math
import torch
import torch.nn as nn


def derive_macro(h, w, s_min=4, d_max=3, c0=32, n=2):
    """
    Makro-Struktur aus der Eingabegroesse ableiten.
      D = clamp(floor(log2(min(H,W)/s_min)), 0, d_max)
    Rueckgabe: (c0, n, d) = Startkanaele, Bloecke pro Stage, Anzahl Downsamples.
    """
    m = max(1, min(int(h), int(w)))
    d = int(math.floor(math.log2(m / s_min))) if m > s_min else 0
    d = max(0, min(d_max, d))
    return c0, n, d


class ResidualBlock(nn.Module):
    """Fester Residual-Block (Platzhalter fuer die spaeter gesuchte Zelle)."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.proj = None
        if stride != 1 or cin != cout:               # Dimensionen fuer den Skip angleichen
            self.proj = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.proj is None else self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class Skeleton(nn.Module):
    """Stem -> (d+1) Stages mit je n Bloecken -> Global-Pool -> Linear."""
    def __init__(self, in_ch, num_classes, c0, n, d):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c0, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c0), nn.ReLU(inplace=True))

        blocks = []
        cin = c0
        for i in range(d + 1):                        # i=0 ohne Downsample
            cout = c0 * (2 ** i)
            for b in range(n):
                stride = 2 if (b == 0 and i > 0) else 1   # Downsample am Stage-Beginn
                blocks.append(ResidualBlock(cin, cout, stride=stride))
                cin = cout
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TinyNet(nn.Module):
    """Ultimativer Fallback: laeuft fuer beliebige C/H/W."""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
