"""
NAS - Schritt 1 (Fallback-first Baseline)

In diesem Schritt gibt es noch KEINE Architektursuche. search() liefert ein
bekannt gutes, robustes Modell zurueck:
  * adaptiertes ResNet18 (Stem an Kanalzahl angepasst, kein Download -> offline ok)
  * bei kleinen Inputs kein aggressives Downsampling (MaxPool raus),
    damit winzige Grids (z.B. 8x8) nicht "weggepoolt" werden
  * schlaegt der Aufbau fehl -> TinyNet, das fuer beliebige C/H/W funktioniert

Die eigentliche Suche (Zell-Suchraum + Proxies) kommt in Schritt 3.
"""

import torch
import torch.nn as nn

try:
    import torchvision
    _HAS_TV = True
except Exception:
    _HAS_TV = False


class _TinyNet(nn.Module):
    """Ultimativer Fallback: kleines CNN, laeuft fuer jede Eingabeform."""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_baseline_model(in_ch, num_classes, h, w):
    """Adaptiertes ResNet18; Fallback auf TinyNet, falls torchvision fehlt."""
    if not _HAS_TV:
        return _TinyNet(in_ch, num_classes)
    model = torchvision.models.resnet18()  # pretrained=False (Default) -> kein Netzwerkzugriff
    # Stem an Kanalzahl anpassen, stride 1 statt 2 (Inputs sind meist klein)
    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Bei kleinen Bildern MaxPool entfernen, sonst geht zu viel Aufloesung verloren
    if min(int(h), int(w)) <= 32:
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

    def _infer_chw(self):
        """C/H/W direkt aus einem echten Batch lesen (robuster als Metadata)."""
        try:
            xb = next(iter(self.train_loader))
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            return int(xb.shape[1]), int(xb.shape[2]), int(xb.shape[3])
        except Exception:
            s = self.metadata['input_shape']
            return int(s[1]), int(s[2]), int(s[3])

    def search(self):
        num_classes = int(self.metadata['num_classes'])
        in_ch, h, w = self._infer_chw()
        try:
            return build_baseline_model(in_ch, num_classes, h, w)
        except Exception as e:
            print("[NAS] Fallback auf TinyNet wegen:", repr(e))
            return _TinyNet(in_ch, num_classes)
