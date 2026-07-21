"""
NAS - Schritt 2

search() baut jetzt ein adaptives Makro-Geruest (Skeleton), dessen Struktur aus
der Eingabegroesse abgeleitet wird, und stellt per Speicher-Probe sicher, dass
das Modell mit dem realen Batch mindestens einen Trainingsschritt schafft.
Bei OOM wird in fester Reihenfolge geschrumpft: C0 -> N -> D. Scheitert alles,
gibt es das TinyNet.

Noch keine echte Architektursuche - die kommt in Schritt 3 und ersetzt den
festen Block im Skeleton durch gesuchte Zellen.
"""

import torch
import torch.nn as nn

from model import Skeleton, TinyNet, derive_macro


def _is_oom(e):
    return isinstance(e, RuntimeError) and 'out of memory' in str(e).lower()


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

    def _sample_batch(self):
        xb, yb = next(iter(self.train_loader))
        return xb, yb

    def _fits(self, model, xb, yb, device):
        """Ein echter forward+backward als Speicher-Probe. True = passt."""
        try:
            model.to(device)
            opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            opt.zero_grad(set_to_none=True)
            out = model(xb.to(device))
            loss = nn.CrossEntropyLoss()(out, yb.to(device))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except RuntimeError as e:
            if _is_oom(e):
                try:
                    model.to('cpu')
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
            raise

    def search(self):
        num_classes = int(self.metadata['num_classes'])
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Realen Batch ziehen; klappt das nicht, direkt TinyNet aus der Metadata
        try:
            xb, yb = self._sample_batch()
            in_ch, h, w = int(xb.shape[1]), int(xb.shape[2]), int(xb.shape[3])
        except Exception as e:
            print('[NAS] Kein Batch verfuegbar, TinyNet:', repr(e))
            s = self.metadata['input_shape']
            return TinyNet(int(s[1]), num_classes)

        c0, n, d = derive_macro(h, w)
        print('[NAS] Start-Makro: c0={} n={} d={} (H={},W={})'.format(c0, n, d, h, w))

        try:
            while True:
                model = Skeleton(in_ch, num_classes, c0, n, d)
                if self._fits(model, xb, yb, device):
                    self.metadata['macro'] = {'c0': c0, 'n': n, 'd': d}
                    print('[NAS] Gewaehltes Makro: c0={} n={} d={}'.format(c0, n, d))
                    return model
                # Speicher zu knapp -> schrumpfen: erst C0, dann N, dann D
                if c0 > 8:
                    c0 = max(8, c0 // 2)
                elif n > 1:
                    n -= 1
                elif d > 0:
                    d -= 1
                else:
                    break   # schon minimal -> TinyNet
                print('[NAS] OOM bei Probe, schrumpfe auf c0={} n={} d={}'.format(c0, n, d))
        except Exception as e:
            print('[NAS] Aufbau fehlgeschlagen, TinyNet:', repr(e))

        return TinyNet(in_ch, num_classes)
