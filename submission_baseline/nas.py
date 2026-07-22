"""
NAS - Step 3 (training-free search with NASWOT + SynFlow, short verification)

search() proceeds in three phases:
  1. Macro sizing: same OOM-safe probe/shrink loop as Step 2, using the fixed
     ResidualBlock skeleton, to determine (c0, n, d) that fits in memory.
  2. Cell search: sample K random genotypes (K derived from a measured proxy
     eval time and the remaining search budget), discard degenerate ones,
     score with NASWOT + SynFlow, combine via rank aggregation, verify the
     top-n with a few quick training epochs, and keep the best by
     validation accuracy.
  3. Fallback: if the search budget is too small, or anything fails, fall
     back to the fixed ResidualBlock skeleton (Step 2 baseline). TinyNet
     remains the last-resort fallback if even that doesn't fit.
"""

import random
import time

import torch
import torch.nn as nn

from model import TinyNet, MinimalNet, build_skeleton, derive_macro, EDGES, OPS, is_degenerate
from proxies import naswot_score, synflow_score


def _is_oom(e):
    return isinstance(e, RuntimeError) and 'out of memory' in str(e).lower()


def _random_genotype():
    return [random.choice(OPS) for _ in EDGES]


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

    def _remaining(self):
        try:
            return float(self.clock.check())
        except Exception:
            return 1e9

    def _fits(self, model, xb, yb, device):
        """One real forward+backward as a memory probe. True = fits."""
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

    def _find_macro(self, in_ch, num_classes, h, w, xb, yb, device):
        """OOM-safe macro sizing, using the fixed ResidualBlock as a stand-in."""
        c0, n, d = derive_macro(h, w)
        while True:
            model = build_skeleton(in_ch, num_classes, c0, n, d, genotype=None)
            if self._fits(model, xb, yb, device):
                return c0, n, d
            # too little memory -> shrink in order: C0, then N, then D
            if c0 > 8:
                c0 = max(8, c0 // 2)
            elif n > 1:
                n -= 1
            elif d > 0:
                d -= 1
            else:
                return c0, n, d   # already minimal

    def _quick_val_acc(self, model, device, max_batches, deadline=None):
        """One short training pass over at most max_batches, then val accuracy.
        If deadline (perf_counter timestamp) is given, the validation part
        is also time-bounded so a slow/large valid set can't blow through
        the (small) per-dataset search budget."""
        model.to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        model.train()
        for bi, (data, target) in enumerate(self.train_loader):
            if bi >= max_batches:
                break
            try:
                opt.zero_grad(set_to_none=True)
                out = model(data.to(device))
                loss = crit(out, target.to(device))
                loss.backward()
                opt.step()
            except RuntimeError as e:
                if _is_oom(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                try:
                    out = model(data.to(device))
                    pred = out.argmax(1).cpu()
                    correct += int((pred == target).sum())
                    total += target.numel()
                except RuntimeError as e:
                    if _is_oom(e) and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                if deadline is not None and time.perf_counter() >= deadline:
                    break
        return correct / total if total else 0.0

    def search(self):
        num_classes = int(self.metadata['num_classes'])
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        try:
            xb, yb = next(iter(self.train_loader))
            in_ch, h, w = int(xb.shape[1]), int(xb.shape[2]), int(xb.shape[3])
        except Exception as e:
            print('[NAS] no batch available, using TinyNet:', repr(e))
            s = self.metadata['input_shape']
            # no batch/device to fit-check against here - this is the one
            # path where we truly have nothing to probe with, so we just
            # return our best (size-adaptive) guess
            return TinyNet(int(s[1]), num_classes, h=int(s[2]), w=int(s[3]))

        # --- Phase 1: memory-safe macro sizing ---
        try:
            c0, n, d = self._find_macro(in_ch, num_classes, h, w, xb, yb, device)
            self.metadata['macro'] = {'c0': c0, 'n': n, 'd': d}
            print('[NAS] macro: c0={} n={} d={} (H={}, W={})'.format(c0, n, d, h, w))
        except Exception as e:
            print('[NAS] macro sizing failed, using TinyNet:', repr(e))
            return self._safe_fallback(in_ch, num_classes, h, w, xb, yb, device)

        # --- Phase 2: cell search (skipped if the budget is too small) ---
        total_budget = self._remaining()
        search_budget = max(0.0, min(0.10 * total_budget, 90.0))
        min_search_budget = 8.0   # below this, searching isn't worth it
        genotype = None          # None -> Skeleton uses the fixed ResidualBlock

        if search_budget >= min_search_budget:
            try:
                genotype = self._search_cell(in_ch, num_classes, c0, n, d, xb, device, search_budget)
            except Exception as e:
                print('[NAS] cell search failed, using fixed block:', repr(e))
                genotype = None
        else:
            print('[NAS] search budget too small ({:.1f}s), using fixed block'.format(search_budget))

        # --- Phase 3: build the final model ---
        try:
            model = build_skeleton(in_ch, num_classes, c0, n, d, genotype=genotype)
            # sanity check: a searched cell can be heavier/lighter than the
            # ResidualBlock used for sizing, so re-verify it still fits
            if not self._fits(model, xb, yb, device):
                print('[NAS] searched cell does not fit, falling back to fixed block')
                model = build_skeleton(in_ch, num_classes, c0, n, d, genotype=None)
                if not self._fits(model, xb, yb, device):
                    print('[NAS] even the minimal fixed-block skeleton does not fit')
                    return self._safe_fallback(in_ch, num_classes, h, w, xb, yb, device)
            return model
        except Exception as e:
            print('[NAS] final build failed, using TinyNet:', repr(e))
            return self._safe_fallback(in_ch, num_classes, h, w, xb, yb, device)

    def _safe_fallback(self, in_ch, num_classes, h, w, xb, yb, device):
        """Bottom of the fallback chain: TinyNet, fit-checked, and if even
        that doesn't fit, MinimalNet (near-zero memory, virtually always
        fits). Used whenever the searched/fixed Skeleton can't be used."""
        try:
            tiny = TinyNet(in_ch, num_classes, h=h, w=w)
            if self._fits(tiny, xb, yb, device):
                return tiny
            print('[NAS] TinyNet does not fit either, falling back to MinimalNet')
        except Exception as e:
            print('[NAS] TinyNet build/fit-check failed, falling back to MinimalNet:', repr(e))
        return MinimalNet(in_ch, num_classes)

    def _search_cell(self, in_ch, num_classes, c0, n, d, xb, device, search_budget):
        deadline = time.perf_counter() + search_budget
        xb_s = xb[:min(32, xb.shape[0])].to(device)

        # calibrate K from a measured proxy-evaluation time
        t0 = time.perf_counter()
        probe_model = build_skeleton(in_ch, num_classes, c0, n, d, genotype=_random_genotype()).to(device)
        naswot_score(probe_model, xb_s)
        synflow_score(probe_model, in_ch, xb.shape[2], xb.shape[3], device)
        t_eval = max(1e-3, time.perf_counter() - t0)
        k_budget = max(0.0, deadline - time.perf_counter())
        k = int(0.7 * k_budget / t_eval)
        k = max(20, min(k, 300))
        print('[NAS] sampling K={} genotypes (t_eval~{:.2f}s)'.format(k, t_eval))

        # sample and score candidates
        scored = []   # list of (naswot, synflow, genotype)
        tries = 0
        while len(scored) < k and tries < k * 3 and time.perf_counter() < deadline:
            tries += 1
            geno = _random_genotype()
            if is_degenerate(geno):
                continue
            try:
                m = build_skeleton(in_ch, num_classes, c0, n, d, genotype=geno).to(device)
                nw = naswot_score(m, xb_s)
                sf = synflow_score(m, in_ch, xb.shape[2], xb.shape[3], device)
                scored.append((nw, sf, geno))
            except RuntimeError as e:
                if _is_oom(e) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        if not scored:
            print('[NAS] no valid genotypes scored, using fixed block')
            return None

        # rank aggregation (scale-invariant, robust to outliers): rank 0 = best
        by_naswot = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        by_synflow = sorted(range(len(scored)), key=lambda i: -scored[i][1])
        rank_nw = {idx: r for r, idx in enumerate(by_naswot)}
        rank_sf = {idx: r for r, idx in enumerate(by_synflow)}
        order = sorted(range(len(scored)), key=lambda i: rank_nw[i] + rank_sf[i])

        n_verify = min(10, len(order))
        shortlist = [scored[i][2] for i in order[:n_verify]]
        print('[NAS] scored {} genotypes, verifying top {}'.format(len(scored), n_verify))

        # short verification: a few batches of training each, best val acc wins
        best_geno, best_acc = None, -1.0
        for geno in shortlist:
            if time.perf_counter() >= deadline:
                break
            t_start = time.perf_counter()
            try:
                m = build_skeleton(in_ch, num_classes, c0, n, d, genotype=geno).to(device)
                acc = self._quick_val_acc(m, device, max_batches=20, deadline=deadline)
            except RuntimeError as e:
                if _is_oom(e) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            if acc > best_acc:
                best_acc, best_geno = acc, geno
            print('  [NAS] verify: val={:.2%} t={:.1f}s'.format(acc, time.perf_counter() - t_start))

        if best_geno is not None:
            print('[NAS] selected genotype (val={:.2%}): {}'.format(best_acc, best_geno))
        return best_geno
