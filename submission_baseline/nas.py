"""
NAS - Step 3 (training-free search with NASWOT + SynFlow, short verification)

search() proceeds in three phases:
  1. Macro sizing: same OOM-safe probe/shrink loop as Step 2, using the fixed
     ResidualBlock skeleton, to determine (c0, n, d) that fits in memory.
  2. Cell search: iterate the full genotype search space (shuffled, so a
     time cutoff still samples it uniformly), discard degenerate ones,
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

from model import TinyNet, MinimalNet, build_skeleton, derive_macro, all_genotypes, is_degenerate
from proxies import naswot_score, synflow_score


def _is_oom(e):
    return isinstance(e, RuntimeError) and 'out of memory' in str(e).lower()


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

    def _fits_with_shrink(self, model, xb, yb, device, min_batch=8):
        """Like _fits, but on OOM retries with a halved batch instead of
        giving up immediately. Phase 1 only sizes (c0, n, d) against the
        fixed ResidualBlock, so a searched cell can legitimately need more
        memory per sample; this salvages that case instead of discarding
        the genotype outright. Returns (fits, working_batch_size)."""
        bs = xb.shape[0]
        while True:
            if self._fits(model, xb[:bs], yb[:bs], device):
                return True, bs
            if bs <= min_batch:
                return False, None
            bs = max(min_batch, bs // 2)

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

    def _quick_val_acc(self, model, device, max_batches, deadline=None, batch_size=None):
        """One short training pass over at most max_batches, then val accuracy.
        If deadline (perf_counter timestamp) is given, the validation part
        is also time-bounded so a slow/large valid set can't blow through
        the (small) per-dataset search budget.
        batch_size: if set, every loaded batch is sliced down to this size.
        Used when the loader's natural batch size is too large for this
        particular genotype (see _calibrate_verify_cost)."""
        model.to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        model.train()
        for bi, (data, target) in enumerate(self.train_loader):
            if bi >= max_batches:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if batch_size is not None:
                data, target = data[:batch_size], target[:batch_size]
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
                if batch_size is not None:
                    data, target = data[:batch_size], target[:batch_size]
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
        search_budget = max(0.0, 0.10 * total_budget)
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
            # ResidualBlock used for sizing, so re-verify it still fits.
            # Try shrinking the batch first instead of discarding the
            # genotype outright on the first OOM.
            fits, working_bs = self._fits_with_shrink(model, xb, yb, device)
            if not fits:
                print('[NAS] searched cell does not fit even at reduced batch size, falling back to fixed block')
                model = build_skeleton(in_ch, num_classes, c0, n, d, genotype=None)
                if not self._fits(model, xb, yb, device):
                    print('[NAS] even the minimal fixed-block skeleton does not fit')
                    return self._safe_fallback(in_ch, num_classes, h, w, xb, yb, device)
            elif working_bs < xb.shape[0]:
                # architecture is fine, just needs a smaller batch than the loader
                # default - record this so the trainer can pick it up (Step 5)
                print('[NAS] searched cell only fits at batch_size={} (loader default {}); '
                      'recording nas_batch_size_hint in metadata'.format(working_bs, xb.shape[0]))
                self.metadata['nas_batch_size_hint'] = working_bs
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

    def _calibrate_verify_cost(self, genotype, in_ch, num_classes, c0, n, d, device, min_batch=8):
        """Time one training batch and one full validation pass for `genotype`.
        Used to size the verify phase (how many candidates, how many batches
        each) to the time actually left, instead of hardcoded constants.

        The genotype can be heavier than the fixed ResidualBlock used for
        macro-sizing, so this can OOM at the loader's natural batch size
        even though the architecture itself is fine. On OOM we halve the
        batch and retry instead of aborting; the batch size that ends up
        working is returned so the caller can run the rest of verification
        at a consistent (safe) size. If even min_batch OOMs, we give up and
        let the caller fall back (that's a genuinely-too-heavy genotype).
        Returns (t_batch, t_val, batch_size_used)."""
        m = build_skeleton(in_ch, num_classes, c0, n, d, genotype=genotype).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
        crit = nn.CrossEntropyLoss()

        data, target = next(iter(self.train_loader))
        bs = data.shape[0]
        m.train()
        while True:
            try:
                d_try = data[:bs].to(device)
                t_try = target[:bs].to(device)
                t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                out = m(d_try)
                loss = crit(out, t_try)
                loss.backward()
                opt.step()
                t_batch = max(1e-4, time.perf_counter() - t0)
                break
            except RuntimeError as e:
                if _is_oom(e) and bs > min_batch:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    bs = max(min_batch, bs // 2)
                    continue
                raise   # not OOM, or already at min_batch -> let caller handle/log it

        m.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            for vd, _ in self.valid_loader:
                m(vd[:bs].to(device))
        t_val = max(1e-4, time.perf_counter() - t0)
        return t_batch, t_val, bs

    def _search_cell(self, in_ch, num_classes, c0, n, d, xb, device, search_budget):
        deadline = time.perf_counter() + search_budget
        xb_s = xb[:min(32, xb.shape[0])].to(device)

        # zero-cost proxies are cheap, so score the whole search space instead
        # of a random subsample; shuffle so a time cutoff still covers it
        # roughly uniformly rather than favoring one region.
        genotypes = all_genotypes()
        random.shuffle(genotypes)
        print('[NAS] scoring up to {} genotypes (full search space)'.format(len(genotypes)))

        # score candidates
        scored = []   # list of (naswot, synflow, genotype)
        counter = 0
        for geno in genotypes:
            if counter % 1000 == 0:
                print('[NAS] scored {} genotypes'.format(counter))
            if counter == 1000:
                break
            if time.perf_counter() >= deadline:
                break
            if is_degenerate(geno):
                continue
            try:
                m = build_skeleton(in_ch, num_classes, c0, n, d, genotype=geno).to(device)
                nw = naswot_score(m, xb_s)
                sf = synflow_score(m, in_ch, xb.shape[2], xb.shape[3], device)
                scored.append((nw, sf, geno))
                counter += 1
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

        top_geno = scored[order[0]][2]   # best by proxy rank alone; used if there's no time to verify

        if time.perf_counter() >= deadline:
            print('[NAS] no time left to verify, using top-ranked genotype by proxy score')
            return top_geno

        # calibrate training/validation cost on the top candidate so n_verify
        # and max_batches can be sized to the time actually remaining, rather
        # than being fixed constants.
        try:
            t_batch, t_val, verify_bs = self._calibrate_verify_cost(top_geno, in_ch, num_classes, c0, n, d, device)
        except RuntimeError as e:
            if _is_oom(e) and torch.cuda.is_available():
                torch.cuda.empty_cache()
            # print the real exception - "OOM even at min_batch" and "some
            # unrelated RuntimeError" need different fixes, don't hide which one it was
            print('[NAS] verify timing calibration failed ({}), using top-ranked genotype by proxy score'.format(repr(e)))
            return top_geno

        verify_budget = deadline - time.perf_counter()
        breadth_cap = 100    # ceiling on candidates: past this, more training depth per
                             # candidate is worth more than more candidates, so surplus
                             # time should go to max_batches, not n_verify
        min_batches = 20     # floor: every verified candidate gets at least this much training
        max_batches_cap = 2000   # safety valve only; the deadline check inside the training
                                  # loop is what actually bounds a candidate's runtime
        cost_per_candidate = t_val + min_batches * t_batch

        if verify_budget <= 0:
            print('[NAS] no time left to verify, using top-ranked genotype by proxy score')
            return top_geno

        n_verify = max(1, min(len(order), breadth_cap, int(verify_budget / cost_per_candidate)))
        leftover = verify_budget - n_verify * t_val
        max_batches = max(min_batches, min(max_batches_cap, int(leftover / (n_verify * t_batch))))

        shortlist = [scored[i][2] for i in order[:n_verify]]
        print('[NAS] scored {} genotypes, verifying top {} (max_batches={}, batch_size={}, t_batch~{:.3f}s, t_val~{:.2f}s)'
              .format(len(scored), n_verify, max_batches, verify_bs, t_batch, t_val))

        # short verification: a few batches of training each, best val acc wins
        best_geno, best_acc = None, -1.0
        for geno in shortlist:
            if time.perf_counter() >= deadline:
                break
            t_start = time.perf_counter()
            try:
                m = build_skeleton(in_ch, num_classes, c0, n, d, genotype=geno).to(device)
                # use the batch size calibration found safe for this genotype family,
                # not the loader's default, so we don't just OOM again per-candidate
                acc = self._quick_val_acc(m, device, max_batches=max_batches, deadline=deadline, batch_size=verify_bs)
            except RuntimeError as e:
                if _is_oom(e) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            if acc > best_acc:
                best_acc, best_geno = acc, geno
            print('  [NAS] verify: val={:.2%} t={:.1f}s'.format(acc, time.perf_counter() - t_start))

        if best_geno is None:
            print('[NAS] no candidate finished verification in time, using top-ranked genotype by proxy score')
            return top_geno

        print('[NAS] selected genotype (val={:.2%}): {}'.format(best_acc, best_geno))
        return best_geno
