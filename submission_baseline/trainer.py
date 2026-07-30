"""
Trainer - Step 1 (fallback-first baseline) + Step 2 (OOM-safe runtime)

Guiding principle: NEVER trigger a -10 outcome (timeout / crash / missing
predictions). Therefore:
  * Training is clock-budgeted: a reserve for predict() is held back and
    training stops in time
  * The best validation checkpoint is kept and restored at the end
  * OOM batches are caught (clear cache, skip batch); persistent OOM shrinks
    the batch size and rebuilds the loader instead of crashing
  * predict() ALWAYS returns exactly n_test predictions in order - filled
    with the most frequent training class as a last resort, and halves the
    batch recursively on OOM
  * The cosine LR schedule is driven by the CLOCK, not by an estimated epoch
    count, so a cut-short epoch can no longer corrupt it (see train())
  * The batch size hint recorded by NAS is applied up front, so the first
    epoch doesn't have to discover the same OOM all over again
  * No training batch of size 1 ever reaches BatchNorm (see helpers.safe_drop_last)

Training policy: label smoothing, and weight decay on matrix-valued
parameters only.

LOGGING: every epoch line carries train accuracy, the train-val gap and the mean
train loss alongside validation accuracy, and train() ends with a one-line
diagnosis (see _log_training_summary). This exists because the 2026-07-27
13-dataset run logged ONLY validation accuracy, which made its single largest
problem invisible: 5 of 13 datasets reached their best checkpoint in the first
few percent of the budget and had already memorised the training split, and that
could only be established by re-running them by hand afterwards.

** The logged train accuracy is a RUNNING figure, not a clean one. ** It is
accumulated from the forward passes the training step already performs, so it is
measured on AUGMENTED batches, in train() mode (BatchNorm using batch statistics),
with the weights changing underneath it across the epoch. A clean eval-mode pass
over unaugmented training data would need a second traversal of the training set
and roughly double the cost of an epoch; this is free (measured: 7.4s/epoch
against a 7.2s baseline on Chesseract, i.e. inside the noise).

How far apart the two are depends entirely on how strong the augmentation is:
on Chesseract, whose policy is noise-only, the running figure hit 99.98% against
a clean 99.99% - indistinguishable. On a `continuous` dataset getting
pad+crop+flip the running figure will read materially lower, and it lags within
an epoch because early batches are scored by weaker weights. So: read it as a
trend and as a gap, do NOT read a few points of difference from a clean
diagnostic number as a real change.
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from helpers import safe_drop_last


def _select_amp():
    """Bind the AMP API this runtime actually supports, warning-free.

    `torch.cuda.amp.autocast/GradScaler` work everywhere we might land but emit
    a FutureWarning on torch >= 2.4 (several lines per epoch - it swamps the
    training log). The modern spelling, `torch.amp.*` with an explicit device
    string, does not exist on older torch: `torch.amp.GradScaler(device)`
    arrived well after 1.10, which the starter kit pins and which the
    evaluation server may still run (see CLAUDE.md 3 - requirements.txt is NOT
    shipped, so we cannot assume a version).

    So probe once at import: actually construct both objects and enter the
    context manager, with warnings escalated to errors. Only if that is clean
    do we adopt the modern spelling; anything at all wrong falls back to the
    legacy one. This keeps a torch-1.10 server working while giving a modern
    one a clean log.

    Returns (autocast, GradScaler) adapters taking a single `enabled` kwarg,
    so the call sites stay identical either way.
    """
    try:
        import warnings
        from torch.amp import autocast as _ac, GradScaler as _gs
        with warnings.catch_warnings():
            warnings.simplefilter('error')      # any deprecation -> fall back
            _gs('cuda', enabled=False)
            with _ac('cuda', enabled=False):
                pass

        def autocast(enabled=True):
            return _ac('cuda', enabled=enabled)

        def GradScaler(enabled=True):
            return _gs('cuda', enabled=enabled)

        return autocast, GradScaler
    except Exception:
        from torch.cuda.amp import autocast as _ac, GradScaler as _gs
        return _ac, _gs


autocast, GradScaler = _select_amp()

try:
    from sklearn.metrics import accuracy_score
    def _acc(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
except Exception:
    def _acc(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float((y_true == y_pred).mean()) if len(y_true) else 0.0


MAX_EPOCHS = 5000

BASE_LR = 0.01
WEIGHT_DECAY = 3e-4
LABEL_SMOOTHING = 0.1

# --- Ensembling on surplus budget (member lifecycle is inline in train()) -------------------
# Some datasets reach their best checkpoint in the first few percent of the
# budget and gain nothing from the rest: measured best-val at epoch 4 of 393
# (Chesseract) and 9 of 341 (Language), both at ~100% train accuracy. Extra time
# cannot help those, and extra capacity would make them worse. Training several
# INDEPENDENT models and averaging their predictions can, and it scales with
# whatever budget we are given - which matters because the per-dataset budget is
# set by the organisers and not disclosed.
#
# Measured (ab_ensemble.py, identical 60-epoch budget for every arm, 4 members):
#   Chesseract  single 54.27%  ->  58.62%   (benchmark 57.83, first time above)
#   Language    single 80.10%  ->  85.47%   (benchmark 85.20, first time above)
#
# MEMBERS MUST BE INDEPENDENTLY RE-INITIALISED. Warm-restart snapshots - the
# textbook SGDR/"Snapshot Ensembles" recipe - gave NOTHING on both datasets
# (54.08% and 79.75%, i.e. slightly worse than a single model, with an ensemble
# gain over their own best member of -0.10 and +0.01 points): a model that has
# memorised its training split just re-memorises after a restart, so every
# member ends up computing the same function. The individual members here are
# no better than a single model either - the entire gain is decorrelation.
MAX_MEMBERS = 24           # caps host RAM for snapshots and predict() cost
MIN_PATIENCE = 10          # floor on "no improvement" epochs before declaring saturation
PATIENCE_FRACTION = 0.50   # ...and it also scales with how long the member has run
MEMBER_LENGTH_FACTOR = 1.5  # a member gets this multiple of the time member 1 needed to converge


def _saturation_patience(epochs_in_member):
    """Epochs without a validation improvement before a member counts as done.

    RELATIVE to how long the member has already trained, so the condition reads:
    fire once the best checkpoint sits in the first (1 - FRACTION) = 50% of what
    has run - a statement about the shape of the curve rather than a bare
    constant.

    This value has moved twice and BOTH moves were driven by measurement, so the
    history matters more than the number:

    1. It was 0.2, which was too eager. On AddNIST it fired on a noisy plateau
       (84.97% at epoch 43, then ten epochs oscillating 79.8-84.0 *while train
       accuracy climbed 91 -> 94%*), split the budget and cost a measured -3.53
       (CLAUDE.md 7j). The argmax of a noisy validation curve is not evidence of
       convergence.
    2. 0.75 fixed that but over-corrected, and the evidence is the strongest we
       have: the competition ORGANISER ran our submission on their own test
       datasets (CLAUDE.md 7l). Theirs are ~10x smaller than ours (4,500-5,300
       training samples, 6-minute budgets) and ALL THREE memorised completely -
       100% train accuracy, val gaps of +67.7 / +32.0 / +11.3 - which is exactly
       the profile ensembling exists for. At 0.75 the gate fired on NONE of them.
       Ganges spent 359 of its 513 epochs, 70% of its budget, at 100% train
       accuracy with zero validation improvement, and we declined to ensemble.
       On a small dataset the validation split is small, so noise keeps nudging
       the running max and resetting the drought counter even when the model is
       demonstrably finished - the statistic simply does not transfer between
       dataset sizes, and 0.75 was tuned on our 45k-sample locals.

    0.50 is measured, not reasoned: Gutenberg subsampled to exactly 1/10 (4,500
    train / 1,500 valid - a near-replica of the organiser's Ganges, same 1x27x18
    shape and 6 classes) scored on its real 6,000-sample test split at a 360s
    budget, genotype fixed across arms so search variance could not contaminate
    it. 0.75 -> 31.63% / 31.30% test; 0.50 -> 33.28% / 32.17%, i.e. +1.26 points
    on average with the same sign on both seeds. 0.35 is indistinguishable from
    0.50, so there is no reason to go lower.

    What it costs locally is one dataset: replaying both thresholds over the
    clean 2026-07-27 curves, only conway changes (to a false positive), measured
    at -0.046. All five genuine saturators still fire and fire much EARLIER
    (Gutenberg ep 97->49, Language 33->19, Windspeed 113->19), which also
    recovers most of the 3-31% of budget that detection was consuming.

    TWO STANDING WARNINGS for anyone changing this again:
      * Replay against real per-epoch TRAJECTORIES. The original 0.2 rule was
        "validated" against (best_epoch, total_epochs) summary pairs, which
        silently assumes the running best equals the final best. It does not,
        and that is why the gate misfired in production.
      * Weight the SMALL-data regime. The organiser's datasets, not ours, are
        the ones that resemble what gets scored.
    """
    return max(MIN_PATIENCE, int(PATIENCE_FRACTION * max(0, epochs_in_member)))


def _is_oom(err):
    return isinstance(err, RuntimeError) and 'out of memory' in str(err).lower()


class Trainer:
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock

        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer(model)
        self.fallback_label = int(metadata.get('fallback_label', 0))

        # Ensembling state. Empty/one-element means predict() takes exactly the
        # single-model path it always has - which is what happens on every
        # dataset that does not saturate (see _saturation_patience).
        self.ensemble_states = []
        self._last_eval_time = 0.0            # cost of one validation pass
        self._best_state_for_fallback = None  # single best weights, for fallback

        # AMP: FP16 compute + GradScaler. GradScaler(enabled=False) and
        # autocast(enabled=False) are documented no-ops, so this same code
        # runs unchanged on CPU-only boxes or GPUs without AMP benefit.
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        # NAS records nas_batch_size_hint when the searched cell only fits at
        # a smaller batch than the loader's default (see nas.py). Ignoring it
        # meant the first epoch was GUARANTEED to walk into the persistent-OOM
        # guard below, which aborts that epoch part-way through - wasting the
        # epoch and (before the clock-driven schedule) corrupting the LR
        # schedule's epoch estimate. Apply it before training starts instead.
        self._apply_nas_batch_size_hint()

    def _build_criterion(self):
        """CrossEntropyLoss with label smoothing where the runtime supports it.
        `label_smoothing` arrived in torch 1.10 and the evaluation server's
        version is unknown to us (see CLAUDE.md 3), so degrade rather than
        crash on an older one."""
        try:
            return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        except TypeError:
            print('[Trainer] label_smoothing unsupported by this torch, '
                  'falling back to plain CrossEntropyLoss')
            return nn.CrossEntropyLoss()

    def _build_optimizer(self, model, quiet=False):
        """SGD with weight decay applied to matrix-valued parameters only.
        BatchNorm affine terms and biases are 1-D; decaying them pulls the
        normalisation's learned scale and shift toward zero, which costs
        accuracy while regularising essentially nothing. Splitting on
        dimensionality needs no module introspection and so cannot miss a
        custom block in the searched cell."""
        decay, no_decay = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() > 1 else no_decay).append(p)
        groups = [{'params': decay, 'weight_decay': WEIGHT_DECAY},
                  {'params': no_decay, 'weight_decay': 0.0}]
        if not quiet:
            print('[Trainer] weight decay on {} tensors, none on {} (BN/bias)'.format(
                len(decay), len(no_decay)))
        return optim.SGD(groups, lr=BASE_LR, momentum=0.9)

    def _apply_nas_batch_size_hint(self):
        """Shrink the train loader up front if NAS found the model only fits
        at a smaller batch size than the DataProcessor chose."""
        try:
            hint = self.metadata.get('nas_batch_size_hint')
            current = self.train_dataloader.batch_size
            if hint is None or current is None or int(hint) >= int(current):
                return
            new_bs = self._rebuild_train_loader(int(hint))
            print("[Trainer] applying nas_batch_size_hint: train batch size {} -> {}".format(
                int(current), new_bs))
        except Exception as e:
            # a bad hint must never stop training - the runtime OOM guard
            # in train() remains as the safety net
            print("[Trainer] could not apply nas_batch_size_hint:", repr(e))

    def _remaining(self):
        try:
            return float(self.clock.check())
        except Exception:
            return 1e9   # clock unavailable -> don't abort artificially

    def _snapshot(self):
        """Copy the current weights to HOST memory. Snapshots must not sit on the
        GPU: several of them plus the live model is exactly the resident-memory
        pattern that made the reverted SWA attempt a plausible -10 (CLAUDE.md 7d).
        A 9.5M-parameter model is ~38 MB here, so MAX_MEMBERS of them is ~300 MB
        of ordinary RAM."""
        return {k: v.detach().to('cpu', copy=True)
                for k, v in self.model.state_dict().items()}

    def _reinit_model(self):
        """Fresh random weights for the SAME architecture, in place.

        Independent initialisation is the whole mechanism (see MAX_MEMBERS
        above) - re-loading the member's own starting weights, or carrying
        weights over across a restart, both produce members that agree with each
        other and gain nothing when averaged.

        Walks the module tree calling reset_parameters() wherever torch provides
        it (Conv2d, Linear, BatchNorm2d - which is everything parameterised in
        the search space). A custom block without it would simply keep its
        weights, which costs diversity but cannot break anything. The optimizer
        is rebuilt too, so momentum does not leak across members.
        """
        n = 0
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                try:
                    m.reset_parameters()
                    n += 1
                except Exception:
                    pass
        self.optimizer = self._build_optimizer(self.model, quiet=True)
        self.scaler = GradScaler(enabled=self.use_amp)
        return n

    def _rebuild_train_loader(self, bs):
        """Rebuild the train loader at batch size `bs`, keeping the
        shuffle/drop_last semantics the DataProcessor used."""
        # floor of 2, not 1: a full loader of single-sample batches would make
        # BatchNorm raise on every batch (see helpers.safe_drop_last)
        bs = max(2, int(bs))
        ds = self.train_dataloader.dataset
        drop_last = safe_drop_last(len(ds), bs)
        self.train_dataloader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, drop_last=drop_last)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return bs

    def _shrink_train_loader(self):
        """Halve the batch size and rebuild the train loader (runtime OOM guard)."""
        return self._rebuild_train_loader(self.train_dataloader.batch_size // 2)

    def train(self):
        """Train within the clock budget and return the best model found.

        Structure, in one place because it is spread over a long loop below:

        * The LR is a cosine driven by the CLOCK, never by an epoch estimate.
        * Every epoch logs train accuracy and the train-val gap (see the module
          docstring), and the run ends with an automatic diagnosis.
        * ONE model is trained unless the dataset saturates. If a member stops
          improving for `_saturation_patience()` epochs while a whole further
          member still fits, its best weights are snapshotted, the network is
          re-initialised, and another is trained; `predict()` then averages them.
          Member 1 uses the global cosine; later members get their own cosine
          across their own slice, sized by how long member 1 took to converge.
        * Nothing here may raise: the loop is wrapped, and `best_state` is
          restored at the end whatever happened.

        Returns the model loaded with the single best validation checkpoint.
        The ensemble, if any, lives in `self.ensemble_states` for `predict()`.
        """
        try:
            self.model.to(self.device)
        except Exception:
            pass

        best_state = copy.deepcopy(self.model.state_dict())
        best_val = -1.0

        budget = self._remaining()
        # Reserve for predict() + overhead, proportional to the budget.
        # IMPORTANT: no large fixed floor - under very short budgets (<=60s)
        # a fixed 20s floor used to eat the ENTIRE budget, so not a single
        # epoch ever ran (bug found via a 20s stress test).
        margin = max(2.0, min(0.15 * budget, 60.0))

        # Cosine LR driven by the CLOCK rather than by an estimated epoch count.
        # The old version froze a `planned` total epoch count from epoch 1's
        # duration - but epoch 1 is precisely the epoch that can be cut short
        # (persistent-OOM guard below, cudnn autotuning, allocator warm-up).
        # An aborted 0.9s epoch 1 followed by real 31.7s epochs yielded
        # planned~3500 while only ~93 epochs actually ran, so
        # cos(pi*93/3500) ~ 1.0 and the LR never annealed at all - observed on
        # 4 of 9 datasets. Deriving progress from the budget actually consumed
        # is immune to partial epochs, OOM restarts and varying epoch cost, and
        # reaches lr~0 exactly as the training budget runs out.
        usable = max(1e-6, budget - margin)   # wall-clock actually available for training
        # _remaining() returns 1e9 when the clock is unavailable. Without this
        # check that sentinel makes progress ~0.0 forever, pinning the LR at
        # base_lr for every one of MAX_EPOCHS epochs with no annealing at all.
        clock_ok = budget < 1e8
        if not clock_ok:
            print('[Trainer] clock unavailable - holding LR at base_lr, no annealing')

        epoch_time = 0.0     # duration of the last COMPLETED epoch (see below)
        lr = BASE_LR
        best_epoch, epochs_run = 0, 0
        train_acc = float('nan')

        # --- ensemble member bookkeeping (see MAX_MEMBERS) -------------------
        # Member 1 trains exactly as it always has, against the global
        # clock-driven cosine, and is watched for saturation. Only if it
        # saturates with enough budget left do further members happen, and how
        # long member 1 took to CONVERGE is what calibrates theirs. On a dataset
        # that keeps improving none of this fires and the run is byte-for-byte
        # today's.
        self.ensemble_states = []

        def stop_margin():
            """Time that must remain unspent when training stops.

            `margin` alone covers ONE pass over the test set. Each extra
            ensemble member costs roughly one more, so once members exist the
            training loop has to stop correspondingly earlier - otherwise the
            last member trains right up to `margin` and predict() then has to
            drop members it already paid to train. (Dropping them is safe, which
            is why this is a waste rather than a -10, but it is still waste.)
            Returns exactly `margin` while no ensemble exists, so the
            single-model path is unaffected."""
            return margin + self._ensemble_predict_reserve()
        member_start = time.time()
        member_budget = None        # set once member 1 has shown how long it needs
        member_best_val, member_best_state = -1.0, None
        member_best_time = 0.0      # how long this member took to reach its best
        epochs_in_member, since_improve = 0, 0

        try:
            for epoch in range(MAX_EPOCHS):
                # only start if the estimated epoch + reserve still fits
                if self._remaining() - epoch_time < stop_margin():
                    break

                if clock_ok:
                    if member_budget is None:
                        # member 1 (or the no-ensemble case): anneal across the
                        # whole remaining budget, exactly as before
                        progress = 1.0 - (self._remaining() - margin) / usable
                    else:
                        # later members get their OWN cosine across their own
                        # slice, so each one is properly annealed rather than
                        # inheriting whatever LR the previous member ended on
                        progress = (time.time() - member_start) / member_budget
                    progress = min(1.0, max(0.0, progress))
                    lr = 0.5 * BASE_LR * (1.0 + math.cos(math.pi * progress))
                    for g in self.optimizer.param_groups:
                        g['lr'] = lr

                t0 = time.time()
                self.model.train()
                oom_batches = 0
                epoch_complete = True    # cleared if this epoch is cut short
                # Running TRAIN accuracy/loss, accumulated from the forward
                # passes the training step already does - see the module docstring
                # for what this number does and does not mean.
                # Kept as GPU tensors and read ONCE per epoch: calling .item()
                # per batch would force a host sync every step.
                tr_correct = torch.zeros((), device=self.device)
                tr_loss = torch.zeros((), device=self.device)
                tr_seen, tr_batches = 0, 0
                for data, target in self.train_dataloader:
                    # Belt-and-braces against a single-sample batch: BatchNorm
                    # raises on one sample in train mode, and that is not an
                    # OOM, so it would propagate and end training entirely.
                    if data.shape[0] < 2:
                        continue
                    try:
                        data = data.to(self.device)
                        target = target.to(self.device)
                        self.optimizer.zero_grad(set_to_none=True)
                        with autocast(enabled=self.use_amp):
                            out = self.model(data)
                            loss = self.criterion(out, target)
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        # free: reuses `out`, which we already computed
                        with torch.no_grad():
                            tr_correct += (out.argmax(1) == target).sum()
                            tr_loss += loss.detach()
                        tr_seen += int(target.shape[0])
                        tr_batches += 1
                    except RuntimeError as e:
                        if _is_oom(e):
                            oom_batches += 1
                            self.optimizer.zero_grad(set_to_none=True)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if oom_batches > 5:
                                new_bs = self._shrink_train_loader()
                                print("[Trainer] persistent OOM -> batch size reduced to {}".format(new_bs))
                                epoch_complete = False
                                break   # abort this epoch, next one uses the smaller loader
                            continue
                        raise
                    if self._remaining() < stop_margin():  # avoid a hard timeout
                        epoch_complete = False
                        break

                # Only a COMPLETED epoch says anything about what the next one
                # will cost; a partial epoch under-estimates it badly, which is
                # what used to poison the schedule. Keep the previous estimate
                # in that case (0.0 on the first epoch -> just try again).
                this_epoch_time = time.time() - t0
                if epoch_complete:
                    epoch_time = this_epoch_time

                # Never start a validation pass that could eat into the
                # margin reserved for predict() - a slow/large valid set
                # would otherwise risk pushing the whole dataset past its
                # time limit (-> instant -10 for this dataset).
                remaining = self._remaining()
                if remaining <= stop_margin():
                    print("  [Trainer] Epoch {:>3} | skipping validation - "
                          "no time left beyond the {:.0f}s margin".format(epoch + 1, stop_margin()))
                    break

                eval_budget = remaining - stop_margin()
                t_eval = time.time()
                val = self._evaluate(time_budget=eval_budget)
                self._last_eval_time = time.time() - t_eval
                if val >= best_val:
                    best_val = val
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(self.model.state_dict())

                # per-member tracking, independent of the global best above:
                # each member contributes ITS OWN best checkpoint to the ensemble
                epochs_in_member += 1
                if val > member_best_val:
                    member_best_val = val
                    member_best_state = self._snapshot()
                    member_best_time = time.time() - member_start
                    since_improve = 0
                else:
                    since_improve += 1

                # one host sync per epoch, not per batch
                train_acc = float(tr_correct.item()) / tr_seen if tr_seen else float('nan')
                train_loss = float(tr_loss.item()) / tr_batches if tr_batches else float('nan')
                epochs_run = epoch + 1

                print("  [Trainer] Epoch {:>3}{} | train={:5.2f}% val={:5.2f}% gap={:+6.2f} | loss={:.4f} | "
                      "lr={:.5f} | t/ep={:5.1f}s t/eval={:4.1f}s | rem={:6.0f}s".format(
                          epochs_run,
                          '' if not self.ensemble_states else ' m%d' % (len(self.ensemble_states) + 1),
                          train_acc * 100, val * 100, (train_acc - val) * 100,
                          train_loss, lr, this_epoch_time, time.time() - t_eval, self._remaining()))

                # --- should this member be closed and a fresh one started? ---
                elapsed_member = time.time() - member_start
                # Time a FRESH member would need: convergence plus an annealing
                # tail. Computed here as well as below because the "is there room
                # for another member?" test must use it.
                t_conv = max(member_best_time, 0.34 * elapsed_member, 1.0)
                if member_budget is None:
                    # member 1: watch for saturation
                    done = since_improve >= _saturation_patience(epochs_in_member)
                    # NOT elapsed_member. A later member is sized from CONVERGENCE
                    # time and does not repeat member 1's patience tail, so asking
                    # for `elapsed_member` of room demanded ~4x what the member we
                    # would actually build costs - and once detection passes ~50%
                    # of the budget that test can never pass. Measured: windspeed
                    # detected saturation at epoch 141 of 268 and built ZERO
                    # members, needing 1664s of room for a 637s member
                    # (CLAUDE.md 7m).
                    next_cost = MEMBER_LENGTH_FACTOR * t_conv
                else:
                    # later members run their calibrated slice to completion; the
                    # cosine has annealed by then, so there is nothing to detect
                    done = elapsed_member >= member_budget
                    next_cost = member_budget
                if done and member_best_state is not None:
                    # Only start another member if a WHOLE one still fits, plus the
                    # reserve. A half-trained member is worse than no member, and
                    # every extra member also costs a full pass in predict().
                    room = self._remaining() - margin - self._ensemble_predict_reserve()
                    have_room = (len(self.ensemble_states) + 1 < MAX_MEMBERS
                                 and room >= next_cost)
                    if not have_room and member_budget is not None:
                        # A later member has finished its slice and fully annealed,
                        # and another does not fit. Continuing would train at the
                        # cosine's lr=0 floor for the rest of the budget, learning
                        # nothing - stop and let the final append collect it.
                        # (Member 1 is deliberately excluded: with no ensemble this
                        # must behave exactly as it always has and keep training.)
                        break
                    if have_room:
                        self.ensemble_states.append(member_best_state)
                        if member_budget is None:
                            # Size the remaining members by how long member 1 took to
                            # CONVERGE, not by how long it took to prove it had
                            # converged - the patience tail is detection cost and a
                            # fresh member does not have to repeat it. Then stretch
                            # them to fill the budget exactly, so nothing is left on
                            # the table. Sizing by elapsed_member instead left 25% of
                            # a 600s Chesseract budget unused and bought only 2
                            # members where the budget supported 7.
                            # How long ONE member needs: enough to converge plus a
                            # tail for the cosine to anneal. Sizing members as
                            # avail/MAX_MEMBERS instead made them ~10x longer than
                            # necessary once the member cap bound, which it did even
                            # at 1h: Chesseract ran 7 x 440s while converging in 33s,
                            # and Sudoku at 3.5h ran 7 x 1678s while converging in
                            # 178s - ~90% of every member's time spent after it had
                            # stopped improving (CLAUDE.md 7j, issue 2).
                            #
                            # KNOWN WEAKNESS (CLAUDE.md 7f.2b): member_best_time is the
                            # time to the ARGMAX epoch, which is noisy when the
                            # validation curve is flat - three 600s Chesseract runs
                            # gave 7, 2 and 7 members. Every run still beat the single
                            # model, so this is magnitude, not correctness, but a
                            # robust convergence estimate belongs here.
                            # (t_conv is computed above, where the room test needs it)
                            target = MEMBER_LENGTH_FACTOR * t_conv

                            # predict() costs one test pass per member, so the reserve
                            # grows as members accumulate. Estimate it for the member
                            # count we are about to plan rather than the one we have
                            # now, otherwise the last members silently do not fit and
                            # their budget is wasted.
                            avail0 = max(0.0, self._remaining() - margin)
                            k0 = max(1, min(int(avail0 // target), MAX_MEMBERS - 1))
                            avail = max(0.0, avail0 - k0 * self._last_eval_time * 1.5)

                            k_more = int(avail // target)
                            k_more = max(1, min(k_more, MAX_MEMBERS - 1))
                            # avail/k_more is >= target by construction, so this fills
                            # the budget exactly while keeping members near `target`
                            member_budget = max(1.0, avail / k_more)
                            print("[Trainer] saturated after {} epochs ({:.0f}s, best at {:.0f}s) with "
                                  "{:.0f}s left -> up to {} more independent members of {:.0f}s each"
                                  .format(epochs_in_member, elapsed_member, member_best_time,
                                          self._remaining(), k_more, member_budget))
                        n_reset = self._reinit_model()
                        print("[Trainer] member {} done (best val {:.2f}%); re-initialised {} modules "
                              "for member {}".format(len(self.ensemble_states), member_best_val * 100,
                                                     n_reset, len(self.ensemble_states) + 1))
                        member_start = time.time()
                        member_best_val, member_best_state = -1.0, None
                        member_best_time = 0.0
                        epochs_in_member, since_improve = 0, 0
        except Exception as e:
            print("[Trainer] training ended early:", repr(e))

        # the member in progress when the clock ran out still counts
        try:
            if self.ensemble_states and member_best_state is not None:
                self.ensemble_states.append(member_best_state)
        except Exception:
            pass

        self._log_training_summary(epochs_run, best_epoch, best_val, train_acc)
        if len(self.ensemble_states) > 1:
            print("[Trainer]   -> ensembling {} independently initialised members "
                  "in predict()".format(len(self.ensemble_states)))

        # restore the best model seen so far. This is what predict() uses when
        # there is no ensemble, and what it falls back to if ensembling fails.
        try:
            self.model.load_state_dict(best_state)
            self._best_state_for_fallback = best_state
        except Exception:
            pass
        return self.model

    def _log_training_summary(self, epochs_run, best_epoch, best_val, train_acc):
        """One line stating the diagnosis, so it does not have to be re-derived
        by hand from hundreds of epoch lines (or by re-running the dataset).

        The two questions worth answering per dataset are 'did we run out of
        time or out of ideas?' and 'are we over- or under-fitting?'. Both are
        answerable from numbers already in hand:
          * best_epoch / epochs_run - if the best checkpoint arrives in the
            first few percent of the run, every remaining epoch was wasted and
            a longer budget would not have helped. In the 2026-07-27 run this
            was true of 5 of 13 datasets (Chesseract peaked at epoch 4 of 393)
            and it was invisible because only val accuracy was logged.
          * train - val - a large positive gap with high train accuracy means
            memorisation, so the answer is regularisation, NOT more capacity
            or more time. Measured 99.99%/53% on Chesseract, 100%/79% on
            Language.
        """
        if not epochs_run:
            print("[Trainer] summary: no epoch completed")
            return
        frac = 100.0 * best_epoch / epochs_run
        gap = (train_acc - best_val) * 100
        notes = []
        if frac <= 25.0:
            notes.append("SATURATED - best checkpoint in the first {:.0f}% of the run, "
                         "the remaining {} epochs gained nothing".format(frac, epochs_run - best_epoch))
        if train_acc == train_acc and gap >= 15.0 and train_acc >= 0.95:
            notes.append("MEMORISING - train {:.1f}% vs val {:.1f}%, gap {:+.1f}pts "
                         "(needs regularisation, not capacity/time)".format(
                             train_acc * 100, best_val * 100, gap))
        print("[Trainer] summary: {} epochs, best val {:.2f}% at epoch {} ({:.0f}% in), "
              "final train {:.2f}%".format(
                  epochs_run, best_val * 100, best_epoch, frac,
                  train_acc * 100 if train_acc == train_acc else float('nan')))
        for n in notes:
            print("[Trainer]   -> {}".format(n))

    def _evaluate(self, time_budget=None):
        """Validation pass. If time_budget (seconds) is given, the pass
        stops early once it's exceeded, so a slow or unexpectedly large
        validation set can never run past it - it just scores on however
        many batches it got through instead of blowing the time budget."""
        self.model.eval()
        y_true, y_pred = [], []
        t0 = time.time()
        with torch.no_grad():
            for data, target in self.valid_dataloader:
                try:
                    data = data.to(self.device)
                    with autocast(enabled=self.use_amp):
                        out = self.model(data)
                    y_pred += torch.argmax(out, 1).cpu().tolist()
                    y_true += target.tolist()
                except RuntimeError as e:
                    if _is_oom(e) and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                if time_budget is not None and (time.time() - t0) > time_budget:
                    print("  [Trainer] validation time budget ({:.0f}s) reached, "
                          "scoring on {}/{} valid samples".format(
                              time_budget, len(y_true), len(self.valid_dataloader.dataset)))
                    break
        return _acc(y_true, y_pred) if y_true else 0.0

    def _predict_batch(self, data):
        """Predict one batch; halve recursively on OOM (order is preserved)."""
        try:
            with autocast(enabled=self.use_amp):
                out = self.model(data.to(self.device))
            return torch.argmax(out, 1).cpu().tolist()
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if data.shape[0] > 1:
                mid = data.shape[0] // 2
                return self._predict_batch(data[:mid]) + self._predict_batch(data[mid:])
            return [self.fallback_label]   # a single sample still doesn't fit -> fallback

    def _ensemble_predict_reserve(self):
        """Seconds to hold back in train() for the EXTRA test passes ensembling
        will need. One pass is already covered by `margin`; each additional
        member costs roughly one more.

        train() never sees the test loader, so the validation pass is the only
        measurement available. Test and validation splits are usually similar in
        size but need not be, so this is an estimate - the guarantee comes from
        predict() re-checking the clock before every member and simply stopping
        with fewer, which is always safe."""
        if len(self.ensemble_states) < 1 or not self._last_eval_time:
            return 0.0
        return len(self.ensemble_states) * self._last_eval_time * 1.5

    def _predict_probs(self, test_loader, n_test):
        """Class probabilities for the whole test split, [n_test, C] on CPU.
        Returns None if this member could not be scored at all."""
        self.model.eval()
        chunks = []
        try:
            with torch.no_grad():
                for data in test_loader:
                    chunks.append(self._predict_probs_batch(data))
        except Exception as e:
            print("[Trainer] member scoring failed:", repr(e))
            return None
        if not chunks:
            return None
        p = torch.cat(chunks, dim=0)
        return p if p.shape[0] == n_test else None

    def _predict_probs_batch(self, data):
        """Softmax probabilities for one batch; halves recursively on OOM so the
        row order is preserved (same contract as _predict_batch)."""
        try:
            with autocast(enabled=self.use_amp):
                out = self.model(data.to(self.device))
            return torch.softmax(out.float(), dim=1).cpu()
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if data.shape[0] > 1:
                mid = data.shape[0] // 2
                return torch.cat([self._predict_probs_batch(data[:mid]),
                                  self._predict_probs_batch(data[mid:])], dim=0)
            raise

    def predict(self, test_loader):
        """Predicted class labels for the test split.

        CONTRACT (the harness depends on all three): exactly `n_test` labels, in
        the loader's order, and this function must never raise. Everything below
        is arranged around that - OOM halves the batch rather than failing, a
        short result is padded with the majority training class, and a long one
        is truncated.

        With more than one ensemble member the members' softmax outputs are
        averaged; any problem at all in that path falls through to the
        single-model path, which is what has always run.
        """
        n_test = len(test_loader.dataset)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        # More than one member -> average their probabilities. Anything at all
        # wrong here falls through to the single-model path below, which is
        # unchanged and is what has always run.
        if len(getattr(self, 'ensemble_states', [])) > 1:
            try:
                preds = self._predict_ensemble(test_loader, n_test)
                if preds is not None:
                    return preds
            except Exception as e:
                print("[Trainer] ensemble predict failed, using single model:", repr(e))
            # restore the single best weights before the fallback path runs -
            # self.model currently holds whichever member was scored last
            if self._best_state_for_fallback is not None:
                try:
                    self.model.load_state_dict(self._best_state_for_fallback)
                except Exception as e:
                    print('[Trainer] could not restore best weights:', repr(e))

        preds = []
        self.model.eval()
        try:
            with torch.no_grad():
                for data in test_loader:
                    preds += self._predict_batch(data)
        except Exception as e:
            print("[Trainer] predict() emergency handling:", repr(e))

        # force the length to exactly n_test - never shorter, never longer
        if len(preds) < n_test:
            preds += [self.fallback_label] * (n_test - len(preds))
        elif len(preds) > n_test:
            preds = preds[:n_test]
        return preds

    def _predict_ensemble(self, test_loader, n_test):
        """Average the members' probabilities. Returns None to fall back.

        Members are scored ONE AT A TIME with a clock check in between, and the
        first pass is timed so the check is based on a measurement rather than a
        guess. Running out of time therefore costs members, not the dataset:
        whatever has been accumulated is used. That is the property that keeps
        this off the -10 path, since a full pass per member is the one cost here
        that scales."""
        prob_sum, used, t_pass = None, 0, None
        for i, state in enumerate(self.ensemble_states):
            if used >= 1:
                # need room for another full pass plus a safety factor
                if t_pass is not None and self._remaining() < t_pass * 1.5:
                    print("[Trainer] stopping ensemble at {}/{} members - {:.0f}s left, "
                          "a pass costs ~{:.0f}s".format(used, len(self.ensemble_states),
                                                         self._remaining(), t_pass))
                    break
            try:
                self.model.load_state_dict(state)
            except Exception as e:
                print("[Trainer] could not load member {}: {}".format(i + 1, repr(e)))
                continue
            t0 = time.time()
            p = self._predict_probs(test_loader, n_test)
            if p is None:
                continue
            t_pass = time.time() - t0 if t_pass is None else t_pass
            prob_sum = p if prob_sum is None else prob_sum + p
            used += 1
        if prob_sum is None or used < 2:
            return None       # nothing gained - let the single-model path run
        print("[Trainer] ensembled {}/{} members over {} test samples".format(
            used, len(self.ensemble_states), n_test))
        preds = prob_sum.argmax(dim=1).tolist()
        if len(preds) != n_test:
            return None
        return preds
