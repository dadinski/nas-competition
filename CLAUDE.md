# CLAUDE.md — NAS Unseen-Data Challenge 2026 Submission

This file gives Claude (in any interface: chat, Code, etc.) the context needed to work on this
repository without re-deriving it from scratch. Team: Samuel Kostiuk (10053722), Daniel Gleim
(10043913). Module: AutoML project exam.

**Priority order for facts:** nascompetition.com / nascompetition.com/rules > the official starter
kit (github.com/Towers-D/NAS-Comp-Starter-Kit) > this file > general NAS knowledge. If something
here looks like it might be stale, re-check the website before relying on it — the competition is
live and details can change between phases.

---

## 1. Project Goal

We are entering the **NAS Unseen-Data Challenge 2026** (nascompetition.com). The goal is a robust,
competition-ready NAS pipeline (`DataProcessor`, `NAS`, `Trainer`) that performs well on three
*completely novel, hidden* datasets under strict per-dataset time and memory budgets. Overfitting
to the bundled example datasets is explicitly against the spirit of the competition — the pipeline
must generalize to arbitrary image shapes, channel counts, and class counts sight-unseen.

Guiding proposal (see `Proposal.pdf` in the project): search **training-free** using zero-cost
proxies over a compact NASBench-201-style cell space, spend the majority of the time budget
training the single selected architecture, and distribute time dynamically across datasets using
the harness's `time_remaining` field. Robustness (fallbacks, error handling, checkpointing) is
built in throughout, not bolted on at the end.

Rough phased build plan (`Grober_Ablaufplan`, already substantially executed):
1. Get the harness running locally against the example submission.
2. Fallback-first skeleton (fixed ResNet-like model, no search yet, try/except everywhere) —
   tested at 5/30/120 min budgets. Goal: no more `-10` scores possible from this point on.
3. Adaptive architecture (arbitrary channel stem, downsampling from H×W, OOM-safe build loop).
4. NAS search (cell search space, NASWOT + SynFlow rank aggregation, verify top-n; skip search
   entirely if the clock is tight).
5. Training policy (+ optional offline-pretrained weights): optimizer, LR schedule, AMP,
   clock-budgeted epochs, conservative augmentation.
6. Hardening: benchmark across many datasets, strip `main.py`/`score.py` from the submission zip,
   verify offline, then submit (submission budget is limited — see §2).

---

## 2. Competition Timeline & Rules (verified against nascompetition.com, 2026-07-27)

**Phase status as of today: Phase 2 is open now. Phase 3 starts August 1st, 2026.** That is very
soon — Phase 2 submission needs to be in a genuinely working state well before then, since **the
last working Phase 2 submission is reused automatically for Phase 3; there is no separate Phase 3
submission.** Finalists are informed in August; results are announced at AutoML 2026 in Ljubljana,
September 28 – October 1.

Key rules (nascompetition.com/rules):
- **Submission limit in Phase 2: 7 total.** Don't spam the submit button — rapid clicks can send
  duplicate submissions and burn the budget. Multiple accounts to bypass this are forbidden.
- A `run_clock`/time budget is passed into the pipeline; the harness independently tracks its own
  clock too. **We must self-terminate early if we're going to overrun — attempting to bypass the
  time limit is against the rules**, not just bad for score.
- **Timeout on a dataset → score of `-10` for that dataset.** Same for any crash caused by
  exceeding RAM/VRAM. These are the two failure modes our fallback/robustness work exists to
  prevent.
- Do not attempt to download the final-stage evaluation datasets — that's an instant
  disqualification (they're kept secret by design).
- Organiser decisions are final; teams must work toward one shared submission.

Scoring (nascompetition.com/info): each dataset has a `benchmark` accuracy baked into its metadata
— matching it scores 0, up to +10 for 100% accuracy, down to -10 for very poor accuracy (in
addition to the automatic -10 for timeout/crash described above).

---

## 3. Evaluation Harness — How It Actually Works

Source: nascompetition.com/info and the starter kit README (github.com/Towers-D/NAS-Comp-Starter-Kit).

**Required interface** (see `submission_template/` in the starter kit for the authoritative
signatures — don't hand-guess these):
- `DataProcessor.__init__(train/valid/test numpy arrays, metadata)` → `.process()` returns 3
  PyTorch dataloaders (train/valid/test).
- `NAS.__init__(dataloaders, metadata)` → `.search()` returns a PyTorch model.
- `Trainer.__init__(dataloaders, model)` → `.train()` returns a fully trained model;
  `.predict(test_dataloader)` returns predicted class labels.

**Pipeline per dataset:** raw arrays → `DataProcessor` → dataloaders → `NAS.search()` → model →
`Trainer.train()` → trained model → `Trainer.predict()` → predictions → scored by `score.py`
against the benchmark.

**Do not include `main.py` or `score.py` in the submission zip** — any files with those names get
overwritten by the harness's own copies and can invalidate the submission.

**Metadata dict** passed to all three classes includes: `num_classes`, `input_shape` (shape of
`train_x`, i.e. `(#images, channels, H, W)`), `codename`, `benchmark` (accuracy needed to score 0),
and dynamically, `time_remaining` (seconds left, added by the harness at runtime).

**Per-dataset time limit:**
- The 2026 starter kit defines the time limit **per dataset**, defaulting to **30 minutes**, and
  reads an optional `time_limit` field (in **hours**) from that dataset's metadata file if present
  (e.g. `"time_limit": 1.0` for a 1-hour budget). This is separate from the overall submission
  runtime limit below.
- `load_dataset_metadata()` in `main.py` currently unconditionally overwrites
  `metadata['time_limit'] = 3.0` (3 hours), ignoring whatever the dataset's own metadata file
  specifies. **This is intentional, not a bug** — it's a temporary testing scaffold Daniel and
  Samuel added on purpose to get generous, predictable budgets during local development. **It must
  be removed before the real submission**, so that `time_limit` is read from each dataset's actual
  metadata (falling back to the real 0.5-hour default when absent) — see the to-do in §6.

**Overall submission runtime limit: 24 hours**, confirmed by Daniel as the correct figure (matches
nascompetition.com/**info**, which states the total budget across all three datasets is 24 hours,
enforced via the `time_remaining` field). The starter kit README separately mentions a `TIME_LIMIT`
constant in `main.py` defaulting to 12 hours with only 1 hour given in Phase 2 test runs — treat
that as either outdated or referring to something other than our actual allotment; **24 hours from
the competition site is the number to design and budget around.**

**Local testing:** the starter kit's `Makefile` runs the same evaluation scripts used server-side —
`make submission=$SUBMISSION_DIRECTORY all` to test end-to-end, `make submission=$SUBMISSION_DIRECTORY
zip` to bundle for submission. Always test with this before submitting; it's the closest thing we
have to a dress rehearsal of the real harness.

---

## 4. Architecture (current design)

- **Micro search space:** NASBench-201-style cell — 4 nodes, 6 edges, 5 candidate ops per edge
  (conv3x3, conv1x1, avgpool3x3, skip/identity, none) → 5⁶ ≈ 15,625 possible cells.
- **Macro skeleton:** adaptive — stacked-cell count, channel width, and downsampling steps are
  derived from `input_shape` and available GPU memory (via OOM probing), not fixed. This lets one
  search space span both large images and tiny inputs (e.g. 8×8) without over-downsampling.
- **Fallback cascade:** primary searched architecture → `TinyNet` (adaptive, spatial downsampling
  with an `s_min=4` floor) → `MinimalNet` (GlobalAvgPool + Linear, absolute last resort).
  `_safe_fallback()` fit-checks before escalating down this chain.

**Three-phase NAS search** (`nas.py`):
1. Macro sizing via OOM probing — uses the fixed `ResidualBlock`, *not* the searched genotype, to
   size the skeleton. Consequence: the eventually-found cell can still OOM at verification time at
   the loader's default batch size — this is expected, not a hard failure, and is handled
   gracefully (batch shrink, then fall back down the cascade) rather than treated as an abort.
2. Proxy-ranked cell search over the **full shuffled search space** (not a pre-computed random
   sample) — `_search_cell()` iterates `all_genotypes()` shuffled, checking the phase deadline
   *before* any expensive work on every iteration, so it can never meaningfully overrun
   `search_budget` (10% of `time_remaining`, floor `min_search_budget = 8s` below which search is
   skipped entirely). Scoring also stops early at a hard cap of **1000 scored non-degenerate
   candidates**, whichever comes first — verified safe against the actual `nas.py` source: the
   deadline check always runs first each iteration, so this cap can't cause an overrun; it exists
   to hand leftover
   `search_budget` time to verification once ~1000 candidates is a large-enough sample, rather than
   scoring the full ~15,561-genotype space for marginal ranking gain. Survivors are scored with
   NASWOT and SynFlow, combined via **rank aggregation** (not score averaging), and the shortlist
   size `n_verify` is computed dynamically from remaining time and per-candidate cost (capped at
   `breadth_cap = 100`) — this differs from the proposal's fixed "start with n=10."
3. Final build with fallback cascade — briefly train-verify the top-n shortlist at reduced
   fidelity, pick the best validation accuracy, build the winner; fall back down the TinyNet →
   MinimalNet chain if it doesn't fit memory.

**Codebase layout:** `data_processor.py`, `model.py`, `nas.py`, `proxies.py`, `trainer.py`,
`helpers.py`. *(§4's search-phase description above is verified directly against `nas.py` as of
2026-07-27; the macro-sizing and fallback-cascade descriptions are still from earlier context and
proposal text, not yet re-checked against current source — treat those as likely-but-unconfirmed
until a file surfaces to check them against.)*

---

## 5. Working Conventions for Claude on This Project

- Stick to the rough plan above and the proposal's approach unless Daniel and/or Samuel direct otherwise.
- When generating code, add short, precise **English** comments explaining the logic.
- When editing existing files, output **only the changed files**, not the full set of six.
- Follow nascompetition.com and nascompetition.com/rules over general NAS knowledge or training
  data when the two conflict — the competition can and does change details year to year.
- If uncertain about a harness/rules detail, say so and ask rather than guessing — a wrong
  assumption here risks an automatic -10.
- **Before the final submission is zipped: remove the `metadata['time_limit'] = 3.0` testing
  override in `main.py`'s `load_dataset_metadata()` (§3)** so real per-dataset time limits apply
  again. It's a deliberate local-testing shortcut, kept in place on purpose for now — easy to
  forget precisely because it's intentional rather than a bug someone will trip over.
