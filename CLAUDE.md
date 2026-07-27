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

**Status (2026-07-27):** steps 1–5 are implemented in `submission_baseline/`; step 6 has *not*
happened. The pipeline is robust (no crash or timeout across a 9-dataset run) but was **not yet
competitive** at last measurement: `Final_Score 2.167`, with 2 of 3 datasets scoring below
benchmark — and that run had 3h per dataset, 3× the Phase 2 allotment. A round of diagnosed fixes
has since landed (see §4a and §4) and is **unmeasured**; re-benchmarking at a realistic 1-hour
budget is the next milestone.

A larger batch of changes was implemented on 2026-07-27, reviewed, and **reverted** — the pipeline
deliberately sits at a small, individually verified set of fixes so that the next benchmark measures
something attributable. **Read §7a before re-attempting any of that work**, and treat §7 as the single
source of truth for what remains rather than restating items here.

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
  `metadata['time_limit'] = 1.0` (1 hour — the value was 3.0 earlier, so don't trust a remembered
  number, re-read the file), ignoring whatever the dataset's own metadata file specifies. **This is
  intentional, not a bug** — it's a temporary testing scaffold Daniel and Samuel added on purpose
  to get generous, predictable budgets during local development. **It must be removed before the
  real submission** — see the to-do in §6, and note the landmine described there.

**Overall submission runtime limit: 24 hours**, confirmed by Daniel as the correct figure (matches
nascompetition.com/**info**, which states the total budget across all three datasets is 24 hours,
enforced via the `time_remaining` field). The starter kit README separately mentions a `TIME_LIMIT`
constant in `main.py` defaulting to 12 hours with only 1 hour given in Phase 2 test runs — treat
that as either outdated or referring to something other than our actual allotment; **24 hours from
the competition site is the number to design and budget around.**

**Per-dataset clocks are independent — there is no cross-dataset time redistribution.**
`main.py` constructs a fresh `Clock(dataset_limit)` for each dataset, so unused time on dataset 1
cannot be handed to dataset 2; the 24h total is enforced externally. The proposal's goal of
"distributing time dynamically across datasets" is therefore **not implementable against this
harness and has been dropped** (decision: Samuel, 2026-07-27). Adapting *within* a dataset via the
clock is still central to the design and unaffected.

**Phase 2 gives 1 hour; Phase 3 gives "an unknown amount of runtime"** (starter kit README,
"Runtime"). Design for adaptivity across a wide budget range rather than tuning to any one number.

**The evaluation environment is fixed and unknown to us, and `requirements.txt` does NOT reach the
server.** The `Makefile`'s `zip` target is `cd $(submission); zip -r ../submission.zip *`, which
bundles only the submission directory — `requirements.txt` lives in the repo root and is never
included. Its own first line says "choose correct CUDA version for *your system*"; it configures
local development only. The README warns that "trying to import something that doesn't exist in our
environment will break your submission." **Practical consequence: submission code must stay
version-portable.** Concretely, keep `from torch.cuda.amp import autocast, GradScaler` — the modern
`torch.amp.autocast('cuda', ...)` spelling does **not** exist on torch 1.10 and would hard-fail if
the server is old. Likewise `torch.cuda.mem_get_info` (used by `DataProcessor._choose_batch_size`)
may be absent there; it is already guarded, but that means our batch sizing may silently fall back
to the crude heuristic on the real server.

**Local testing:** the starter kit's `Makefile` runs the same evaluation scripts used server-side —
`make submission=$SUBMISSION_DIRECTORY all` to test end-to-end, `make submission=$SUBMISSION_DIRECTORY
zip` to bundle for submission. Always test with this before submitting; it's the closest thing we
have to a dress rehearsal of the real harness. Run everything with the **`stx_r` conda env**
(`C:\Users\Samue\anaconda3\envs\stx_r\python.exe`) — the default `python` on PATH has no torch.

---

## 4. Architecture (current design)

- **Micro search space:** NASBench-201-style cell — 4 nodes, 6 edges, 5 candidate ops per edge
  (conv3x3, conv1x1, avgpool3x3, skip/identity, none) → 5⁶ ≈ 15,625 possible cells.
- **Macro skeleton:** adaptive — channel width, blocks per stage and downsampling steps are derived
  from `input_shape` and available GPU memory (OOM probing), not fixed. `derive_macro()` gives
  `c0=32, n=2, d=clamp(log2(min(H,W)/4), 0, 3)`; `_find_macro()` then **shrinks** c0 → n → d until a
  forward+backward fits. It only ever shrinks — see §7 for why the time-aware growth experiment was
  reverted.
- **Fallback cascade:** primary searched architecture → `TinyNet` (adaptive, spatial downsampling
  with an `s_min=4` floor) → `MinimalNet` (GlobalAvgPool + Linear, absolute last resort).
  `_safe_fallback()` fit-checks before escalating down this chain.

**Three-phase NAS search** (`nas.py`):
1. Macro sizing via OOM probing — uses the fixed `ResidualBlock`, *not* the searched genotype.
   Consequence: the eventually-found cell can still OOM at verification time at the loader's default
   batch size — expected, not a hard failure, handled by batch shrink (recorded as
   `nas_batch_size_hint`) then the fallback cascade.
2. Proxy-ranked cell search over the **full shuffled search space** — `_search_cell()` iterates
   `all_genotypes()` shuffled, checking the phase deadline *before* any expensive work each
   iteration, so it cannot meaningfully overrun `search_budget` (10% of `time_remaining`, floor
   `min_search_budget = 8s` below which search is skipped entirely). Scoring stops at a hard cap of
   1000 scored non-degenerate candidates or the deadline, whichever comes first. Survivors are
   scored with NASWOT + SynFlow and combined via **rank aggregation** (not score averaging). The
   shortlist size `n_verify` and `max_batches` are sized from the measured `t_batch`/`t_val`, capped
   at `breadth_cap = 100`.
3. Flat verification — train each shortlisted genotype briefly, best validation accuracy wins;
   build the winner, falling back down TinyNet → MinimalNet if it doesn't fit.

**Timing measurements must warm up and synchronise.** `_measure_step_time()` discards a warm-up step
and calls `torch.cuda.synchronize()` around the timed region, and `_calibrate_verify_cost()` uses it.
Without both, a measurement carries two errors at once — the first step includes cudnn autotuning
(inflates) and an unsynchronised timer measures kernel *launches*, not execution (deflates) — so the
result is unreliable in an unpredictable direction (measured 17× over on a small model; production
logs implied ~2× under on a larger one). Any new timing code must follow the same pattern.

**⚠ But `_measure_step_time` measures GPU compute only.** It times an already-materialised tensor, so
it excludes the DataLoader and per-sample augmentation. With `num_workers=0` that omitted cost is the
*majority* of a real epoch — measured **4.5×–6.7×** on AddNIST. It is fine for its current use
(comparing candidates against each other, where the omission is a shared constant). **It must never
be used to project wall-clock epoch counts** — that mistake is what invalidated the macro-growth
experiment (§7).

### 4a. Training & data invariants (don't regress these)

Each of these fixed a diagnosed bug. Changing them needs a reason.

- **The Trainer's cosine LR is driven by the clock, not by an epoch estimate.**
  `progress = 1 - (remaining - margin) / usable`, so `lr` runs `BASE_LR → 0` exactly as the training
  budget is consumed. The previous version froze a `planned` total epoch count from *epoch 1's
  duration*, and epoch 1 is precisely the epoch that gets cut short (persistent-OOM guard, cudnn
  autotune, allocator warm-up). An aborted 0.9s epoch 1 followed by real 31.7s epochs gave
  `planned ≈ 3500` while ~93 epochs ran, so `cos(π·93/3500) ≈ 1` and **the LR never annealed at
  all** — observed on 4 of 9 datasets. Never reintroduce an epoch-count-based schedule.
- **The cosine only reaches ~0 when epochs are fine-grained relative to the budget.** The LR is set at
  the *start* of each epoch and the loop exits when the next epoch wouldn't fit, so roughly one
  epoch's worth of budget is always left unannealed. Negligible over hundreds of epochs (measured
  final LR ~0.00001), material over few: an 8-epoch run ends at **~9% of `BASE_LR`**. Not a bug, but
  it means a dataset with few, long epochs never completes its anneal — worth remembering before
  attributing a poor score on such a dataset to anything else.
- **Guard the clock-unavailable sentinel.** `_remaining()` returns `1e9` when `clock.check()` fails.
  Fed into the progress formula that pins `progress` at 0.0 forever. `clock_ok = budget < 1e8`
  detects it and holds the LR at `BASE_LR` instead of annealing against a fiction.
- **Only a *completed* epoch updates `epoch_time`.** A partial epoch badly under-estimates the next
  one's cost, and that estimate gates "is there time for another epoch?".
- **`nas_batch_size_hint` must be honoured.** NAS records it when the searched cell only fits below
  the loader's default batch size; the Trainer applies it in `__init__`. Ignoring it made epoch 1
  walk into the persistent-OOM guard *by construction* — which is what produced the partial epoch 1
  above. The runtime OOM guard remains a safety net, not the primary mechanism.
- **No training batch of size 1, ever.** `helpers.safe_drop_last(n, bs)` picks `drop_last` so the
  final batch is never a single sample; the Trainer floors rebuilt batch sizes at 2 and skips any
  batch shorter than 2. BatchNorm raises on one sample in train mode; that `RuntimeError` is *not* an
  OOM, so it propagates past the OOM handler and the outer guard silently ends training for the whole
  dataset. Triggers only when `n_train % bs == 1`. **Known gap:** `safe_drop_last` cannot help when
  `bs == 1`, and `DataProcessor._choose_batch_size` can still return 1 for `n_train ∈ {2,3}` — every
  batch is then skipped and the model gets *zero* gradient steps, silently. See §7.
- **The validation split is served through one fixed seeded permutation**
  (`VALID_PERM_SEED`, applied lazily via `_ArrayDataset.order`, no array copy). The valid loader
  stays `shuffle=False` so epoch-to-epoch numbers stay comparable — but `Trainer._evaluate` and
  `NAS._quick_val_acc` both stop on a deadline, and without the permutation a truncated pass scores a
  fixed *prefix*. On a dataset whose validation split is sorted by class, that prefix is one class and
  the number is meaningless: it picks the wrong checkpoint *and* the wrong genotype.
- **The test split is never permuted or shuffled, and `predict()` always returns exactly `n_test`
  labels in order.** The harness matches predictions to labels by position and asserts the test loader
  is neither shuffling nor dropping.
- **Weight decay applies to matrix-valued parameters only.** `_build_optimizer` splits on
  `p.dim() > 1`, so BatchNorm affine terms and biases land in a `weight_decay=0.0` group. Splitting on
  dimensionality rather than module type cannot miss a custom block inside a searched cell.
- **`label_smoothing` is constructed defensively.** It arrived in torch 1.10 and the server's version
  is unknown (§3), so `_build_criterion` falls back to plain `CrossEntropyLoss` on `TypeError`.
- **Truncated and full validation accuracies are not equal-fidelity.** `_evaluate(time_budget=...)`
  may score on a few hundred samples while an earlier epoch scored on the full split, and
  `if val >= best_val` then compares them directly. The permutation makes the truncated sample
  *unbiased*, but not equal-*variance*. Known, unfixed — see §7.

---

## 5. Working Conventions for Claude on This Project

- Stick to the rough plan above and the proposal's approach unless Daniel and/or Samuel direct otherwise.
- When generating code, add short, precise **English** comments explaining the logic.
- When editing existing files, output **only the changed files**, not the full set of six.
- Follow nascompetition.com and nascompetition.com/rules over general NAS knowledge or training
  data when the two conflict — the competition can and does change details year to year.
- If uncertain about a harness/rules detail, say so and ask rather than guessing — a wrong
  assumption here risks an automatic -10.
- **Keep this file current.** Whenever you change the pipeline, update `CLAUDE.md` in the same pass
  — §4 for behaviour changes, §4a for new invariants, §6 for anything that must happen before
  submission, §7 for the remaining backlog (tick items off there; don't restate them elsewhere).
  Standing instruction from Samuel (2026-07-27); don't wait to be asked.
- Record *why* a fix exists and the evidence for it, not just what changed — several of the fixes in
  §4a look like removable complexity until you know which bug they prevent.
- Verify changes by running them, not by inspection — the bugs found so far (LR schedule, batch-of-1,
  prefix-biased validation, calibration timing) were all invisible in review and obvious in a run.
  Where practical, write the test so it **reproduces the old bug first**, then shows the fix.

---

## 6. Before the final submission is zipped

- **Remove the `metadata['time_limit']` testing override in `main.py`'s `load_dataset_metadata()`**
  (§3) so real per-dataset limits apply again. Deliberate local-testing shortcut — easy to forget
  precisely because it's intentional rather than a bug someone trips over.
- **⚠ Do that and reset the dataset metadata in the same change.** The `metadata` files currently
  carry smoke-test budgets (`"time_limit": 0.016` ≈ 58s for AddNIST/Chesseract, `0.013` ≈ 47s for
  Sudoku). They're inert *only* while the override is in place. Removing the override on its own
  silently gives every dataset ~1 minute.
- **Strip `main.py` and `score.py` from the submission directory** (§3) — the harness overwrites
  files with those names.
- **Fix `requirements.txt` locally** (it never ships — see §3): `sklearn` is the deprecated stub
  package that now refuses to install and is what broke the last local scoring run; it should be
  `scikit-learn`. Also the `-f .../cu113/...` index line sits above a `+cu132` torch pin, which
  cannot resolve.
- **Re-benchmark and actually score it before submitting.** Everything since `Final_Score 2.167`
  (Adaline −1.182 / Chester −0.128 / Sokoto +3.477) is unmeasured, and that run used 3h per dataset
  — 3× what Phase 2 actually gives.

---

---

## 7. Findings, reverted work, and planned work

### 7a. What happened on 2026-07-27 — read this before re-attempting any of it

A large batch of changes (macro growth, successive-halving verification, a dataset-size parameter
guard, SWA, and a fallback ladder in `DataProcessor`) was implemented and then **reverted** after a
four-way code review found that several rested on a broken measurement and two introduced fresh `-10`
paths. What survives is a deliberately small, individually verified set: **clock-driven cosine LR,
`nas_batch_size_hint`, the validation permutation, the batch-of-one guards, the calibration timing
fix, label smoothing, and the weight-decay split** (all documented in §4/§4a).

The full reverted diff is preserved as a patch at
`<scratchpad>/full-changes-before-revert.patch` (session-temp — if it matters later, commit it to a
branch). Everything needed to rebuild it properly is recorded below.

**The lesson, stated plainly:** every reverted item was individually plausible and locally tested, and
they were landed in sequence without a single scored benchmark between them. The failures were not in
the ideas but in unvalidated *premises* shared across them. Land fewer changes per benchmark.

### 7b. Confirmed `-10` paths that exist RIGHT NOW (highest priority)

Both are live in the current code. Neither is triggered by the three local datasets (all float32,
contiguous), so a local benchmark will not surface them — but an unseen dataset could.

1. **`data_processor.py`, `_to_4d_float` → `process()` has no error handling at all.** Any exception
   propagates to `main.py` → dataset failed. Confirmed escapes: a **negative-stride** numpy array
   (`torch.as_tensor` raises `ValueError`, e.g. any array saved from a flipped/transposed view without
   `.copy()`), an object/ragged dtype array, non-numeric labels, and zero training samples. Note a
   fallback ladder was tried and reverted — it did *not* fix this, because every rung called the same
   `_to_4d_float`. The real fix is `np.ascontiguousarray(...).astype(np.float32)` plus catching
   `ValueError`, at the conversion site.
2. **`nas.py`, the "no batch available" handler:** `TinyNet(int(s[1]), ..., h=int(s[2]), w=int(s[3]))`
   indexes `metadata['input_shape']` unguarded — inside the handler that exists to prevent a crash.
   `IndexError` there escapes `NAS.search()` → `-10`. **This is not hypothetical: `input_shape` is
   demonstrably unreliable in this competition's own data** (see 7c).

### 7c. Facts about the shipped data (verified 2026-07-27)

`metadata['input_shape']` is **not** a reliable mirror of `train_x`. Measured:

| dataset | metadata `input_shape` | actual `train_x.shape` |
|---|---|---|
| AddNIST | `[50000, 3, 28, 28]` | `(45000, 3, 28, 28)` — **sample count wrong** |
| Chesseract | `[49998, 12, 8, 8]` | `(49998, 12, 8, 8)` — ok |
| Sudoku | `[50000, 1, 9, 9]` | `(50000, 9, 9)` — **3-D array, 4-element metadata** |

Always derive shapes from the actual arrays/loader, never from `input_shape`. Also note **GameOfLife
has only 5,000 training samples** (vs ~45–50k for the others), so "all competition datasets are ~50k"
is false — a small-dataset regime exists in this family.

### 7d. Why the macro-growth and successive-halving work was reverted

Do not simply re-apply the patch; the premise below has to be fixed first.

- **The epoch projection was 2–4.5× optimistic.** `_projected_epochs = train_budget / (t_step ×
  n_batches × CELL_COST_FACTOR)` used `t_step` from `_measure_step_time`, which excludes the
  DataLoader and augmentation (§4). Measured on AddNIST: projected 411 epochs, real 182. The loader is
  **56%** of a real epoch. `CELL_COST_FACTOR = 2.0` was entirely consumed by the cell-vs-ResidualBlock
  gap alone (measured 1.88×), leaving nothing for loader overhead. Consequence: `MIN_TRAIN_EPOCHS =
  100` was really a ~50-epoch guard. **The fix is to measure a real mini-epoch through the actual
  loader — not to inflate `CELL_COST_FACTOR`, which was already doing two unrelated jobs.**
- **Successive halving inherited the same bad `t_batch`.** On a real AddNIST run the planned ladder
  cost ~1.9× its estimate and truncated: rung 1 re-trained the rung-0 leader and compared it against
  nothing. At budgets ≤300s it collapses to `rungs=[1]` — one candidate "wins" by default, which is
  exactly what `top_geno` returns for free, after burning ~6% of the dataset budget. Any future
  attempt needs a `len(rungs) == 1` short-circuit.
- **The growth ladder aborted on first rejection.** Every rejection path (`over_budget`, `oom`, epoch
  floor) did `break`, so a failed *width* candidate never fell through to try *depth*. On GameOfLife,
  `c0=64 n=3` and `n=4` both fit under budget and were never attempted — roughly half the permitted
  capacity unused.
- **The parameter guard's constant was fragile.** `MAX_PARAMS_PER_SAMPLE = 1700` was calibrated
  against three datasets, but AddNIST's real `n_train` is 45,000 (not the 50,000 in metadata), making
  its true ratio identical to Gutenberg's — so there was **one** calibration point, cleared by 2.6%.
  If reattempted: the guard must be counted against `MAX_PARAM_GENOTYPE` (conv3x3 on every edge),
  which **is** an exact upper bound — verified by brute force over 400 genotypes, zero exceed, zero
  tie — and the constant is meaningless without stating that reference model (ResidualBlock is
  **3.14–3.17×** smaller, a typical searched cell 1.85×).
- **SWA was cut on its merits, not just for surface area.** `SWA_START_FRACTION = 0.75` opened the
  averaging window where the cosine had already decayed the LR to 15.5% of base and falling to zero.
  Textbook SWA averages a trajectory held at *constant high* LR; averaging ~200 near-identical points
  returns approximately that point, which best-checkpointing already provides. It also fired
  non-deterministically (1 run in 5 skipped it) because the reserve omitted the post-loop validation
  pass, and `_recompute_bn` had no wall-clock deadline while four copies of the model were resident on
  GPU — a plausible `-10`. **If ever revisited, it needs a constant LR during the averaging window.**

### 7e. Other findings from the review, still unfixed

- **`_estimate_value_cardinality` allocates a full-length `randperm`** to draw 20,000 samples —
  measured **3.1s and ~847 MB** on AddNIST, ~84% of `process()` runtime, scaling linearly with dataset
  size. `torch.randint(0, numel, (max_check,))` gives the same signal at ~zero cost. Easy win.
- **Horizontal flip is enabled on glyph data and measurably hurts.** `_CARDINALITY_THRESHOLD = 32`
  classifies any 8-bit image as "continuous → photo-like → mirror is safe", which catches AddNIST — a
  *digit* dataset, where a mirrored digit is not that digit. A/B over 3 seeds: **−1.33%, −1.45%,
  −0.73%**, consistent sign (short runs, so direction is established but not magnitude). Augmentation
  also costs **23.4 s/epoch vs 5.1 s** of pure dataloading (`num_workers=0`, all serialised in front of
  the GPU). Adaline is one of the datasets currently scoring below benchmark. Strongly worth an A/B at
  full length.
- **`_choose_batch_size` can return 1** (`n_train ∈ {2,3}`), and every batch is then skipped by the
  size-<2 guards → zero gradient steps, silently. Floor it at 2, as `_rebuild_train_loader` already
  does.
- **`_choose_batch_size`'s "memory-aware" probe is effectively inert** — all three local datasets hit
  the 512 cap, and it has no knowledge of the model NAS will build. Either make it model-aware after
  `search()` or stop describing it as memory-aware.
- **`DataProcessor` never reads the clock.** Measured 7.4s of a 60s budget (12%). The one component
  with no harness protection is also outside the time accounting.
- **`train_x` is materialised as float32 twice concurrently** in `process()` (`train_t` is still live
  when `_ArrayDataset` re-converts). Free for float32 input; a 4× double copy for uint8, which is the
  normal on-disk format for images. `del train_t` before building the dataset costs nothing.
- **Host-RAM OOM is not caught by `_is_oom`** — it matches `'out of memory'`, but a CPU allocator
  failure says `DefaultCPUAllocator: can't allocate memory`.
- **`torchvision` is imported unguarded at module level.** If it were absent on the server, *all three*
  datasets fail at import (−30). Near-certainly fine (it's in the starter kit), but it is the hardest
  dependency in the submission.
- **`main.py`'s `grace_time` is cosmetic** — it prints "predictions will still be ran" and then calls
  `fail_dataset()` anyway, before `predict()`. There is no grace in practice; the margin must never
  reach zero.
- **Cosmetic:** `nas.py`'s `if counter % 1000 == 0: print(...)` fires on every iteration while
  `counter == 0`, since it only increments on a successful score.

### 7f. Planned work

**1. Re-benchmark the reverted baseline and score it.** Nothing else should land first. The last
scored result (`Final_Score 2.167`; Adaline −1.182 / Chester −0.128 / Sokoto +3.477) predates every
change and used 3h per dataset — 3× the Phase 2 allotment.

**2. Fix the two `-10` paths in 7b.** Small, and independent of any accuracy work.

**3. A/B horizontal flip** (7e) — plausibly the largest single score item identified, and it is
*pre-existing* behaviour rather than anything recently added.

**4. Training recipe, remainder.** Warmup (implemented and verified during the reverted batch —
linear over a budget *fraction*, with a floor so a low-epoch-count dataset doesn't waste its first
epoch at ~0 LR; reverted only to keep the benchmark clean) and optionally mixup/cutmix gated on the
`is_continuous` signal `_build_train_transform` already computes.

**5. Re-attempt time-aware macro growth** — only after fixing the projection basis (7d). The
underlying motivation stands: an overnight Adaline run trained 1009 epochs on a 1.2M-param model,
plateaued at ~83% val, and still scored below benchmark. That run was capacity-limited, not
time-limited.

**6. Equal-fidelity validation comparison** (§4a) — record `n_scored` alongside each accuracy and
refuse to displace a checkpoint scored on substantially more samples.

### 7g. Test-suite notes

The consolidated suite for the current baseline lives at `<scratchpad>/test_baseline.py` (run it with
the `stx_r` interpreter, passing the `submission_baseline` path as argv[1]). Two habits worth keeping,
both of which caught real bugs: **reproduce the old bug first** in the same test that shows the fix,
and **assert against a measured quantity**, not a shape. Two habits worth avoiding, both of which hid
real bugs here: assertions with an `or ... is not None` clause that is unconditionally true, and
`check(True, ...)` tautologies that only "fail" by raising. Also: a suite that passes once is not a
suite that passes — SWA's flakiness (1 in 5) was invisible until the same test was run five times.
