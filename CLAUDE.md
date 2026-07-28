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

**Status (2026-07-28):** steps 1–5 are implemented in `submission_baseline/`; step 6 has *not*
happened. The pipeline is **robust** — a 13-dataset run at a realistic 1h/dataset budget
(`outputs/output_1h_each_27_07`) finished with **zero failures and zero timeouts**, `Final_Score
32.078`. Robustness is no longer the bottleneck; *accuracy* is.

⚠ **That 32.078 is inflated.** Cryptic, Sudoku and Voxel ship `"benchmark": 0.0` in their metadata,
and `score.py` computes `(raw − benchmark) · 10/(100 − benchmark)`, so any accuracy above zero scores
positive on them. They contributed **+17.57 of the 32.08**. Over the 10 datasets with a real
benchmark the total is **+14.50, mean +1.45/dataset**, with 5 of 10 *below* benchmark. Always quote
the 10-dataset number; the three placeholders tell us nothing.

**Where to pick up (as of end of 2026-07-28): run the benchmark. See §7f.1 for the command, what to
compare against, and the written-down predictions.** Everything landed that day is verified by tests
and by end-to-end smoke runs but has **no score attached**. Landed: the cross-dataset batch-size
collapse, `stem_stride` macro sizing, two `-10` paths, NAS verification fixes, train-accuracy logging
with an automatic saturation/overfitting diagnosis, and **ensembling on surplus budget gated on the
saturation signal** (§4a, §7f.2 — the largest measured effect so far). A full code review of all four
pipeline files was done in the same pass (§7i).

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
- **Macro skeleton:** adaptive — channel width, blocks per stage, downsampling steps and **stem
  stride** are derived from the actual batch shape and available GPU memory (OOM probing), not fixed.
  `derive_macro()` returns `(c0=32, n=2, d, stem_stride)` where
  `d_full = floor(log2(min(H,W)/4))`, `d = clamp(d_full, 0, 3)` and `stem_stride = 2**(d_full − d)`;
  `_find_macro()` then **shrinks** c0 → n → d until a forward+backward fits (never the stem stride —
  reducing it makes the model both slower *and* larger). It only ever shrinks — see §7d for why the
  time-aware growth experiment was reverted.
- **Every input ≥32×32 now ends at a 4×4 feature map.** Before `stem_stride` existed, `d` was capped
  at 3, so total downsampling was at most 8× *however large the input was*: a 128×128 dataset ran its
  whole network at 128/64/32/16 and cost ~28× a 28×28 one per sample. See §7h for the measurement and
  the score it cost. Inputs ≤32×32 get `stem_stride == 1` and are bit-for-bit unaffected.
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
- **Surplus budget is spent on an ensemble of INDEPENDENTLY RE-INITIALISED models, gated on the
  saturation signal.** When a member goes `max(8, 0.2 × epochs_run_in_member)` epochs without a
  validation improvement *and* a whole further member still fits, `train()` snapshots that member's
  best weights to host RAM, calls `reset_parameters()` across the module tree, and trains another;
  `predict()` averages their softmax outputs. Three details are load-bearing:
  - **Re-initialisation, not warm restarts.** The textbook SGDR/"Snapshot Ensembles" recipe was
    measured and gives **nothing** here (§7f.2) — a memorised model re-memorises after a restart and
    every member computes the same function. Do not "simplify" this back to carrying weights over.
  - **The patience is RELATIVE.** A constant would fire on datasets that are still improving slowly:
    AddNIST's best checkpoint is at epoch 166 of 197, i.e. a 31-epoch dry spell. Verified against all
    13 datasets — fires on exactly the 5 that saturate, never on the 5 that use their budget nor the
    3 that run out of time. Confirmed live: AddNIST at a 600s budget produces **0 members** and is
    behaviourally unchanged.
  - **Later members are sized by CONVERGENCE time, not saturation-detection time.** The patience tail
    is detection cost a fresh member need not repeat, and the members are then stretched to fill the
    budget. Sizing by elapsed time instead left 25% of a 600s Chesseract budget unused and bought 2
    members where the budget supported 7 (measured: 2 members +1.95pt vs 7 members +3.43pt).
  `predict()` scores members one at a time, timing the first pass and re-checking the clock before
  each subsequent one, so running short costs *members*, not the dataset — and it still returns
  exactly `n_test` labels in order. Snapshots live on the **CPU**; several models resident on the GPU
  is the pattern that made the reverted SWA attempt a plausible `-10` (§7d).
- **Every epoch line logs train accuracy, the train−val gap and mean train loss, and `train()` ends
  with a one-line diagnosis.** Added 2026-07-28 because the 13-dataset run logged *only* validation
  accuracy, which hid its largest single problem: 5 of 13 datasets had already memorised the training
  split and peaked in the first few percent of the budget, and establishing that needed a separate
  by-hand re-run of each dataset (§7h). `_log_training_summary` prints `best val X% at epoch N (P% in)`
  plus a `SATURATED` note when `P ≤ 25` and a `MEMORISING` note when the gap ≥ 15pts at ≥95% train
  accuracy — so the next run answers "out of time or out of ideas?" and "over- or under-fitting?" on
  its own. Verified on Chesseract: both notes fire, `best val 54.27% at epoch 5 (17% in), final train
  99.98%`. **The train figure is a RUNNING one** (accumulated from the training forward passes, on
  augmented batches, in train mode) — that is what makes it free (7.4s/epoch vs a 7.2s baseline,
  inside the noise). It matched the clean 99.99% on Chesseract only because that dataset's
  augmentation is noise-only; on a `continuous` dataset with pad+crop+flip it will read lower. Never
  add a second clean pass over the training set for this — it would roughly double epoch cost.
- **Batch sizing must release the allocator cache before reading free GPU memory.**
  `DataProcessor._query_free_gpu_mem_mb()` calls `torch.cuda.empty_cache()` and counts
  `memory_reserved − memory_allocated` as available. The harness runs **all datasets in one process**
  (`main.py` loops over them), so when dataset 2 or 3 is sized, dataset 1's model and activations are
  dead in Python but still held by PyTorch's caching allocator — and `torch.cuda.mem_get_info()`
  reports *driver*-free memory, which does not count that pool. Measured on this box: **0 MB "free"
  with 7878 MB reserved and 16 MB actually live**, collapsing the batch size from 512 to the floor
  of 4. This is not theoretical: it hit both datasets that followed a memory-heavy one in the
  13-dataset run (§7h). Phase 2/3 run three datasets per process, so two of three are exposed.
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

### 5.0 Standing rules from Samuel (2026-07-28) — these override convenience

1. **Work step by step, and verify each step's output before building on it.** After every step, check
   the result for faults — a measurement that contradicts a known quantity, a script that silently
   produced nothing, a timing taken while another job held the GPU. **If a step's output is faulty,
   discard it and redo it; never carry a suspect number forward.** Two concrete traps already hit in
   this project: piping a long run through `grep`/`tail` buffers *all* output until the process exits
   (so it looks hung and you learn nothing until the end — write to a file and poll instead), and
   running two GPU jobs concurrently silently corrupts any timing measurement (kill one and re-measure
   rather than reasoning about the contaminated number).
2. **Keep `CLAUDE.md` continuously up to date, to the standard that a brand-new session reading only
   this file knows exactly where to pick up and what the rules are.** Update it in the *same* pass as
   the change, not afterwards: §4/§4a for behaviour and invariants, §6 for pre-submission must-dos, §7
   for the backlog and for findings (tick items off there rather than restating them elsewhere).
   Record *why* a change exists and the evidence for it, not just what changed.

### 5.1 General conventions

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

**What landed on 2026-07-28 (unmeasured — see §7h for the evidence behind each):** the two `-10` paths
(§7b), the cross-dataset batch-size collapse (§4a), `stem_stride` macro sizing (§4), the NAS
`n_verify <= 1` short-circuit and bounded `t_val` probe, plus three small items from §7e (batch floor
of 2, `randint` cardinality sampling, `del train_t`). Each was verified by a run that reproduces the
old behaviour first — `<scratchpad>/test_changes.py`, all passing — and they target **disjoint sets of
datasets**, which is what makes landing them together still attributable. **Mixup was deliberately NOT
landed** despite a positive first measurement; see §7f.2 for why.

### 7b. The two `-10` paths — FIXED 2026-07-28

Both are closed; kept here so nobody reopens them. Neither was triggered by the local datasets (all
float32, contiguous), so a local benchmark would never have surfaced them — an unseen dataset could.

1. **`data_processor.py`, `_to_4d_float` → `process()` had no error handling at all.** Fixed at the
   conversion site: `_to_4d_float` retries through `np.ascontiguousarray(...).astype(np.float32)` on
   `ValueError`/`TypeError`, which normalises negative strides (any array saved from a flipped or
   transposed view without `.copy()`), non-native byte order, and odd dtypes; it also now accepts 2-D
   `[N, F]` input. `process()` additionally wraps `_process()` and degrades to `_minimal_process()`
   (no stats, no augmentation, no memory query, fixed batch 32). **Note the honest limit, documented in
   the code:** `_minimal_process` still calls `_to_4d_float`, so it is *not* a second line of defence
   for a genuinely unconvertible array — that is precisely the flaw that sank the reverted fallback
   ladder, and it is why the real fix had to be at the conversion site.
2. **`nas.py`, the "no batch available" handler** indexed `metadata['input_shape']` unguarded, inside
   the handler that exists to prevent a crash. Now length-checked with per-field defaults and a
   `TinyNet → MinimalNet` fallback. `input_shape` is demonstrably unreliable in this competition's own
   data (see 7c), so this path was a live risk, not a hypothetical one.

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
- ~~**`_choose_batch_size` can return 1**~~ — FIXED 2026-07-28, floored at 2.
- ~~**`train_x` is materialised as float32 twice concurrently**~~ — FIXED 2026-07-28, `del train_t`.
- ~~**`_estimate_value_cardinality` allocates a full-length `randperm`**~~ — FIXED 2026-07-28,
  `torch.randint` instead (see the first bullet above, kept for its measurement).
- **`_choose_batch_size`'s probe is still not model-aware** — it now reads memory correctly (§4a) but
  has no knowledge of the model NAS will build, and most datasets still land on the 512 cap. Making it
  model-aware would have to happen after `search()`.
- **`DataProcessor` never reads the clock.** Measured 7.4s of a 60s budget (12%), and **148s on
  Myofibre** in the 1h run. The one component with no harness protection is also outside the time
  accounting.
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

### 7f. Planned work — START HERE

**1. Re-benchmark and score the 2026-07-28 changes. NOTHING ELSE SHOULD LAND FIRST.** Six substantive
changes are in across four files and none has a score attached — precisely the situation §7a is about.

```bash
make submission=submission_baseline all
```

13 datasets × 1h ≈ 13h; `main.py` already forces `time_limit = 1`, so no edits are needed. Keep
Cryptic/Sudoku/Voxel in for robustness coverage (odd shapes: 1×6×768, 20×20×20, 12-channel) but
**ignore their scores** — their benchmarks are 0.0. Compare the **10-dataset subtotal (+14.50)**,
not the 32.078 headline.

The changes target *disjoint* dataset groups, which is what makes one run still attributable.
Predictions, written down in advance so the run can falsify them:

| dataset | prediction | if wrong, suspect |
|---|---|---|
| Myofibre | large gain, −3.96 → positive | `stem_stride` |
| CIFARTile, GeoClassing | moderate gain | `stem_stride` |
| Chesseract, Language, Windspeed, Voxel, Gutenberg | ensembling fires; Chesseract/Language cross benchmark | val→test gap on the ensemble |
| AddNIST, MultNIST, GameOfLife | **unchanged, 0 members** | the saturation gate |
| Sudoku, Cryptic | may *drop* | batch 4→512 against a fixed `BASE_LR` |

The logs now self-diagnose — the `[Trainer] summary:` line per dataset says which group it landed in,
so grepping `summary:` and `ensembl` is the fastest read of the result.

**1b. Known gap to check in that output:** the ensemble gain is so far measured only on *validation*
data. Chesseract and Language are exactly where a val→test gap would hurt most, and MultNIST showed a
4.8-point val→test drop in the previous run. This benchmark is what settles it.

**2. Attack the saturation group — this is where the remaining score is.** Five datasets waste 88–99%
of their hour after fully memorising the training split (§7h(i)), and four of them score below
benchmark. Two independent levers, in order of confidence:
   - **Regularisation, specifically mixup.** It is the one strong augmentation that assumes *nothing*
     about what an axis means — it only convex-combines whole samples and their labels — which is
     exactly the property this competition demands, and unlike flip/crop it is safe on a one-hot
     board. Applies to the `quantized/categorical` datasets that currently receive **no effective
     augmentation at all**. A/B harness: `<scratchpad>/ab_mixup.py`.
     **First measurement (Chesseract, seed 0, 60 epochs, same init per arm):** baseline best val
     **53.67%**, mixup α=0.2 **55.99%**, mixup α=0.4 **55.54%** — i.e. **+2.3 / +1.9 points**, both
     alphas agreeing in direction. **NOT LANDED**, deliberately: one seed only, and α=0.2's best came
     at epoch 3, which looks like it could be the top of a noise distribution rather than a trend.
     Final train accuracy stayed at 99.99% in every arm, so mixup at these strengths *reduces* the
     generalisation gap without actually preventing memorisation — larger α is worth testing too.
     **Next step: repeat over ≥3 seeds and on Language/Gutenberg before landing** (the flip A/B in
     7e used 3 seeds for exactly this reason, and §7g's "a suite that passes once is not a suite that
     passes" applies here).
   - **Spend the surplus on an ensemble — MEASURED, and it is the largest effect found so far.**
     Harness: `<scratchpad>/ab_ensemble.py`. Every arm gets an *identical* 60-epoch budget, so this
     measures how the budget is spent, not how much of it there is. Seed 0, 4 members:

     | dataset | benchmark | single | snapshot (warm restart) | **independent re-init** |
     |---|---:|---:|---:|---:|
     | Chesseract | 57.83 | 54.27 | 54.08 | **58.62** (+4.35) |
     | Language | 85.20 | 80.10 | 79.75 | **85.47** (+5.37) |

     **Both cross their benchmark for the first time**, from −0.64 and −3.63 adj respectively. The
     members are individually *no better* than the single baseline (Chesseract `[.534 .536 .544 .540]`
     vs .543; Language `[.791 .803 .806 .807]` vs .801) — the entire gain is decorrelation, +4.26 and
     +4.79 over the best member.

     **The critical detail, and the reason this had to be measured rather than assumed: members must
     be INDEPENDENTLY RE-INITIALISED.** Warm-restart snapshots — i.e. the textbook "Snapshot
     Ensembles"/SGDR recipe, cosine restarts carrying the weights over — produced **nothing** on both
     datasets (−0.19 and −0.35 vs single; ensemble gain over own best member −0.10 and +0.01), because
     a model that has memorised its training split just re-memorises after the restart and every
     member lands on the same function. Note this is a *different* mechanism from the SWA attempt
     rejected in §7d: that averaged weights, this averages predictions.

     **LANDED 2026-07-28**, gated on saturation — see §4a for the invariants. End-to-end through the
     real `Trainer` on Chesseract at a 600s budget: **7 members, 59.22% vs 55.79%** for the best single
     member (+3.43), **above the 57.83 benchmark**; against production's actual single-model 54.43% the
     gain is ~+4.8. Non-regression verified live: AddNIST at the same budget produces **0 members**, no
     saturation message, unchanged behaviour. Suite: `<scratchpad>/test_ensemble.py`.

     **Still open:** single seed on both the A/B and the end-to-end check, and the gain has only been
     measured on validation data — Chesseract and Language are exactly the datasets where a val→test
     gap would matter most. The scored re-benchmark (item 1) is what settles it.

**2b. HOW MEMBERS ARE SIZED IS THE WEAKEST PART OF THE ENSEMBLING WORK — fix this first after the
benchmark.** Two distinct symptoms, one root cause: `member_budget` is derived from `t_conv =
max(member_best_time, 0.34 × elapsed_member)`, and that is a **noisy** statistic.

- **Run-to-run variance at a fixed budget.** Three runs of `test_ensemble.py` on Chesseract at 600s
  gave **7, 2, and 7 members** (gains +4.03, +1.95, +4.45; the 2-member run also left 235s of 600s
  unused). Chesseract's validation curve is essentially flat (51–55%), so *which* epoch happens to be
  the best is close to random within the plateau — when it lands late, `member_best_time` is large,
  which buys few long members instead of many short ones. Since 4–7 members are worth roughly twice
  what 2 are, this is losing about half the available gain half the time. **Everything still passes
  and every run beats the single model** — this is magnitude, not correctness.
- **It stops absorbing surplus at long budgets.** `k_more` is capped at `MAX_MEMBERS − 1 = 7`, so at
  8h with `t_conv ≈ 40s` and `avail ≈ 28000s` **each member gets 4000s while converging in 40** — back
  to wasting the surplus, the exact problem ensembling was built to solve. Matters for Phase 3, which
  reuses the Phase 2 submission unchanged.

Direction for the fix (needs measurement, do not just apply it): make the member length depend on a
*robust* convergence estimate rather than the argmax epoch — e.g. the first time the member came
within ~1 point of its eventual best — cap `member_budget` at a small multiple of it, and let the
member *count* grow with the budget instead, raising `MAX_MEMBERS`. Host RAM is ~38 MB/member; the
binding cost is `predict()`, one test pass per member, and `_ensemble_predict_reserve()` already
scales the reserve with member count, so that part is in place.

**3. Couple the learning rate to the batch size.** Surfaced by the smoke run (§7g0): `BASE_LR` is a
fixed 0.01 no matter whether the loader batch is 4 or 512, so the number of optimizer steps a dataset
gets varies by two orders of magnitude with no compensation. The standard remedy is the linear scaling
rule (`lr ∝ batch_size`, relative to a reference batch). **Measure before landing** — this is exactly
the shape of premise that sank the 2026-07-27 batch (§7a). A cheaper, related knob: the 512 cap in
`_choose_batch_size` is arbitrary and unmeasured; on a small-image dataset a smaller batch buys many
more steps per epoch for little wall-clock.

**4. A/B horizontal flip** (7e) — measured −0.73% to −1.45% over 3 seeds on AddNIST, consistent sign.
*Pre-existing* behaviour, not anything recently added.

**5. Check whether NAS verification carries signal at all.** On AddNIST the verified candidates spread
4.4–10.7% around a 5% random baseline (§7h). If verification-time val accuracy does not correlate with
final trained accuracy, the ~7% of the budget it costs is better spent training. This is a cheap
experiment and could free a large, guaranteed slice of time.

**5b. `MAX_EPOCHS = 5000` is a hard-coded epoch cap in a clock-driven pipeline — OPEN DECISION.**
Samuel declined this change on 2026-07-28; recorded so it stays a decision rather than an oversight.
It is not the training budget (the clock is); it exists only so the loop terminates if `clock.check()`
fails and `_remaining()` returns its `1e9` sentinel. But it binds at long budgets: at the measured
~7.0s/epoch of GameOfLife it is reached after **11h**, Chesseract/Gutenberg/Sudoku at 12.8–13.3h.
Phase 2's 1 hour never reaches it; the starter kit's own `TIME_LIMIT` default is **12 hours** and
Phase 3 is "an unknown amount of runtime". If it binds, training stops with hours unused *and* the
clock-driven cosine frozen partway down, so the final model never had its LR annealed. Proposed fix
if ever wanted: keep 5000 as the no-clock safety valve, use an effectively unbounded cap when
`clock_ok` is true.

**6. Warmup.** Implemented and verified during the reverted batch — linear over a budget *fraction*,
with a floor so a low-epoch-count dataset doesn't waste its first epoch at ~0 LR; reverted only to
keep the benchmark clean.

**7. Re-attempt time-aware macro growth** — only after fixing the projection basis (7d), and note that
§7h partly re-frames it: the datasets that looked capacity-limited (Chesseract, Language) turn out to
be *over*-fitting, so growing them would make things worse, not better. The remaining candidates for
growth are the ones that were still improving when the clock stopped.

**8. Equal-fidelity validation comparison** (§4a) — record `n_scored` alongside each accuracy and
refuse to displace a checkpoint scored on substantially more samples.

### 7g0. End-to-end verification of the 2026-07-28 changes (smoke run, 8 min/dataset)

Run through the real `evaluation/main.py` on AddNIST → Myofibre → Sudoku at **480s per dataset** —
1/7.5 of the production budget — specifically to check the two structural fixes in the harness path.

**Myofibre, the `stem_stride` case, is decisive:**

| | production (3600s budget) | smoke (480s budget) |
|---|---|---|
| macro | `d=3` → final map **16×16** | `d=3 stem_stride=4` → final **4×4** |
| epoch time | **525s** | **45s** (11.7× faster) |
| `DataProcessor` | 148s | 10s |
| `t_val` calibration | 131.7s | 1.46s |
| candidates verified | **1** ("nothing to compare against") | **8** |
| best val reached | 82.86% (5 epochs, 1 hour) | **85.70%** (6 epochs, 8 minutes) |

So: **7.5× less wall-clock and +2.8 points of validation accuracy.** The measured 11.7× epoch speedup
against the 16× predicted from conv FLOPs is the expected gap — the DataLoader and the CPU-side
pad/crop/flip on 128×128 are unchanged by this fix and now dominate what remains. AddNIST was
verified unaffected (`stem_stride=1`, cost identical by forward-hook measurement in `test_changes.py`).

**Second smoke run (7 min/dataset), after ensembling landed**, through the same real `main.py` — the
gate behaving correctly end-to-end, not just in a unit test:

```
========== Dataset  Adaline ==========
[Trainer] summary: 12 epochs, best val 47.85% at epoch 12 (100% in), final train 50.18%
        (no saturation, no ensembling - still improving when the clock stopped)
========== Dataset  Chester ==========
[Trainer] saturated after 12 epochs (112s, best at 38s) with 266s left -> up to 5 more members of 41s each
[Trainer] member 1 done (best val 54.44%) ... member 4 done (best val 55.18%)
[Trainer]   -> ensembling 5 independently initialised members in predict()
[Trainer] ensembled 5/5 members over 9999 test samples
```

Both finished inside budget (355s and 348s of 420s), no errors. Note `best at 38s` versus `112s`
elapsed — that difference is exactly the patience tail that later members are sized to skip.

**Sudoku confirms the batch-size fix works — and exposes a second problem it does not solve.**
Running straight after memory-heavy Myofibre, it got `batch_size=512` (production: **4**), and epochs
went 168s → **7.4s**, 17 epochs/hour → 43 epochs in 8 minutes. But its validation accuracy came out
**worse**: peaked ~15.2% at epoch 19 versus production's 34.1% and still climbing. The reason is
arithmetic, and it matters:

| | batch | steps/epoch | epochs | **total optimizer steps** |
|---|---:|---:|---:|---:|
| production (60 min) | 4 | 12,500 | 17 | **212,500** |
| smoke (8 min) | 512 | 97 | 43 | 4,171 |
| smoke scaled to 60 min | 512 | 97 | ~322 | **31,282 — 6.8× fewer** |

**`BASE_LR` is a fixed 0.01 regardless of batch size**, so moving from batch 4 to 512 is a 128×
increase in batch with no compensating LR change: each epoch is 22× cheaper but makes far less
progress per sample seen. Sudoku's pathological batch of 4 was *accidentally* giving it 200k+ updates.

**This does not mean the batch fix is wrong** — `batch_size=4` was a non-deterministic accident of
whatever the *previous* dataset happened to leave in the allocator, and 11 of 13 datasets already ran
at 512, so 512 is the regime `BASE_LR = 0.01` is implicitly tuned for. But it does mean **Sudoku and
Cryptic may score lower** after this change, and neither has a real benchmark, so we have *no*
evidence about how a small batch behaves on a scored dataset. The principled follow-up is LR scaling
(§7f.3), not reinstating the accident.

### 7g. Test-suite notes

Suites live in the session scratchpad; run them with the `stx_r` interpreter, passing the
`submission_baseline` path as argv[1]:
- `test_baseline.py` — the older consolidated suite.
- `test_changes.py` — covers the 2026-07-28 changes: `derive_macro`/stem-stride table, a
  **regression guard that 28×28 cost is unchanged**, aliasing check (stem must use stride-2 convs
  only), measured conv-FLOP reduction via forward hooks, `_to_4d_float` on negative-stride /
  big-endian / uint8 / 3-D / 2-D arrays, the batch-size collapse (old formula reproduced alongside
  the new one), and that `process()` returns loaders instead of raising.
- `smoke.sh <minutes> <dataset>...` — end-to-end run through the real `evaluation/main.py` on a chosen
  subset at a short budget. **Dataset order matters**: put a memory-heavy dataset before a small one
  or the cross-dataset batch-size collapse cannot appear. Like the Makefile it *copies* the datasets
  into `smoke_pkg/` (9.7 GB for three of them) — `rm -rf smoke_pkg` afterwards; it is gitignored.
- `test_train_logging.py <sub> <dataset> <seconds>` — checks the epoch lines parse, that `gap` really
  equals `train − val`, that train accuracy rises and loss falls, that the `SATURATED`/`MEMORISING`
  notes fire on Chesseract, **and that median epoch time is not inflated** by the accumulation.
- `test_ensemble.py <sub> <dataset> <seconds>` — drives the real `Trainer`: asserts the members hold
  **genuinely different weights** (a silently failed `reset_parameters()` would give clones and a
  useless ensemble), that the ensemble beats the single model on held-out data, that `predict()`
  returns exactly `n_test` both ways, and that it degrades to fewer members instead of overrunning
  when the clock is nearly exhausted. **Use 600s+ on Chesseract.** This test was initially flaky at
  420s and the fix was to the *test*, not the code: batch shuffling and the noise augmentation are
  unseeded, so member 1's saturation point moves run to run, and near a marginal budget there is
  sometimes no room for a second member — where the gate is behaving correctly by declining. It now
  asserts the **coherence invariant** (`saturation logged ⇔ members created`) always, and the
  ensemble-beats-single payoff only when an ensemble actually formed. A worked example of §7g's
  "a suite that passes once is not a suite that passes".
- `ab_ensemble.py` — the single/warm-restart/re-init comparison at an identical epoch budget per arm.
- `diag_fit.py` / `ab_mixup.py` — train/val-gap diagnosis and the regularisation A/B. Both are now
  largely redundant for *diagnosis*: the pipeline logs this itself. Keep them for clean-number
  measurement and for A/B work.

Two habits worth keeping,
both of which caught real bugs: **reproduce the old bug first** in the same test that shows the fix,
and **assert against a measured quantity**, not a shape. Two habits worth avoiding, both of which hid
real bugs here: assertions with an `or ... is not None` clause that is unconditionally true, and
`check(True, ...)` tautologies that only "fail" by raising. Also: a suite that passes once is not a
suite that passes — SWA's flakiness (1 in 5) was invisible until the same test was run five times.

### 7h. The 13-dataset 1h/dataset run of 2026-07-27 — full analysis (done 2026-07-28)

Log: `outputs/output_1h_each_27_07`. Scores: `scoring/final_results.json`. **13/13 completed, no
failure, no timeout** — every dataset used ~3500s of its 3600s and left ~60s for `predict()`. The
robustness work is done; nothing below is about crashes.

**Scores (⚠ Cryptic/Sokoto/Volga have `benchmark: 0.0`, so their scores are meaningless — see §1):**

| dataset | codename | benchmark | raw | adj | epochs | best val @ epoch |
|---|---|---|---:|---:|---:|---|
| GameOfLife | conway | 47.53 | 99.81 | **+9.96** | 433 | 399 |
| GeoClassing | Sadie | 80.33 | 91.78 | **+5.82** | 76 | 72 ← still rising |
| AddNIST | Adaline | 89.85 | 93.65 | **+3.74** | 197 | 166 |
| CIFARTile | Caitie | 47.01 | 56.58 | +1.81 | 67 | 62 ← still rising |
| MultNIST | Mateo | 90.87 | 92.30 | +1.57 | 179 | 148 |
| Windspeed | windspeed | 13.49 | 13.14 | −0.04 | 269 | **29** |
| Gutenberg | Gutenberg | 40.98 | 40.23 | −0.13 | 383 | **23** |
| Chesseract | Chester | 57.83 | 55.12 | −0.64 | 393 | **4** |
| Language | LaMelo | 85.20 | 79.83 | **−3.63** | 341 | **9** |
| Myofibre | Myopia | 87.93 | 83.15 | **−3.96** | **5** | 5 ← still rising |
| *Cryptic* | *Cryptic* | *0.0* | *70.39* | *+7.04* | *20* | *19* |
| *Voxel* | *Volga* | *0.0* | *71.01* | *+7.10* | *266* | *31* |
| *Sudoku* | *Sokoto* | *0.0* | *34.33* | *+3.43* | *17* | *17 ← still rising* |

**The single clearest pattern: every dataset scoring below benchmark reached its best validation
accuracy in the first ~10% of its budget, or never got enough budget at all.** There is no dataset
that used its hour well and still lost. Two disjoint failure modes, needing opposite fixes:

**(i) Saturation — 5 datasets waste 88–99% of the hour.** Chesseract peaks at epoch 4 of 393, Language
at 9 of 341, Windspeed 9 of 269, Gutenberg 23 of 383, Voxel 31 of 266. **Diagnosed by measurement, not
inference** (the production logs print only val accuracy, so this was previously invisible):

| dataset | train acc | val acc | gap | when |
|---|---:|---:|---:|---|
| Chesseract | **99.99%** | 53.5% | **+46.5** | epoch 32 |
| Language | **100.00%** | 79.2% | **+20.8** | epoch 10 |

Both **completely memorise the training split**, then spend 300+ further epochs at zero benefit. This
is an over-fitting problem: more time and more capacity cannot help, only regularisation can. Note
that all five are `quantized/categorical` datasets, for which `_build_train_transform` applies **noise
only** — Gaussian noise at `std=0.03` on binary data is effectively no augmentation at all. So the
group with zero augmentation is exactly the group that memorises.

**(ii) Starvation — the 3 large-image datasets never finished converging.** Myofibre (128×128) got
**5 epochs at 525s each** and was still climbing steeply (78.5 → 82.9); CIFARTile and GeoClassing were
both still improving at their final epoch. Root cause found and fixed (§4, `stem_stride`): `d` was
capped at 3, so total downsampling was ≤8× regardless of input size, and Myofibre ran its whole
network at 128/64/32/16. Analytic conv cost, heaviest cell: **7287 MFLOPs/sample vs AddNIST's 262
(27.9×)** — consistent with the observed 525s vs 14.6s per epoch. With the stem stride it is 455
MFLOPs, a **16× cut**, and every input ≥32×32 now lands at 4×4.

**Other findings from this run:**
- **Batch size collapsed to 4 on exactly the two datasets that followed a memory-heavy one** (Cryptic
  after CIFARTile, Sudoku after Myofibre) → 20 and 17 epochs instead of hundreds. Root-caused and
  fixed; see §4a, and note this is *worse* in the real competition, where 2 of 3 datasets are exposed.
- **NAS verification degenerated to `verifying top 1` on all three large-image datasets** — one
  candidate "verified" against nothing, which is what `top_geno` already returns for free. On Myofibre
  that cost ~270s (a full 131.7s of it just timing one validation pass) out of a budget where one
  epoch cost 525s. Fixed: `n_verify <= 1` now short-circuits, and `t_val` is measured from ≤12 warmed
  batches and extrapolated instead of a full pass.
- **The verification signal is close to noise on some datasets.** On AddNIST the 93 verified
  candidates scored 4.4%–10.7% against a 5% random baseline; the winner at 10.67% may be a real signal
  or may be the top of a noise distribution. Worth a dedicated check: does verification-time val
  accuracy actually correlate with final trained accuracy? If not, the ~7% of budget it costs is
  better spent training.
- **NAS takes a flat 10% of the budget** regardless of whether an epoch costs 6s or 525s. Largely
  mitigated by the `stem_stride` fix (which makes epochs cheap on exactly the datasets where this
  hurt), but the principle is unguarded.
- **Val→test gaps are not uniform:** Mateo 97.12 val → 92.30 test (−4.8), Caitie 60.49 → 56.58 (−3.9),
  while Adaline and Sadie land within 0.4. Partly winner's-curse from taking the max over ~180 noisy
  evaluations, but the size of it suggests genuine split differences. Not currently actionable, but
  don't read a val number as a test number.

### 7i. Code review of the whole pipeline, 2026-07-28

Full read of `data_processor.py`, `model.py`, `nas.py`, `trainer.py` after the day's changes. Findings
fixed in the same pass — listed because each was a real defect, not a style point:

- **`trainer.py`: the training loop's stop condition ignored the ensemble's `predict()` cost.** The
  loop exited at `margin`, which covers *one* test pass, but an N-member ensemble needs N. The last
  member therefore trained right up to `margin` and `predict()` had to drop members it had already
  paid to train. Fixed with a `stop_margin()` local = `margin + _ensemble_predict_reserve()`, used by
  the epoch-start check, the mid-epoch timeout check, and the validation budget. It returns exactly
  `margin` while no ensemble exists, so the single-model path is untouched. **Not a `-10`** — the
  in-`predict()` clock check already made overrun impossible — but pure waste. Measured effect on the
  Chesseract 600s test: ensemble gain rose from +3.43 to **+4.46** (60.08% vs 55.62%).
- **`data_processor.py`: `_minimal_process` did not apply the validation permutation.** That
  permutation is a correctness property, not an optimisation (§4a): a deadline-truncated validation
  pass otherwise scores a fixed prefix, which is meaningless on a class-ordered split — and the
  Trainer and NAS truncate validation regardless of which rung built the loaders. Added, guarded, and
  reading `np.asarray(...).shape[0]` rather than constructing an `_ArrayDataset` (which would
  materialise the whole split as float32 just to read a length). It also now derives the real majority
  `fallback_label` instead of defaulting to 0.
- **`trainer.py`: `predict()` could call `load_state_dict(None)`** when the ensemble path bailed out
  before `train()` had set `_best_state_for_fallback`. The exception was swallowed, leaving the last
  *member's* weights loaded rather than the best ones. Now an explicit `is not None` check that logs
  if restoring fails.
- Stale comments corrected: a reference to a `_log_train_metrics` that never existed, a
  `_maybe_close_member` that is actually inline in `train()`, and a "supported 5 members" that the
  measurement had since made 7.

Checked and found correct (no change needed): the bounded `t_val` probe's extrapolation arithmetic and
its unbound-`t0` edge case; `Skeleton`'s stem for `stem_stride == 1` being byte-identical to the old
single conv; `BatchNorm` running stats being reset by `reset_parameters()` (so re-initialised members
really are fresh); `load_state_dict` of CPU snapshots into a CUDA model preserving device; and the
`n_val_batches == 0` path degrading without crashing.
