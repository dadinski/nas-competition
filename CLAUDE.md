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

**The 2026-07-28 changes are MEASURED and they worked: the 10-dataset real subtotal went
+14.50 → +24.61, and every one of the 10 is now at or above its benchmark.** Full analysis of that run
and of Daniel's 3.5h run in **§7j**. The pipeline is also confirmed clean on torch 1.10.1 with a 3 GB
card, which is the closest proxy we have for the evaluation server.

**Where to pick up (as of 2026-07-29): §7f.1 — fix the saturation gate's false positives.** It fired
on 9 of 13 datasets rather than 5; on AddNIST that cost **−3.53** and left 32% of the budget unused.
The fix is two constants, already validated against real per-epoch curves (§7j, issue 1). Second
priority is member sizing (§7f.2b), which is now measured to waste ~90% of each member's time and
13–32% of every ensembling dataset's budget.

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
  saturation signal.** When a member goes `max(10, 0.75 × epochs_run_in_member)` epochs without a
  validation improvement *and* a whole further member still fits, `train()` snapshots that member's
  best weights to host RAM, calls `reset_parameters()` across the module tree, and trains another;
  `predict()` averages their softmax outputs. Three details are load-bearing:
  - **Re-initialisation, not warm restarts.** The textbook SGDR/"Snapshot Ensembles" recipe was
    measured and gives **nothing** here (§7f.2) — a memorised model re-memorises after a restart and
    every member computes the same function. Do not "simplify" this back to carrying weights over.
  - **The patience is RELATIVE, and the fraction is 0.75** — i.e. fire only once the best checkpoint
    sits in the first 25% of what has run, the same definition as the `SATURATED` log note. It was
    0.2 and that was too eager: it cost **−3.53 on AddNIST** in the 2026-07-28 run (§7j). Validated by
    replaying the shipped `_saturation_patience` against the clean per-epoch curves of all 13
    datasets — **0 misclassified**, versus 5 with the old constants. Do not re-tune this against
    summary statistics; see §7j for why that validation is invalid.
  - **Later members are sized by CONVERGENCE time, not saturation-detection time**, and their length
    is capped at `MEMBER_LENGTH_FACTOR × t_conv` so that surplus budget buys MORE members rather than
    longer ones. Sizing them as `avail / MAX_MEMBERS` made members ~10× longer than needed as soon as
    the member cap bound — which it did even at 1h (Chesseract 7 × 440s while converging in 33s;
    Sudoku at 3.5h 7 × 1678s while converging in 178s) and left 13–32% of every ensembling dataset's
    budget unused. `MAX_MEMBERS` is 24; host RAM is ~38 MB per snapshot and the binding cost is one
    test pass per member in `predict()`, for which `_ensemble_predict_reserve()` is estimated at the
    planned member count up front rather than the current one.
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

**What is actually IN the zip** (verified 2026-07-29 by building it and running the harness against
the extracted contents): `cd $(submission); zip -r ../submission.zip *` bundles **only
`submission_baseline/`**. So `evaluation/main.py`, `evaluation/score.py`, the `datasets/` metadata and
`requirements.txt` are **not** part of the submission and cannot affect it — the organisers supply
their own `main.py`/`score.py`. Several items below are therefore about *local testing fidelity*, not
submission validity; they are marked accordingly so nobody blocks a submission on them.

- **⚠ SHIPS IN THE ZIP: delete `submission_baseline/__pycache__/` before zipping.** `zip -r *` globs
  it in — measured ~165 KB of `.pyc` across two Python versions. Not dangerous (Python revalidates a
  `.pyc` against its source's mtime and size, so a stale one is ignored, and a mismatched magic number
  is simply skipped), but it is junk in the deliverable and leaks the interpreter version.
  `rm -rf submission_baseline/__pycache__` immediately before `make zip`.
- **Multi-file submissions are fine — no need to fold `model.py`/`proxies.py` into `helpers.py`.**
  The template README specifies only the three *classes* and their methods, never a file layout, and
  the template's own `helpers.py` is literally `# use this however you need`. The build step is
  `cp -R $(submission)/* package`, so every module lands beside `main.py` and the flat same-directory
  imports resolve. Proven end-to-end: the extracted zip plus `main.py` ran GameOfLife to completion.
  Keep the modules split — it is easier to review than one large file.
- **Confirm `main.py` and `score.py` are absent from the submission directory** (§3) — the harness
  overwrites files with those names. Currently neither is present.
- *(local testing only — does NOT ship)* The `metadata['time_limit'] = 1` override in
  `evaluation/main.py` and the smoke budgets in the dataset `metadata` files (`0.016` ≈ 58s, `0.013`
  ≈ 47s). These only distort **local** runs. Reset them before any benchmark you intend to trust, but
  they are not a submission blocker.
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

**0. RE-BENCHMARK FIRST — the 2026-07-29 gate and member-sizing fixes are unmeasured.**
`make submission=submission_baseline all`, 13 × 1h. Predictions, written down so the run can falsify
them:

| check | expectation | if wrong, suspect |
|---|---|---|
| `grep -c 'saturated after'` | **5**, not 9 | the 0.75 fraction |
| AddNIST | 0 members, back to ~+3.7 | the gate fix failed |
| GameOfLife | 0 members, ~+9.9 | the gate fix failed |
| Chesseract, Language, Gutenberg, Windspeed, Voxel | 15–24 members; unused budget <10% (was 13–32%) | `MEMBER_LENGTH_FACTOR` / `MAX_MEMBERS` |
| real-10 subtotal | ~+28 (from +24.61) | — |

**0b. Split the work across the two machines — they answer different questions.** Samuel's box runs
the 13 × 1h benchmark above (Phase 2's actual regime, comparable to the +24.61 baseline). Daniel's
machine should NOT repeat it: at 3 × 3.5h it is the only place the **member-sizing fix can be tested
in the regime where it was most broken**, and he already has an exact baseline from 2026-07-29 —
same machine, same budget, same datasets, only the code changed. Recommended set:

| dataset | why | baseline raw | before → predicted |
|---|---|---:|---|
| Sudoku | most extreme member-sizing case | 25.80 | 7 × 1686s → **24 × 512s** |
| Voxel | second saturator, exact baseline | 76.33 | 7 × 1513s → **12 × 960s** |
| Language *(swap in for CIFARTile)* | a saturator with a REAL benchmark; Sudoku and Voxel both have `benchmark: 0.0` so only their raw accuracy means anything | — | expect ≥ the 85.77 raw it got at 1h |

He must set `metadata['time_limit'] = 3.5` in his copy of `evaluation/main.py` (the scaffold in §3
forces 1). Checks: member counts rise as predicted, unused budget falls below 10%, and the run stays
clean on torch 1.10.1 — his stack remains our closest proxy for the evaluation server.

⚠ **`MAX_MEMBERS = 24` is the least-validated thing in this change.** It rests on "more members is
better", which is true from 1 → 4 (measured, §7f.2) but **not** measured from 7 → 24, and shorter
members are individually weaker. The 1800s Chesseract check gave best val 55.95% with 22 members
against ~55.5–55.9 with 7 — not a clear win. If the saturating datasets do not improve in this
benchmark, `MAX_MEMBERS` is the first thing to put back down.

**1. LANDED 2026-07-29 — the saturation gate and member sizing are both fixed.** Constants now
`PATIENCE_FRACTION = 0.75`, `MIN_PATIENCE = 10`, `MAX_MEMBERS = 24`, `MEMBER_LENGTH_FACTOR = 1.5`
(§4a). Replaying the shipped `_saturation_patience` against the clean per-epoch curves of all 13
datasets misclassifies **0**, versus 5 before. Member sizing replayed on the production numbers:

| dataset | before | after |
|---|---|---|
| Chesseract 1h | 7 × 440s (13.3× convergence) | 24 × 133s (4.0×) |
| Windspeed 1h | 7 × 440s (5.9×) | 24 × 132s (1.8×) |
| Sudoku 3.5h | 7 × 1686s (9.5×) | 24 × 512s (2.9×) |
| Voxel 3.5h | 7 × 1513s (2.5×) | 12 × 960s (1.6×) |

**UNMEASURED on score.** Expected: AddNIST and GameOfLife return to single-model (recovering ~+3.6),
the five genuine saturators keep ensembling with more members, and the 13–32% of budget those
datasets were leaving unused mostly disappears. The next benchmark should check exactly that — grep
`saturated after` and confirm it appears on five datasets, not nine.

**1b. Detection now costs more, and WHEN it fires still varies run to run.** Two live Chesseract runs
with the shipped constants: one fired at **epoch 41** (~332s in), the other at **epoch 12** (~92s in).
Same cause as item 2 — the drought is measured from the argmax, so whether an early epoch happens to
set a high running max shifts detection by 3×. Consequences measured:

- 600s budget, fired late (ep 41): only ~250s left, so **0 members** — the gate correctly declined.
- 1800s budget, fired early (ep 12): **22 members of 70s each** (before the fix: 7 × 440s), best val
  55.95%, i.e. the member-sizing fix does what it was meant to.

At a 1h budget there is ample runway either way, so this is not expected to matter for Phase 2 — but
it does mean **short-budget runs may no longer ensemble at all**, and `test_ensemble.py` now needs
**1800s+** on Chesseract rather than 600s.

**2. `t_conv` is still derived from the argmax epoch and is noisy.** Three runs of `test_ensemble.py`
on Chesseract at a fixed 600s gave **7, 2 and 7 members** (gains +4.03, +1.95, +4.45). Every run beat
the single model — magnitude, not correctness — and `MEMBER_LENGTH_FACTOR` damps it, but a robust
estimate (e.g. the first epoch within ~1 point of the eventual best) would remove it.

**3. Mixup — the remaining lever for the memorisers.** Ensembling raised the saturating datasets to
about par but did NOT stop them memorising: Gutenberg still ends at 100.00% train vs 40.4% val
(+59.6), Sudoku at 3.5h at 100.00% vs 24.3% (+75.7), Chesseract 100.0% vs 54.4% (+45.5). Mixup is the
one strong augmentation that assumes nothing about what an axis means, so it is safe on a one-hot
board, and these datasets currently receive **no effective augmentation at all** (noise-only).
First measurement (Chesseract, seed 0, 60 epochs, same init per arm): baseline **53.67%**,
α=0.2 **55.99%**, α=0.4 **55.54%** — both alphas agree in direction. **Not landed:** one seed, and
α=0.2's best came at epoch 3, which could be the top of a noise distribution. Test over ≥3 seeds AND
**jointly with ensembling** — both attack overfitting, so measuring mixup alone would over-state what
it adds. Harness: `<scratchpad>/ab_mixup.py`. Note windspeed is the opposite case (38.6% train) and
must not be regularised further.

**4. Couple the learning rate to the batch size — and note this is a PORTABILITY issue, not just a
Sudoku curiosity.** `BASE_LR` is a fixed 0.01 whether the loader batch is 4 or 512, so the number of
optimizer steps varies by orders of magnitude with no compensation. Sudoku demonstrated the cost
directly (raw 34.33 → 23.13 when its batch went 4 → 512, §7g0). **The reason to raise its priority:
`_choose_batch_size` sizes from the GPU actually present, so the batch differs per machine — Daniel's
3 GB card produced 310 and 202 where this box produces the 512 cap (§7j). The evaluation server's GPU
is unknown, so `BASE_LR` may well be mis-scaled there for every dataset.** The standard remedy is the
linear scaling rule (`lr ∝ batch_size` relative to a reference batch). **Measure before landing** —
this is exactly the shape of premise that sank the 2026-07-27 batch (§7a). Related knob: the 512 cap
is arbitrary and unmeasured.

**5. A/B horizontal flip** (7e) — measured −0.73% to −1.45% over 3 seeds on AddNIST, consistent sign.
*Pre-existing* behaviour, not anything recently added.

**6. Check whether NAS verification carries signal at all.** On AddNIST the verified candidates spread
4.4–10.7% around a 5% random baseline (§7h). If verification-time val accuracy does not correlate with
final trained accuracy, the ~7% of the budget it costs is better spent training. This is a cheap
experiment and could free a large, guaranteed slice of time.

**6b. `MAX_EPOCHS = 5000` is a hard-coded epoch cap in a clock-driven pipeline — OPEN DECISION.**
Samuel declined this change on 2026-07-28; recorded so it stays a decision rather than an oversight.
It is not the training budget (the clock is); it exists only so the loop terminates if `clock.check()`
fails and `_remaining()` returns its `1e9` sentinel. But it binds at long budgets: at the measured
~7.0s/epoch of GameOfLife it is reached after **11h**, Chesseract/Gutenberg/Sudoku at 12.8–13.3h.
Phase 2's 1 hour never reaches it; the starter kit's own `TIME_LIMIT` default is **12 hours** and
Phase 3 is "an unknown amount of runtime". If it binds, training stops with hours unused *and* the
clock-driven cosine frozen partway down, so the final model never had its LR annealed. Proposed fix
if ever wanted: keep 5000 as the no-clock safety valve, use an effectively unbounded cap when
`clock_ok` is true.

**7. Warmup.** Implemented and verified during the reverted batch — linear over a budget *fraction*,
with a floor so a low-epoch-count dataset doesn't waste its first epoch at ~0 LR; reverted only to
keep the benchmark clean.

**8. Re-attempt time-aware macro growth** — only after fixing the projection basis (7d), and note that
§7h partly re-frames it: the datasets that looked capacity-limited (Chesseract, Language) turn out to
be *over*-fitting, so growing them would make things worse, not better. The remaining candidates for
growth are the ones that were still improving when the clock stopped.

**9. Equal-fidelity validation comparison** (§4a) — record `n_scored` alongside each accuracy and
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
- `test_labels.py <sub>` — non-0-based (`1..K`) and non-contiguous (`{0,3,9}`) labels must still
  train. Reproduces the old `IndexError` at the old head width first, and asserts
  `n_outputs == num_classes` on the normal 0-based case so the fix is a proven no-op there.
- `test_ensemble.py <sub> <dataset> <seconds>` — drives the real `Trainer`: asserts the members hold
  **genuinely different weights** (a silently failed `reset_parameters()` would give clones and a
  useless ensemble), that the ensemble beats the single model on held-out data, that `predict()`
  returns exactly `n_test` both ways, and that it degrades to fewer members instead of overrunning
  when the clock is nearly exhausted. **Use 1800s+ on Chesseract** — since the 2026-07-29 gate change
  600s is no longer enough runway for a second member to fit (§7f.1b). This test was initially flaky at
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

### 7j. The 2026-07-28 benchmark + Daniel's 3.5h run — analysis (done 2026-07-29)

Logs: `outputs/output_1h_each_28_07` (13 × 1h, this machine) and
`outputs/output_3sets_3.5h_29_07.txt` (3 × 3.5h, Daniel: torch **1.10.1**/cu113, torchvision 0.11.2,
GTX 1060 **3 GB**). **Zero failures in both.**

**Result: the 10-dataset real subtotal went +14.50 → +24.61 (+10.10). Every one of the 10 is now at
or above its benchmark** (lowest: AddNIST +0.22). Headline 32.078 → 41.397, but quote the 10.

| dataset | adj old | adj new | Δ | driver |
|---|---:|---:|---:|---|
| Myofibre | −3.96 | **+3.07** | **+7.03** | `stem_stride` (5 → 81 epochs) |
| Language | −3.63 | +0.39 | +4.01 | ensembling, crossed benchmark |
| Chesseract | −0.64 | +0.42 | +1.06 | ensembling, crossed benchmark |
| Gutenberg | −0.13 | +0.84 | +0.96 | ensembling |
| CIFARTile | +1.81 | +2.59 | +0.79 | `stem_stride` |
| Windspeed | −0.04 | +0.21 | +0.25 | ensembling |
| GameOfLife | +9.96 | +9.92 | −0.05 | — |
| GeoClassing | +5.82 | +5.67 | −0.15 | — |
| MultNIST | +1.57 | +1.29 | −0.27 | — |
| **AddNIST** | +3.74 | **+0.22** | **−3.53** | **false-positive ensembling — see below** |

**§7f.1b is ANSWERED: the ensemble gain transfers to test data.** `summary:` reports the best single
checkpoint; the submitted predictions are the ensemble. test − best_val averages **+3.33 on the 9
ensembled datasets and −2.69 on the 4 that were not** — so the ensemble is worth about **+6.0 points
of test accuracy** over the best single model. Not a validation artefact.

**ISSUE 1 — the saturation gate fires on noisy plateaus. It ran on 9 of 13 datasets, not the 5
predicted, and the one false positive that mattered cost −3.53.**
AddNIST hit 84.97% at epoch 43, then oscillated 79.8–84.0 for ten epochs *while train accuracy climbed
91 → 94%*. `patience = max(8, 0.2·53) = 10`, so it fired, split the budget, and the run ended with
**1138s (32%) unused**. It was plainly not saturated: member 2 reached 87.05% from scratch in 1082s,
and the 27/07 single-model run reached 92.92%.

**The original validation of this gate was invalid and that is the real lesson.** It simulated the
rule from `(best_epoch, total_epochs)` summary pairs, which silently assumes the running best equals
the final best. Re-run against the actual per-epoch curves, the shipped rule fires on Adaline (ep 67),
Caitie (29), conway (28), Sadie (64) and Mateo (59) too — matching what production did. **Validate a
rule against the trajectory it consumes, never against a summary of it.**

**Fix, validated on the clean 27/07 single-model curves:** `PATIENCE_FRACTION 0.2 → 0.75`,
`MIN_PATIENCE 8 → 10`. Measured drought fraction `(n − best)/n` ever reached: saturating datasets
**0.82–0.99**, budget-using datasets **≤0.67** (conway) — a real gap, and any threshold in (0.67, 0.82]
separates them. 0.75 is its midpoint and is stable across frac ∈ [0.70, 0.80] × floor ∈ [10, 20], so it
is a plateau rather than a knife-edge. It also has a principled reading identical to the `SATURATED`
note already logged: *fire only when the best checkpoint sits in the first 25% of what has run*. It
still fires early enough to leave 55–96% of the budget for further members. ⚠ Tuned on 10 datasets —
the width of the plateau is the argument that it generalises, not the fit itself.

**ISSUE 2 — `MAX_MEMBERS = 7` forces absurdly long members. This is §7f.2b, now measured in
production, and it is worse than predicted: it binds at 1h, not just at 8h.**

| run | dataset | members × length | converges in | wasted per member |
|---|---|---|---|---|
| 1h | Chesseract | 7 × 440s | 33s | ~92% |
| 3.5h (Daniel) | Sudoku | 7 × 1678s | 178s | ~89% |
| 3.5h (Daniel) | Voxel | 7 × 1504s | 602s | ~60% |

**ISSUE 3 — every ensembling dataset left 13–32% of its budget unused** (440–1138s each, **1.4 h across
the 9**), via the "no room for another member" break. Same root cause as issue 2: members are too long
to pack. The 4 non-ensembling datasets used 98% of theirs.

**Daniel's run confirms 3.5× the budget only helps the dataset that is not saturating**, which is
exactly what issues 2–3 predict:
CIFARTile 60.75 → **67.95** raw (+7.20, adj +2.59 → +3.95), Sudoku 23.13 → 25.80 (+2.67),
Voxel 75.37 → 76.33 (+0.96).

**PORTABILITY: Daniel's environment is the closest thing we have to the evaluation server and the
submission is clean on it.** torch 1.10.1 + torchvision 0.11.2 + a 3 GB card: no exception, no AMP
fallback, no `label_smoothing` fallback, no OOM, no TinyNet/MinimalNet fallback, no batch-size
shrink. `_choose_batch_size` correctly scaled to the smaller card (310 and 202 rather than the 512 cap
this machine gets), and `stem_stride=2` applied on 64×64. The §3 version-portability work is validated.

**Other observations:**
- **Sudoku behaves exactly as predicted by §7g0**: raw 34.33 → 23.13 with batch 4 → 512. Confirms the
  LR/batch coupling item (§7f.3) is real; its benchmark is fake so the score is not the point.
- **Windspeed is the one dataset that UNDER-fits** — final train **38.6%**, val 16.1%. It cannot fit
  its own training split, so regularisation is the wrong lever there; it is a different problem class
  from the memorisers.
- **Sudoku at 3.5h memorises completely**: 100.00% train vs 24.3% val, gap +75.7. The single strongest
  case for mixup (§7f.2).
- NAS picks a different genotype run-to-run (AddNIST 877,876 vs 1,178,036 params), so small per-dataset
  deltas (±0.3) are search noise, not signal.

### 7k. Generalization audit — is the pipeline over-fitted to the 13 local datasets? (2026-07-29)

Asked by Samuel, and the right question: the competition explicitly forbids tuning to the bundled
examples, and several constants were introduced by *fitting them to those examples*. Every tunable in
the pipeline, classified honestly.

**A. Dataset-agnostic by construction — no concern.** The fallback cascades; the clock-driven cosine;
distrusting `input_shape` (§7c); branching augmentation on measured value *cardinality* rather than on
shape; batch size derived from the GPU actually present (**proven portable — Daniel's 3 GB card chose
310/202 where this box chooses 512**); `stem_stride` targeting a 4×4 final map (this made us *more*
conventional, not less — 32× total reduction is the ImageNet norm); the validation permutation; the
weight-decay split on `p.dim() > 1`; the NASBench-201 search space; `WEIGHT_DECAY`/`LABEL_SMOOTHING`
at textbook values.

**B. Fitted to the local datasets — the actual risk.**

1. **`PATIENCE_FRACTION = 0.75` is the most over-fitted constant in the pipeline.** It was chosen by
   sweeping against 10 local datasets that *I labelled* "saturating" or "not". Three things bound the
   risk, and they should be checked before anyone re-tunes it: (i) it is a wide plateau, not a point
   fit — anything in frac 0.70–0.80 × floor 10–20 separates the classes, and the measured class gap is
   real (0.67 vs 0.82); (ii) it restates as a shape claim independent of the data — *fire only when the
   best checkpoint is in the first 25% of the run*; (iii) **the failure modes are asymmetric.** A false
   negative just yields the pre-ensembling single-model pipeline and costs nothing relative to the
   2026-07-27 baseline; a false positive costs real score (AddNIST, −3.53). 0.75 is the conservative
   direction, so an unseen dataset that this misjudges most likely loses an *opportunity*, not points.
2. **`_CARDINALITY_THRESHOLD = 32` + horizontal flip is a KNOWN generalization defect, still unfixed.**
   It encodes "many distinct values ⇒ photo ⇒ mirroring is safe", which is simply false for glyphs,
   text, spectrograms, or any directional axis. It already misfires on AddNIST — a digit dataset —
   costing a measured −0.73 to −1.45 over 3 seeds. An unseen dataset of rendered characters or
   time-frequency data would be misclassified exactly the same way. **This is the clearest existing
   over-specialization and it is pre-existing, not something ensembling introduced** (§7f.5).
3. **`BASE_LR = 0.01` is implicitly tuned for batch 512** — see §7f.4; the batch varies with the GPU,
   so this is a portability risk on an unknown server, not just a Sudoku curiosity.
4. `MAX_MEMBERS = 24`, `MEMBER_LENGTH_FACTOR = 1.5`, and the `0.34 × elapsed_member` floor inside
   `t_conv` are reasoned rather than fitted, but are unvalidated magic numbers.

**C. Assumptions the local data CANNOT test — these worry me more than B.**

1. ~~**`num_classes` trusted from metadata**~~ — **FRAMING CORRECTED, then FIXED 2026-07-29.** The
   original claim ("metadata might miscount the classes") was wrong, and Samuel was right to push
   back: `num_classes` is the task *specification*, so a wrong count is the organisers' bug and breaks
   every competitor equally. That is NOT the same as `input_shape`, where the errors found in §7c are
   descriptive and harm nobody.

   The real risk is narrower and nobody's fault: **label VALUES that are not 0-based.** Labels `1..K`
   with a perfectly correct `num_classes = K` give a head of width K (indices 0..K-1), and target K
   raises `IndexError: Target K is out of bounds` — **not** a `RuntimeError`, so it escapes
   `NAS._fits` and the Trainer's OOM guard: NAS falls to `MinimalNet`, training dies on batch one, and
   `predict()` returns an untrained model's guesses. Right length, so no `Failed` flag — it just
   scores like noise, which on a high-benchmark dataset floors at −10.

   **Fixed** by `DataProcessor._record_label_facts()`, which derives `n_outputs =
   max(num_classes, max_label + 1)` from the actual labels; `nas.py` uses `n_outputs`. Verified: the
   old head width reproduces the `IndexError` first, the new one trains on 1-based and on
   non-contiguous (`{0,3,9}`) labels, and **`n_outputs == num_classes` on all 13 local datasets**, so
   nothing here changes behaviour. Suite: `<scratchpad>/test_labels.py`.

   **Deliberately NOT a remap to 0..K-1.** That would need the inverse mapping applied in `predict()`,
   and omitting it would silently return `0..K-1` against labels `1..K` — a total loss on a model that
   trained perfectly. Widening the head keeps label values usable directly as output indices, which is
   what `fallback_label` and `predict()` padding already assume.

2. **A pre-existing bug found while testing the above: `np.bincount` rejects non-integer labels.**
   Voxel ships `train_y` as **float64**, so the whole `fallback_label` block fell into its `except`
   and Voxel silently used `fallback_label = 0` instead of its computed majority class. Harmless so
   far only because `predict()` never had to pad on it. Now uses `np.unique`, which tolerates float
   and negative labels. (Voxel's true majority happens to be 0, so no measured behaviour changed.)

3. **`predict()` has no clock check on the single-model path** and `margin` is capped at a flat **60 s**
   regardless of test-set size or budget (`min(0.15·budget, 60)`). Our largest test pass is ~5 s, so
   there is 12× headroom here — but a large, high-resolution unseen test split on a slow server GPU
   could exceed it, and once inside the loop there is no way to abort. The ensemble path checks the
   clock *between* members but never within a pass.
4. `MAX_EPOCHS = 5000` binds above ~11 h (§7f.6b, open decision).

**Verdict (updated after C1/C2 were fixed on 2026-07-29).** The accuracy work — `stem_stride` and
ensembling — is mechanism rather than dataset-fitting and should transfer. Remaining exposure, in
order:

1. **(B2) the flip heuristic** — a *measured* −0.73 to −1.45 on AddNIST, and it will recur on any
   unseen glyph/text/spectrogram-like data. The clearest over-specialization in the pipeline, and it
   pre-dates all the recent work (§7f.5).
2. **(B3) LR/batch coupling** — the batch is sized from whatever GPU is present, so an unknown server
   GPU means an unknown batch and a mis-scaled `BASE_LR` on *every* dataset (§7f.4).
3. **(B1) `PATIENCE_FRACTION = 0.75`** — the most fitted constant, but bounded by the asymmetry: its
   likely failure is a missed opportunity, not lost points.
4. **(C3) `predict()`'s flat 60 s margin with no in-pass clock check** — 12× headroom on our largest
   test split, unknown on an unseen one, and no way to abort once inside the loop.

C1 (non-0-based labels) and C2 (`bincount` on float labels) are closed. Note both were invisible to
every local benchmark: all 13 datasets are 0-based, and the float-label bug only ever suppressed a
`fallback_label` that was never needed. **That is the general lesson of this audit — the local
datasets cannot fail in the ways an unseen one can, so "the benchmark is green" says nothing about
category C.**

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
