# Tests, experiments and diagnostics

Everything here runs against `submission_baseline/`, passed as the first argument, so
the pipeline itself stays free of test-only code and nothing here ships in the
submission zip.

```bash
python tests/test_changes.py submission_baseline
```

Use an interpreter that has torch. On the development machine that was the `stx_r`
conda env; `PY=/path/to/python bash tests/smoke.sh ...` overrides it for the shell script.

Two habits are deliberate throughout, and both caught real bugs:

* **Reproduce the old behaviour first, in the same test that shows the fix.** Every bug
  found in this project was invisible in code review and obvious in a run.
* **Assert against a measured quantity, not a shape.** Several suites compare against
  a number recorded from a production run, so a regression shows up as a number moving
  rather than as an exception.

Conversely, two things to avoid, both of which hid real bugs here: assertions with an
`or ... is not None` clause that is unconditionally true, and `check(True, ...)`
tautologies that can only fail by raising.

---

## Suites (`tests/`)

| script | what it covers |
|---|---|
| `test_changes.py <sub>` | `derive_macro` / stem-stride table, a regression guard that 28x28 cost is *unchanged*, an aliasing check (the stem must use stride-2 convs only), measured conv-FLOP reduction via forward hooks, `_to_4d_float` on negative-stride / big-endian / uint8 / 3-D / 2-D arrays, the cross-dataset batch-size collapse (the old formula is reproduced next to the new one), and that `process()` returns loaders instead of raising |
| `test_labels.py <sub>` | non-0-based (`1..K`) and non-contiguous (`{0,3,9}`) labels must still train. Reproduces the old `IndexError` at the old head width first, and asserts `n_outputs == num_classes` on ordinary 0-based labels so the fix is a proven no-op there |
| `test_ensemble.py <sub> <dataset> <seconds>` | drives the real `Trainer`: members must hold genuinely different weights (a silently failed `reset_parameters()` would give clones and a useless ensemble), the ensemble must beat the single model on held-out data, `predict()` must return exactly `n_test` either way, and it must degrade to fewer members rather than overrun when the clock is nearly exhausted. **Use 1800s+ on Chesseract** — since the gate change, 600s is not always enough runway for a second member |
| `test_train_logging.py <sub> <dataset> <seconds>` | the epoch lines parse, `gap` really equals `train - val`, train accuracy rises and loss falls, the `SATURATED`/`MEMORISING` notes match their own stated conditions, and median epoch time is **not** inflated by the accumulation |
| `smoke.sh <minutes> <dataset>...` | end-to-end through the real `evaluation/main.py` at a short budget. **Dataset order matters** — put a memory-heavy dataset before a small one or the cross-dataset batch-size collapse cannot appear. Copies the datasets into `smoke_pkg/` (9.7 GB for three), so `rm -rf smoke_pkg` afterwards |

Note on flakiness: `test_ensemble.py` and `test_train_logging.py` both assert *invariants*
rather than particular outcomes, because batch shuffling and the noise augmentation are
unseeded and saturation timing moves run to run. An earlier version asserted the outcome
directly and was flaky; the fix was to the test, not the pipeline.

## Experiments (`tests/experiments/`)

These produced the measurements the design rests on. They are A/B harnesses, not pass/fail
tests — each holds everything fixed except the one thing under test.

| script | question it answered |
|---|---|
| `ab_ensemble.py` | single model vs warm-restart snapshots vs independently re-initialised members, at an identical epoch budget per arm. Warm restarts gave nothing; independent re-initialisation gave +4.35 / +5.37 points |
| `ab_smalldata.py` | does the saturation threshold transfer to the small-data regime? Gutenberg subsampled to 1/10 as a replica of an organiser dataset, genotype fixed across arms so search variance could not contaminate it |
| `ab_mixup.py` | does mixup help the datasets that memorise? Promising on one seed, never landed |

## Diagnostics (`tests/diagnostics/`)

One-off scripts kept because each documents a measurement that is quoted in `CLAUDE.md`.

| script | what it measures |
|---|---|
| `repro_bs4.py` | the cross-dataset batch-size collapse: driver-free GPU memory reads 0 MB while the previous dataset's allocator pool is still reserved, so the batch size fell 512 -> 4 |
| `check_costaware.py` | what the proxy-only fallback picks versus the cost-aware pick, with model sizes and step times |
| `speed_levers.py` | `cudnn.benchmark` and `num_workers` through the real DataLoader. `num_workers=4` is worth ~1.5x; `cudnn.benchmark` measured nothing |
| `gatecheck.py` | dumps the complete validation curve of a real `Trainer` run so the saturation gate can be judged on the whole run rather than a truncated tail |
| `diag_fit.py` | train-vs-validation gap, used to establish that the plateauing datasets had memorised their training split |

`speed_levers.py` needs its own `if __name__ == '__main__'` guard on Windows — DataLoader
workers re-import the entry script, and without it every worker re-executes the file.
