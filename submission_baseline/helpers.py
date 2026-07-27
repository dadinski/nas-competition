"""Shared helper functions used by more than one submission module."""


def safe_drop_last(n_samples, batch_size):
    """Choose `drop_last` so a TRAINING loader never yields a final batch of
    one sample.

    BatchNorm raises "Expected more than 1 value per channel when training" on
    a single sample in train mode. That RuntimeError is not an OOM, so the
    Trainer's OOM handler re-raises it and the outer guard then ends training
    for the whole dataset after a single log line - a large, silent score loss
    that only triggers when n_samples % batch_size == 1.

    Keeps the previous behaviour (drop the tail once there are plenty of
    batches) and additionally drops it whenever the remainder would be exactly
    one sample. Returns False when there is only a single batch, so a small
    dataset still gets trained on instead of being dropped entirely.

    NOTE: this cannot help when batch_size == 1 - then every batch is a single
    sample and dropping changes nothing. Callers must floor the batch size at
    2 themselves; see Trainer._rebuild_train_loader.

    Only for train loaders: valid/test must never drop data, and the harness
    asserts the test loader has drop_last=False.
    """
    n_samples = int(n_samples)
    batch_size = max(1, int(batch_size))
    if n_samples <= batch_size:
        return False                      # single (possibly short) batch - keep it
    return (n_samples % batch_size == 1) or (n_samples > 2 * batch_size)
