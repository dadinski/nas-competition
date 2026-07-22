"""
proxies.py - Step 3

Training-free proxies used to rank candidate architectures before any real
training happens. Both take an UNTRAINED model and are cheap (a single
forward/backward pass):

  * naswot_score: how well the network's ReLU activation patterns separate
    different inputs (higher = better).
  * synflow_score: how well signal propagates through the network, computed
    on an all-ones input, independent of the actual data (higher = better).
"""

import numpy as np
import torch
import torch.nn as nn


def naswot_score(model, x):
    """Score based on ReLU activation patterns (Hamming-distance kernel)."""
    device = x.device
    model.to(device)
    model.eval()

    codes = []

    def hook(_module, _inp, out):
        codes.append((out > 0).float().flatten(1))

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.ReLU)]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    if not codes:
        return 0.0
    full = torch.cat(codes, dim=1)               # [B, total_units]
    k = full @ full.T + (1 - full) @ (1 - full).T
    k = k.cpu().numpy().astype(np.float64)
    _sign, logdet = np.linalg.slogdet(k + 1e-6 * np.eye(k.shape[0]))
    return float(logdet)


def synflow_score(model, in_ch, h, w, device):
    """Data-independent proxy: sum(|param * grad|) on an all-ones input,
    with all parameters temporarily taken as absolute values (linearization)."""
    model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)

    signs = [p.data.sign() for p in model.parameters()]
    for p in model.parameters():
        p.data = p.data.abs()

    inp = torch.ones(1, in_ch, h, w, device=device)
    out = model(inp)
    out.sum().backward()

    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += (p.data * p.grad).abs().sum().item()

    # restore original signs (magnitude was never changed)
    for p, s in zip(model.parameters(), signs):
        p.data = p.data * s
    model.zero_grad(set_to_none=True)

    return score
