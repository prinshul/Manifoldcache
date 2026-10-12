# ManifoldCache

Training-free, data-free feature caching for constraint-manifold diffusion models.

| Model | Folder |
|---|---|
| [Med-DDPM](https://github.com/mobaidoctor/med-ddpm) | [`Med-DDPM/`](Med-DDPM) |
| [Proteina](https://github.com/NVIDIA-BioNeMo/proteina) | [`Proteina/`](Proteina) |
| [UniMoMo](https://github.com/kxz18/UniMoMo) | [`UniMoMo/`](UniMoMo) |
| [DiffCSP++](https://github.com/jiaor17/DiffCSP-PP) | [`DiffCSP-PP/`](DiffCSP-PP) |

## How to implement ManifoldCache in your model

ManifoldCache needs only the noise schedule and the block structure of your denoiser.
For each block, keep its last output and the step at which it was computed, and reuse the output while the block's stride has not expired.

```python
# --- 1. Safe-caching boundary from the noise schedule (no data, no training) ---
T_star = max(t for t in range(T) if alpha_bar[t] >= 0.5)

# --- 2. Depth-adaptive stride: deeper blocks are reused for longer ---
def stride(depth, t):                   # depth in [0, 1]: 0 = shallowest, 1 = deepest
    if t > T_star:                      # high-noise phase: always recompute
        return 1
    return floor(S_shallow + (S_deep - S_shallow) * depth)

# --- 3. Cached block call: one integer comparison per block per step ---
cache, last = {}, {}

def run_block(name, block, depth, t, *inputs):
    if name in cache and last[name] - t < stride(depth, t):
        return cache[name]              # reuse cached output
    out = block(*inputs)                # recompute and refresh cache
    cache[name], last[name] = out, t
    return out

# --- 4. Denoiser forward with caching (UNet example) ---
def cached_forward(x, t, emb):
    h, skips = x, []
    for i, blk in enumerate(encoder_blocks):
        h = run_block(f"enc{i}", blk, i / (n_enc - 1), t, h, emb)
        skips.append(h)
    h = run_block("mid", middle_block, 1.0, t, h, emb)
    for i, blk in enumerate(decoder_blocks):
        h_in = concat(h, skips.pop())
        h = run_block(f"dec{i}", blk, 1 - i / (n_dec - 1), t, h_in, emb)
    return out_head(h)

# --- 5. Sampling loop: unchanged except for the forward call ---
cache.clear(); last.clear()
x = randn(shape)
for t in reversed(range(T)):
    eps = cached_forward(x, t, time_embed(t))
    x = sampler_step(x, eps, t)         # DDPM / DDIM / your own update rule
```

For flow-matching models that integrate from noise (t = 0) to data (t = 1), the safe phase is every step with log-SNR(t) >= 0 (see `Proteina/`).
