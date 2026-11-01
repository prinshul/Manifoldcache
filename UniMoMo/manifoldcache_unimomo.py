import functools

import torch


def compute_tstar(transition):
    if hasattr(transition, "var_sched"):
        alpha_bar = transition.var_sched.alpha_bars
    else:
        alpha_bar = transition.c0 ** 2 / (transition.c0 ** 2 + transition.c1 ** 2)
    safe = torch.nonzero(alpha_bar >= 0.5).flatten()
    return int(safe.max()) if safe.numel() > 0 else 0


class LayerCache:
    def __init__(self, nlayers, total_steps, shallow_stride=2, deep_stride=3,
                 warmup_frac=0.10, cooldown_frac=0.05, tag="layers", tstar=None):
        self.nlayers = nlayers
        self.total_steps = total_steps
        self.shallow_stride = shallow_stride
        self.deep_stride = deep_stride
        self.warmup_frac = warmup_frac
        self.cooldown_frac = cooldown_frac
        self.tag = tag
        self.tstar = tstar
        self.cache = {}
        self.last_computed = {}
        self.current_step = -1
        self.total_evals = 0
        self.cached_evals = 0

    def get_stride(self, layer_idx):
        if self.current_step < 0 or self.total_steps <= 1:
            return 1
        progress = 1.0 - self.current_step / max(self.total_steps, 1)
        if self.tstar is not None:
            if self.current_step > self.tstar:
                return 1
        elif progress < self.warmup_frac:
            return 1
        if progress > 1.0 - self.cooldown_frac:
            return 1
        depth = layer_idx / max(self.nlayers - 1, 1)
        return max(1, int(self.shallow_stride + (self.deep_stride - self.shallow_stride) * depth))

    def should_reuse(self, layer_idx):
        if layer_idx not in self.last_computed:
            return False
        return self.last_computed[layer_idx] - self.current_step < self.get_stride(layer_idx)

    def get_cached(self, layer_idx):
        self.total_evals += 1
        self.cached_evals += 1
        return self.cache[layer_idx]

    def store(self, layer_idx, H, V):
        self.total_evals += 1
        self.cache[layer_idx] = (H, V)
        self.last_computed[layer_idx] = self.current_step

    def set_step(self, step):
        self.current_step = step

    def reset(self):
        self.cache.clear()
        self.last_computed.clear()
        self.current_step = -1
        self.total_evals = 0
        self.cached_evals = 0

    def print_stats(self):
        r = self.cached_evals / max(self.total_evals, 1) * 100
        print(f"  L3 Layer Cache [{self.tag}]: {r:.1f}% ({self.cached_evals}/{self.total_evals})")


class GeoCache:
    def __init__(self, geo_stride=3, disp_threshold=0.5, tag="geo"):
        self.geo_stride = geo_stride
        self.disp_threshold = disp_threshold
        self.tag = tag
        self.stats = {k: 0 for k in ("topo_hit", "topo_miss", "batch_hit",
                                     "batch_miss", "bias_hit", "bias_miss")}
        self.reset()

    def set_step(self, s):
        self.current_step = s

    def should_recompute_topo(self, Z_flat):
        if self.topo is None or self.last_topo_step - self.current_step >= self.geo_stride:
            return True
        return (Z_flat - self.last_Z).norm(dim=-1).max().item() > self.disp_threshold

    def should_recompute_batch(self):
        return self.batch_struct is None or self.last_batch_step - self.current_step >= self.geo_stride

    def should_recompute_bias(self):
        return self.attn_bias is None or self.last_bias_step - self.current_step >= self.geo_stride

    def store_topo(self, data, Z_flat):
        self.topo = data
        self.last_topo_step = self.current_step
        self.last_Z = Z_flat.detach().clone()
        self.stats["topo_miss"] += 1

    def get_topo(self):
        self.stats["topo_hit"] += 1
        return self.topo

    def store_batch(self, data):
        self.batch_struct = data
        self.last_batch_step = self.current_step
        self.stats["batch_miss"] += 1

    def get_batch(self):
        self.stats["batch_hit"] += 1
        return self.batch_struct

    def store_bias(self, data):
        self.attn_bias = data
        self.last_bias_step = self.current_step
        self.stats["bias_miss"] += 1

    def get_bias(self):
        self.stats["bias_hit"] += 1
        return self.attn_bias

    def reset(self):
        self.topo = self.batch_struct = self.attn_bias = None
        self.current_step = -1
        self.last_topo_step = self.last_batch_step = self.last_bias_step = -999
        self.last_Z = None
        self.stats = {k: 0 for k in self.stats}

    def print_stats(self):
        def r(h, m):
            return f"{100 * h / max(h + m, 1):.0f}% ({h}/{h + m})"
        s = self.stats
        print(f"  L2 Geo Cache [{self.tag}]: topo={r(s['topo_hit'], s['topo_miss'])}, "
              f"batch={r(s['batch_hit'], s['batch_miss'])}, bias={r(s['bias_hit'], s['bias_miss'])}")


class EpsNetCache:
    def __init__(self, total_steps, eps_stride=2, warmup_frac=0.12, cooldown_frac=0.08,
                 tag="eps", tstar=None):
        self.total_steps = total_steps
        self.eps_stride = eps_stride
        self.warmup_frac = warmup_frac
        self.cooldown_frac = cooldown_frac
        self.tag = tag
        self.tstar = tstar
        self.reset()

    def set_step(self, s):
        self.current_step = s

    def should_reuse(self):
        if self.cached_output is None:
            return False
        progress = 1.0 - self.current_step / max(self.total_steps, 1)
        if self.tstar is not None:
            if self.current_step > self.tstar:
                return False
        elif progress < self.warmup_frac:
            return False
        if progress > 1.0 - self.cooldown_frac:
            return False
        return self.last_step - self.current_step < self.eps_stride

    def get_cached(self):
        self.hits += 1
        return self.cached_output

    def store(self, eps_H, eps_X):
        self.cached_output = (eps_H, eps_X)
        self.last_step = self.current_step
        self.misses += 1

    def reset(self):
        self.cached_output = None
        self.last_step = -999
        self.current_step = -1
        self.hits = 0
        self.misses = 0

    def print_stats(self):
        t = self.hits + self.misses
        print(f"  L1 EpsNet Cache [{self.tag}]: {100 * self.hits / max(t, 1):.1f}% "
              f"full-skip ({self.hits}/{t} steps)")


def _get_transformer(ept_module):
    inner = getattr(ept_module, "encoder", None)
    if inner is not None and hasattr(inner, "n_layers") and "layer_0" in inner._modules:
        return inner
    raise RuntimeError(f"Cannot find Transformer in {type(ept_module).__name__}")


def make_cached_transformer_forward(transformer, layer_cache, geo_cache):
    def cached_forward(H, Z, block_id, batch_id, edges, edge_attr=None,
                       topo_edges=None, topo_edge_attr=None, attn_mask=None):
        from torch_scatter import scatter_mean, scatter_sum
        from utils.nn_utils import graph_to_batch_nx
        from models.modules.GET.tools import _unit_edges_from_block_edges

        Z_flat = Z.view(-1, 3)

        if geo_cache.should_recompute_topo(Z_flat):
            with torch.no_grad():
                edges_c, ea_c = edges, edge_attr
                if topo_edges is not None:
                    mask = edges_c[0] != edges_c[1]
                    edges_c = edges_c.T[mask].T
                    if ea_c is not None:
                        ea_c = ea_c[mask]
                (ur, uc), (bei, _, _) = _unit_edges_from_block_edges(
                    block_id, edges_c.T, Z, k=transformer.sparse_k)
            eai = ea_c[bei] if ea_c is not None else None
            if topo_edges is not None:
                ur = torch.cat([ur, topo_edges[0]])
                uc = torch.cat([uc, topo_edges[1]])
            if topo_edge_attr is not None:
                eai = torch.cat([eai, topo_edge_attr])
            geo_cache.store_topo((ur, uc, eai), Z_flat)
        else:
            ur, uc, eai = geo_cache.get_topo()

        edge_vec = Z_flat[ur] - Z_flat[uc]
        edge_dis = torch.norm(edge_vec, dim=-1)
        dis_feat = transformer.rbf(edge_dis)
        edge_feat = torch.cat([H[ur], H[uc], dis_feat, eai], dim=-1)
        edge_scaler = transformer.edge_mlp(edge_feat)
        inv_f, equiv_f = torch.split(edge_scaler, transformer.d_hidden, dim=-1)
        H = transformer.node_mlp(torch.cat([
            H, scatter_sum(H[uc] * inv_f, ur, dim_size=H.shape[0], dim=0)
        ], dim=-1))
        V = scatter_mean(edge_vec.unsqueeze(-1) * equiv_f.unsqueeze(-2), ur, dim_size=H.shape[0], dim=0)

        btn = batch_id[block_id]
        if geo_cache.should_recompute_batch():
            H_batch, H_mask = graph_to_batch_nx(H, btn, mask_is_pad=False, factor_req=8)
            bs, mn = H_batch.shape[0], H_batch.shape[1]
            lbuf = torch.zeros(bs, dtype=batch_id.dtype, device=batch_id.device)
            lbuf[1:] = torch.cumsum(scatter_sum(torch.ones_like(btn), btn), dim=-1)[:-1]
            lbuf = lbuf[btn]
            si = (torch.cumsum(torch.ones_like(btn), dim=-1) - 1) - lbuf
            geo_cache.store_batch((H_mask, bs, mn, btn, si))
        else:
            H_mask, bs, mn, btn, si = geo_cache.get_batch()

        H_batch, _ = graph_to_batch_nx(H, btn, mask_is_pad=False, factor_req=8)
        V_batch = torch.zeros((bs, mn, *V.shape[1:]), dtype=V.dtype, device=V.device)
        V_batch[H_mask] = V
        Z_batch = torch.zeros((bs, mn, *Z_flat.shape[1:]), dtype=Z_flat.dtype, device=Z_flat.device)
        Z_batch[H_mask] = Z_flat

        if geo_cache.should_recompute_bias():
            df = torch.cat([dis_feat, eai], -1) if transformer.use_edge_feat else dis_feat
            rbf = transformer.rbf_mapping(df)
            if transformer._local_mask:
                rfb = torch.ones((bs, mn, mn, rbf.shape[-1]), dtype=rbf.dtype, device=rbf.device) * float("-inf")
                rfb[~H_mask] = 0.0
            else:
                rfb = torch.zeros((bs, mn, mn, rbf.shape[-1]), dtype=rbf.dtype, device=rbf.device)
            if attn_mask is not None:
                rfb[~attn_mask] = float("-inf")
            rfb[btn[ur], si[ur], si[uc]] = rbf
            rfb = rfb.reshape(bs, mn, mn, transformer.n_layers, -1).permute(3, 0, 4, 1, 2)
            D = -(torch.norm(Z_batch.unsqueeze(1) - Z_batch.unsqueeze(2), dim=-1))
            geo_cache.store_bias((D.detach(), rfb))
        else:
            D, rfb = geo_cache.get_bias()

        cached_info = (D, rfb, H_mask)

        for i in range(transformer.n_layers):
            if layer_cache.should_reuse(i):
                H_batch, V_batch = layer_cache.get_cached(i)
            else:
                H_batch, V_batch = transformer._modules[f"layer_{i}"](H_batch, V_batch, cached_info)
                layer_cache.store(i, H_batch, V_batch)

        if transformer.layer_norm == "pre":
            H_batch = transformer.ln(H_batch)
        Hg = H_batch[H_mask]
        Vg = V_batch[H_mask]
        Vg = Vg / (Vg.norm(dim=-2, keepdim=True) + 1e-5)
        _, Vg = transformer.final_v(Hg, Vg)
        return Hg, Vg

    return cached_forward


def make_cached_dpm_sample(dpm, eps_cache, layer_cache, geo_cache):
    @torch.no_grad()
    def cached_sample(H, X, cond_embedding, chain_ids, generate_mask, lengths,
                      pbar=False, **kwargs):
        from utils.gnn_utils import length_to_batch_id
        from tqdm.auto import tqdm

        batch_ids = length_to_batch_id(lengths)
        gm = generate_mask[:, None]
        X_init = torch.where(gm.expand_as(X), torch.randn_like(X), X)
        H_init = torch.where(gm.expand_as(H), torch.randn_like(H), H)
        traj = {dpm.num_steps: (X_init, H_init)}

        pbar_fn = functools.partial(tqdm, total=dpm.num_steps, desc="Sampling (cached)") if pbar else (lambda x: x)
        edges, edge_types = dpm._get_edges(chain_ids, batch_ids, lengths)

        eps_cache.reset()
        layer_cache.reset()
        geo_cache.reset()

        for t in pbar_fn(range(dpm.num_steps, 0, -1)):
            X_t, H_t = traj[t]
            X_t = torch.round(X_t, decimals=4)
            H_t = torch.round(H_t, decimals=4)
            beta = dpm.trans_x.get_timestamp(t).view(1).repeat(X_t.shape[0])
            t_tensor = torch.full([X_t.shape[0]], fill_value=t, dtype=torch.long, device=X_t.device)

            eps_cache.set_step(t)
            layer_cache.set_step(t)
            geo_cache.set_step(t)

            if eps_cache.should_reuse():
                eps_H, eps_X = eps_cache.get_cached()
            else:
                eps_H, eps_X = dpm.eps_net(
                    H_t, X_t, cond_embedding, edges, edge_types, generate_mask, batch_ids, beta
                )
                eps_cache.store(eps_H, eps_X)

            H_next = dpm.trans_h.denoise(H_t, eps_H, generate_mask, batch_ids, t_tensor)
            X_next = dpm.trans_x.denoise(X_t, eps_X, generate_mask, batch_ids, t_tensor)
            traj[t - 1] = (X_next, H_next)
            traj[t] = (traj[t][0].cpu(), traj[t][1].cpu())

        return traj

    return cached_sample


def make_cached_vae_generate(autoencoder, vae_layer_cache, vae_geo_cache):
    original_generate = autoencoder.generate
    try:
        dec_t = _get_transformer(autoencoder.decoder)
    except RuntimeError:
        return original_generate

    cached_fwd = make_cached_transformer_forward(dec_t, vae_layer_cache, vae_geo_cache)
    orig_fwd = dec_t.forward

    @torch.no_grad()
    def cached_generate(*args, **kwargs):
        n_iter = kwargs.get("n_iter", 10)
        vae_layer_cache.total_steps = n_iter
        vae_layer_cache.reset()
        vae_geo_cache.reset()

        orig_ds = autoencoder.decode_structure
        cnt = [0]

        def tracked_ds(*a, **kw):
            dec_t.forward = cached_fwd
            step = n_iter - cnt[0]
            vae_layer_cache.set_step(step)
            vae_geo_cache.set_step(step)
            cnt[0] += 1
            try:
                return orig_ds(*a, **kw)
            finally:
                dec_t.forward = orig_fwd

        autoencoder.decode_structure = tracked_ds
        try:
            return original_generate(*args, **kwargs)
        finally:
            autoencoder.decode_structure = orig_ds

    return cached_generate


def patch_unimomo(
    model,
    eps_stride=2, eps_warmup_frac=0.12, eps_cooldown_frac=0.08,
    geo_stride=3, displacement_threshold=0.5,
    shallow_stride=2, deep_stride=3, warmup_frac=0.10, cooldown_frac=0.05,
    vae_shallow_stride=2, vae_deep_stride=2, vae_warmup_frac=0.20, vae_cooldown_frac=0.20,
    vae_geo_stride=2, vae_displacement_threshold=0.3,
    warmup="frac",
):
    caches = {}
    is_ldm = hasattr(model, "diffusion") and hasattr(model, "autoencoder")

    if is_ldm:
        dpm = model.diffusion
        num_steps = dpm.num_steps
        try:
            diff_t = _get_transformer(dpm.eps_net.encoder)
        except RuntimeError:
            diff_t = None

        if diff_t is not None and diff_t.n_layers > 0:
            tstar = compute_tstar(dpm.trans_x) if warmup == "tstar" else None
            eps_cache = EpsNetCache(num_steps, eps_stride, eps_warmup_frac, eps_cooldown_frac,
                                    "diff_eps", tstar)
            layer_cache = LayerCache(diff_t.n_layers, num_steps, shallow_stride, deep_stride,
                                     warmup_frac, cooldown_frac, "diff_layers", tstar)
            geo_cache = GeoCache(geo_stride, displacement_threshold, "diff_geo")

            diff_t.forward = make_cached_transformer_forward(diff_t, layer_cache, geo_cache)
            dpm.sample = make_cached_dpm_sample(dpm, eps_cache, layer_cache, geo_cache)
            caches.update(eps=eps_cache, diff_layers=layer_cache, diff_geo=geo_cache)

            print(f"[ManifoldCache] Diffusion: {diff_t.n_layers} layers, {num_steps} steps")
            if tstar is not None:
                print(f"  Warmup: T*={tstar} (L1/L3 caching only for t <= {tstar})")
            print(f"  L1 EpsNet: stride={eps_stride}, warmup={eps_warmup_frac:.0%}, cooldown={eps_cooldown_frac:.0%}")
            print(f"  L2 Geo: stride={geo_stride}, disp_th={displacement_threshold}A")
            print(f"  L3 Layers: shallow={shallow_stride}, deep={deep_stride}")

    ae = model.autoencoder if is_ldm else model
    if hasattr(ae, "decoder"):
        try:
            dec_t = _get_transformer(ae.decoder)
        except RuntimeError:
            dec_t = None
        if dec_t is not None and dec_t.n_layers > 0:
            vl = LayerCache(dec_t.n_layers, 10, vae_shallow_stride, vae_deep_stride,
                            vae_warmup_frac, vae_cooldown_frac, "vae_layers")
            vg = GeoCache(vae_geo_stride, vae_displacement_threshold, "vae_geo")
            ae.generate = make_cached_vae_generate(ae, vl, vg)
            caches.update(vae_layers=vl, vae_geo=vg)
            print(f"[ManifoldCache] VAE: {dec_t.n_layers} layers, L2+L3 only")

    return caches
