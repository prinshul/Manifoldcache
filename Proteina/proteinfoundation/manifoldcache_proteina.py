import math
from collections import defaultdict
from typing import Dict

import torch


def _prod(shape):
    p = 1
    for s in shape:
        p *= int(s)
    return p


def _op_name(func):
    for attr in ("overloadpacket", "_overloadpacket"):
        pkt = getattr(func, attr, None)
        if pkt is not None and hasattr(pkt, "__name__"):
            return pkt.__name__
    parts = str(func).split(".")
    return parts[1] if len(parts) >= 2 else str(func)


try:
    from torch.utils._python_dispatch import TorchDispatchMode

    class SimpleDispatchFlopCounter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.flops = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            try:
                self.flops += self._count(_op_name(func), args, out)
            except Exception:
                pass
            return out

        @staticmethod
        def _count(name, args, out):
            if name == "mm":
                a, b = args[0], args[1]
                return 2 * a.shape[-2] * a.shape[-1] * b.shape[-1]
            if name == "addmm":
                a, b = args[1], args[2]
                return 2 * a.shape[-2] * a.shape[-1] * b.shape[-1]
            if name in ("bmm", "baddbmm"):
                a, b = (args[0], args[1]) if name == "bmm" else (args[1], args[2])
                return 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[-1]
            if name == "linear":
                inp, w = args[0], args[1]
                return 2 * _prod(inp.shape[:-1]) * w.shape[1] * w.shape[0]
            if name == "matmul":
                return 2 * _prod(out.shape) * args[0].shape[-1]
            if "scaled_dot_product" in name:
                *bd, s, d = args[0].shape
                return 4 * _prod(bd) * s * s * d
            return 0

    _DISPATCH_AVAILABLE = True
except Exception:
    SimpleDispatchFlopCounter, _DISPATCH_AVAILABLE = None, False


class ForwardFlopMeter:
    def __init__(self):
        self.calls = defaultdict(int)
        self.flops = defaultdict(float)
        self.per_sample = {}
        self._hooks = []
        self._active = None

    def register(self, module, name):
        def pre_hook(mod, args):
            if not (args and isinstance(args[0], dict) and "x_t" in args[0]):
                return
            b, n = args[0]["x_t"].shape[:2]
            key = (name, int(n))
            self.calls[name] += 1
            if key in self.per_sample or not _DISPATCH_AVAILABLE:
                self.flops[name] += int(b) * self.per_sample.get(key, 0.0)
                return
            counter = SimpleDispatchFlopCounter()
            counter.__enter__()
            self._active = (counter, key, int(b))

        def post_hook(mod, args, out):
            if self._active is None:
                return
            counter, key, b = self._active
            counter.__exit__(None, None, None)
            self._active = None
            self.per_sample[key] = counter.flops / max(b, 1)
            self.flops[key[0]] += counter.flops

        self._hooks.append(module.register_forward_pre_hook(pre_hook))
        self._hooks.append(module.register_forward_hook(post_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def total_flops(self):
        return sum(self.flops.values())

    def summary_dict(self):
        d = {"label": "baseline",
             "num_network_forwards": sum(self.calls.values()),
             "total_flops": self.total_flops()}
        for name in sorted(self.calls):
            d[f"{name}_num_forwards"] = self.calls[name]
            d[f"{name}_total_flops"] = self.flops[name]
        for (name, n), f in sorted(self.per_sample.items()):
            d[f"{name}_flops_per_sample_n{n}"] = f
        return d


def compute_tstar_step(fm, nsteps, schedule_mode, schedule_p):
    ts = fm.get_schedule(mode=schedule_mode, nsteps=nsteps, p1=schedule_p)[:-1]
    log_snr, _ = fm.log_snr(ts)
    safe = torch.nonzero(log_snr >= 0).flatten()
    return int(safe.min()) if safe.numel() > 0 else nsteps


class BlockStrideCache:
    def __init__(self, nlayers, total_steps=400, shallow_stride=2, deep_stride=4,
                 warmup_frac=0.10, cooldown_frac=0.05, tstar_step=None):
        self.nlayers = nlayers
        self.total_steps = total_steps
        self.shallow_stride = shallow_stride
        self.deep_stride = deep_stride
        self.warmup_frac = warmup_frac
        self.cooldown_frac = cooldown_frac
        self.tstar_step = tstar_step

        self.cache = {}
        self.last_computed = {}
        self.current_step = 0
        self.current_tag = "cond"

        self.total_evals = 0
        self.cached_evals = 0
        self.F_full = None
        self.F_layer = None
        self._need_measure = True

    def set_step(self, step_idx, tag="cond"):
        self.current_step = step_idx
        self.current_tag = tag

    def reset(self):
        self.cache.clear()
        self.last_computed.clear()

    def get_stride(self, layer_idx):
        progress = self.current_step / max(self.total_steps, 1)
        if self.tstar_step is not None:
            if self.current_step < self.tstar_step:
                return 1
        elif progress < self.warmup_frac:
            return 1
        if progress > 1.0 - self.cooldown_frac:
            return 1
        depth = layer_idx / max(self.nlayers - 1, 1)
        base = self.shallow_stride + (self.deep_stride - self.shallow_stride) * depth
        if self.current_tag != "cond":
            base *= 1.5
        return max(1, int(base))

    def should_reuse(self, layer_idx):
        key = (layer_idx, self.current_tag)
        if key not in self.last_computed:
            return False
        return self.current_step - self.last_computed[key] < self.get_stride(layer_idx)

    def get_cached(self, layer_idx):
        self.total_evals += 1
        self.cached_evals += 1
        return self.cache[(layer_idx, self.current_tag)]

    def store(self, layer_idx, tensor):
        key = (layer_idx, self.current_tag)
        self.total_evals += 1
        self.cache[key] = tensor
        self.last_computed[key] = self.current_step

    def hit_rate(self):
        return self.cached_evals / max(self.total_evals, 1)

    def print_stats(self):
        print(f"  ManifoldCache: {100 * self.hit_rate():.1f}% layers cached "
              f"({self.cached_evals}/{self.total_evals})")

    def num_forwards(self):
        return self.total_evals / max(self.nlayers, 1)

    def total_flops(self):
        if self.F_full is None or self.F_layer is None:
            return None
        return self.num_forwards() * self.F_full - self.cached_evals * self.F_layer

    def baseline_flops(self):
        return None if self.F_full is None else self.num_forwards() * self.F_full

    def flops_summary(self, name="net"):
        return {
            f"{name}_nlayers": self.nlayers,
            f"{name}_num_forwards": self.num_forwards(),
            f"{name}_total_layer_evals": self.total_evals,
            f"{name}_cached_layer_evals": self.cached_evals,
            f"{name}_cache_hit_rate": self.hit_rate(),
            f"{name}_F_full": self.F_full,
            f"{name}_F_layer": self.F_layer,
            f"{name}_total_flops": self.total_flops(),
            f"{name}_baseline_flops": self.baseline_flops(),
        }


def make_cached_nn_forward(model_nn, cache: BlockStrideCache):
    def cached_forward(batch_nn: Dict[str, torch.Tensor]):
        mask = batch_nn["mask"]
        counter = SimpleDispatchFlopCounter() if (cache._need_measure and _DISPATCH_AVAILABLE) else None
        if counter:
            counter.__enter__()
        try:
            c = model_nn.cond_factory(batch_nn)
            c = model_nn.transition_c_2(model_nn.transition_c_1(c, mask), mask)

            coors_3d = batch_nn["x_t"] * mask[..., None]
            coors_embed = model_nn.linear_3d_embed(coors_3d) * mask[..., None]
            seq_f_repr = model_nn.init_repr_factory(batch_nn)
            seqs = (coors_embed + seq_f_repr) * mask[..., None]

            pair_rep = None
            if model_nn.use_attn_pair_bias:
                pair_rep = model_nn.pair_repr_builder(batch_nn)

            seqs, pair_rep, mask_ext, c = model_nn._extend_w_registers(seqs, pair_rep, mask, c)
            flops_pre_layer = counter.flops if counter else 0

            for i in range(model_nn.nlayers):
                if cache.should_reuse(i):
                    seqs = cache.get_cached(i)
                else:
                    seqs = model_nn.transformer_layers[i](seqs, pair_rep, c, mask_ext)
                    cache.store(i, seqs)
                if counter and i == 0:
                    cache.F_layer = counter.flops - flops_pre_layer

                if model_nn.update_pair_repr and i < model_nn.nlayers - 1:
                    if model_nn.pair_update_layers[i] is not None:
                        pair_rep = model_nn.pair_update_layers[i](seqs, pair_rep, mask_ext)

            seqs, pair_rep, mask_ext = model_nn._undo_registers(seqs, pair_rep, mask_ext)

            final_coors = model_nn.coors_3d_decoder(seqs) * batch_nn["mask"][..., None]
            nn_out = {}
            if model_nn.update_pair_repr and model_nn.num_buckets_predict_pair is not None:
                pair_pred = model_nn.pair_head_prediction(pair_rep)
                final_coors = final_coors + torch.mean(pair_pred) * 0.0
                final_coors = final_coors * batch_nn["mask"][..., None]
                nn_out["pair_pred"] = pair_pred
            nn_out["coors_pred"] = final_coors

            if counter:
                cache.F_full = counter.flops
            return nn_out
        finally:
            if counter:
                counter.__exit__(None, None, None)
                cache._need_measure = False

    return cached_forward


def make_tracked_predict(model, cache_main, cache_ag):
    step_counter = [0]
    total_steps = cache_main.total_steps

    def tracked_predict(batch, guidance_weight=1.0, autoguidance_ratio=0.0):
        step = step_counter[0]
        if step >= total_steps:
            step = 0
            cache_main.reset()
            if cache_ag is not None:
                cache_ag.reset()
        step_counter[0] = step + 1

        if model.motif_conditioning and (
            "fixed_structure_mask" not in batch or "x_motif" not in batch
        ):
            batch.update(model.motif_factory(batch, zeroes=True))

        cache_main.set_step(step, "cond")
        nn_out = model.nn(batch)
        x_pred = model._nn_out_to_x_clean(nn_out, batch)

        if guidance_weight != 1.0:
            assert 0.0 <= autoguidance_ratio <= 1.0
            if autoguidance_ratio > 0.0:
                if cache_ag is not None:
                    cache_ag.set_step(step, "autoguidance")
                nn_out_ag = model.nn_ag(batch)
                x_pred_ag = model._nn_out_to_x_clean(nn_out_ag, batch)
            else:
                x_pred_ag = torch.zeros_like(x_pred)

            if autoguidance_ratio < 1.0:
                assert "cath_code" in batch
                uncond_batch = batch.copy()
                uncond_batch.pop("cath_code")
                cache_main.set_step(step, "uncond")
                nn_out_uncond = model.nn(uncond_batch)
                x_pred_uncond = model._nn_out_to_x_clean(nn_out_uncond, uncond_batch)
            else:
                x_pred_uncond = torch.zeros_like(x_pred)

            x_pred = guidance_weight * x_pred + (1 - guidance_weight) * (
                autoguidance_ratio * x_pred_ag + (1 - autoguidance_ratio) * x_pred_uncond
            )

        v = model.fm.xt_dot(x_pred, batch["x_t"], batch["t"], batch["mask"])
        return x_pred, v

    return tracked_predict


def patch_proteina_model(model, dt=0.0025, shallow_stride=2, deep_stride=4,
                         warmup_frac=0.10, cooldown_frac=0.05, warmup="frac",
                         schedule_mode="uniform", schedule_p=1.0):
    total_steps = math.ceil(1.0 / dt)
    tstar_step = None
    if warmup == "tstar":
        tstar_step = compute_tstar_step(model.fm, total_steps, schedule_mode, schedule_p)

    print(f"[ManifoldCache] Proteina: {model.nn.nlayers} layers, {total_steps} ODE steps")
    warmup_desc = (f"T* step={tstar_step} (caching from step {tstar_step}/{total_steps})"
                   if tstar_step is not None else f"warmup_frac={warmup_frac}")
    print(f"[ManifoldCache] shallow_stride={shallow_stride}, deep_stride={deep_stride}, "
          f"{warmup_desc}, cooldown={cooldown_frac}")

    def new_cache(nlayers):
        return BlockStrideCache(nlayers, total_steps, shallow_stride, deep_stride,
                                warmup_frac, cooldown_frac, tstar_step)

    cache_main = new_cache(model.nn.nlayers)
    model.nn.forward = make_cached_nn_forward(model.nn, cache_main)

    cache_ag = None
    if model.nn_ag is not None:
        cache_ag = new_cache(model.nn_ag.nlayers)
        model.nn_ag.forward = make_cached_nn_forward(model.nn_ag, cache_ag)

    model.predict_clean_n_v_w_guidance = make_tracked_predict(model, cache_main, cache_ag)
    return cache_main, cache_ag
