"""
Usage:
  python sample_brats_Manifoldcache.py -w model/model_brats.pt -i dataset/brats2021/seg --shallow_stride 2 --deep_stride 4
  python sample_brats_Manifoldcache.py -w model/model_brats.pt -i dataset/brats2021/seg --warmup tstar
  python sample_brats_Manifoldcache.py -w model/model_brats.pt -i dataset/brats2021/seg --compare
"""

import os, glob, time, math, argparse, csv
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import torchio as tio
from tqdm import tqdm
from torchvision.transforms import Compose, Lambda

from diffusion_model.trainer_brats import GaussianDiffusion, num_to_groups
from diffusion_model.unet_brats import create_model
from diffusion_model.modules import timestep_embedding
from utils.dtypes_brats import LabelEnum

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


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


_CONV_OPS = ("convolution", "_convolution", "conv3d", "conv2d", "conv1d",
             "convolution_overrideable", "cudnn_convolution",
             "slow_conv3d", "slow_conv_transpose3d", "conv_transpose3d")

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
            if name in _CONV_OPS:
                w = args[1]
                out_t = out[0] if isinstance(out, (tuple, list)) else out
                return 2 * _prod(out_t.shape) * int(w.shape[1]) * _prod(w.shape[2:])
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


def compute_tstar(alphas_cumprod):
    idx = torch.nonzero(alphas_cumprod >= 0.5).flatten()
    return int(idx.max()) if idx.numel() > 0 else -1


class StrideCachedUNet(nn.Module):
    def __init__(self, unet, total_timesteps=250, shallow_stride=2, deep_stride=4,
                 warmup_frac=0.0, cooldown_frac=0.0, mid_boost=1.5, t_star=None):
        super().__init__()
        self.unet = unet
        self.T = total_timesteps
        self.shallow_stride = shallow_stride
        self.deep_stride = deep_stride
        self.warmup_frac = warmup_frac
        self.cooldown_frac = cooldown_frac
        self.mid_boost = mid_boost
        self.t_star = t_star

        n_enc = len(unet.input_blocks)
        n_dec = len(unet.output_blocks)
        self.depth = {f"enc_{i}": i / max(n_enc - 1, 1) for i in range(n_enc)}
        self.depth.update({f"dec_{i}": 1.0 - i / max(n_dec - 1, 1) for i in range(n_dec)})
        self.depth["mid"] = 1.0

        self.cache = {}
        self.last_computed = {}
        self.total_blocks = 0
        self.cached_blocks = 0

        self.F_block = {}
        self.F_full = None
        self.compute_count = defaultdict(int)
        self.skip_count = defaultdict(int)
        self.num_forwards = 0
        self._need_measure = True

    def get_stride(self, depth, step_idx):
        progress = 1.0 - step_idx / self.T

        if self.t_star is not None:
            if step_idx > self.t_star:
                return 1
        elif progress < self.warmup_frac:
            return 1
        if progress > 1.0 - self.cooldown_frac:
            return 1

        base = self.shallow_stride + (self.deep_stride - self.shallow_stride) * depth
        boost = 1.0 + (self.mid_boost - 1.0) * math.exp(-0.5 * ((progress - 0.5) / 0.2) ** 2)
        return max(1, int(base * boost))

    def _should_compute(self, bid, step_idx):
        if bid not in self.cache:
            return True
        return self.last_computed[bid] - step_idx >= self.get_stride(self.depth[bid], step_idx)

    def _run_block(self, bid, fn, step_idx, counter):
        self.total_blocks += 1
        if not self._should_compute(bid, step_idx):
            self.cached_blocks += 1
            self.skip_count[bid] += 1
            return self.cache[bid]
        f0 = counter.flops if counter else 0
        h = fn()
        if counter and bid not in self.F_block:
            self.F_block[bid] = counter.flops - f0
        self.cache[bid] = h
        self.last_computed[bid] = step_idx
        self.compute_count[bid] += 1
        return h

    def reset(self):
        self.cache.clear()
        self.last_computed.clear()
        self.total_blocks = 0
        self.cached_blocks = 0

    def forward(self, x, timesteps, y=None, step_idx=0):
        self.num_forwards += 1
        counter = SimpleDispatchFlopCounter() if (self._need_measure and _DISPATCH_AVAILABLE) else None
        if counter:
            counter.__enter__()
        try:
            emb = self.unet.time_embed(timestep_embedding(timesteps, self.unet.model_channels))
            if self.unet.num_classes is not None and y is not None:
                emb = emb + self.unet.label_emb(y)

            h = x.type(self.unet.dtype)
            hs = []
            for i, module in enumerate(self.unet.input_blocks):
                h = self._run_block(f"enc_{i}", lambda: module(h, emb), step_idx, counter)
                hs.append(h)

            h = self._run_block("mid", lambda: self.unet.middle_block(h, emb), step_idx, counter)

            for i, module in enumerate(self.unet.output_blocks):
                cat_in = torch.cat([h, hs.pop()], dim=1)
                h = self._run_block(f"dec_{i}", lambda: module(cat_in, emb), step_idx, counter)

            out = self.unet.out(h.type(x.dtype))
            if counter:
                self.F_full = counter.flops
            return out
        finally:
            if counter:
                counter.__exit__(None, None, None)
                self._need_measure = False

    def print_stats(self):
        rate = self.cached_blocks / max(self.total_blocks, 1) * 100
        print(f"  Cache rate: {rate:.1f}% ({self.cached_blocks}/{self.total_blocks})")

    def block_eval_hit_rate(self):
        skips = sum(self.skip_count.values())
        return skips / max(sum(self.compute_count.values()) + skips, 1)

    def flops_saved(self):
        if self.F_full is None:
            return None
        return sum(n * self.F_block.get(b, 0.0) for b, n in self.skip_count.items())

    def flops_baseline(self):
        return None if self.F_full is None else self.num_forwards * self.F_full

    def flops_total(self):
        return None if self.F_full is None else self.flops_baseline() - self.flops_saved()

    def flops_summary(self):
        tf, bf = self.flops_total(), self.flops_baseline()
        d = {
            "num_network_forwards": self.num_forwards,
            "block_eval_hit_rate": self.block_eval_hit_rate(),
            "F_full": self.F_full,
            "total_flops": tf,
            "baseline_flops": bf,
        }
        if tf is not None and bf:
            d["flops_ratio_vs_baseline"] = tf / bf
            d["flops_saved_pct"] = 100.0 * (1.0 - tf / bf)
        for b in sorted(self.F_block):
            d[f"Fblock_{b}"] = self.F_block[b]
            d[f"skip_{b}"] = self.skip_count.get(b, 0)
            d[f"compute_{b}"] = self.compute_count.get(b, 0)
        return d

    def print_flops_report(self):
        if self.F_full is None:
            print("\n  FLOPs: not measured (dispatch counter unavailable; needs torch>=2.0).")
            return
        tf, bf = self.flops_total(), self.flops_baseline()
        print(f"\n  ===== FLOPs / NFE report =====")
        print(f"    network forwards (NFE): {self.num_forwards}")
        print(f"    block-eval cache rate:  {100 * self.block_eval_hit_rate():.1f}%")
        print(f"    total FLOPs (cached):   {tf:.4e}")
        print(f"    baseline FLOPs:         {bf:.4e}")
        print(f"    --> FLOPs ratio: {tf / bf:.3f}x  (saved {100 * (1 - tf / bf):.1f}%)")
        saved = {b: n * self.F_block.get(b, 0.0) for b, n in self.skip_count.items()}
        top = [(b, v) for b, v in sorted(saved.items(), key=lambda kv: kv[1], reverse=True)[:5] if v > 0]
        if top:
            print(f"    top FLOPs-saving blocks:")
            for b, v in top:
                print(f"      {b}: F_block={self.F_block.get(b, 0):.3e}, "
                      f"skipped {self.skip_count[b]}x -> saved {v:.3e}")


class CachedDiffusionSampler:
    def __init__(self, diffusion, cached_unet):
        self.diff = diffusion
        self.cached_unet = cached_unet

    @torch.no_grad()
    def sample(self, batch_size, condition_tensors):
        device = self.diff.betas.device
        shape = (batch_size, self.diff.channels, self.diff.depth_size,
                 self.diff.image_size, self.diff.image_size)
        img = torch.randn(shape, device=device)
        self.cached_unet.reset()

        for i in tqdm(reversed(range(self.diff.num_timesteps)),
                      desc='sampling (cached)', total=self.diff.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            net_in = torch.cat([img, condition_tensors], 1) if self.diff.with_condition else img
            noise_pred = self.cached_unet(net_in, t, step_idx=i)

            x_recon = self.diff.predict_start_from_noise(img, t=t, noise=noise_pred)
            x_recon.clamp_(-1., 1.)
            model_mean, _, model_log_var = self.diff.q_posterior(x_start=x_recon, x_t=img, t=t)
            noise = torch.randn_like(img)
            nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (img.dim() - 1)))
            img = model_mean + nonzero_mask * (0.5 * model_log_var).exp() * noise

        return img


def label2masks(masked_img):
    result_img = np.zeros(masked_img.shape + (4,))
    result_img[masked_img == LabelEnum.TUMORAREA1.value, 0] = 1
    result_img[masked_img == LabelEnum.TUMORAREA2.value, 1] = 1
    result_img[masked_img == LabelEnum.TUMORAREA3.value, 2] = 1
    result_img[masked_img == LabelEnum.BRAINAREA.value, 3] = 1
    return result_img


def resize_img_4d(input_img, input_size, depth_size):
    h, w, d, c = input_img.shape
    if h == input_size and w == input_size and d == depth_size:
        return input_img
    result_img = np.zeros((input_size, input_size, depth_size, 4))
    for ch in range(c):
        img = tio.ScalarImage(tensor=input_img[..., ch].copy()[np.newaxis, ...])
        result_img[..., ch] += np.asarray(tio.Resize((input_size, input_size, depth_size))(img))[0]
    return result_img


def processImg(img_path):
    subject = tio.Subject(image=tio.ScalarImage(img_path))
    transform = tio.Compose([tio.RescaleIntensity((0, 1)), tio.CropOrPad((240, 240, 155))])
    transform(subject).image.save(img_path)


def processMsk(msk_path):
    subject = tio.Subject(mask=tio.LabelMap(msk_path))
    tio.CropOrPad((240, 240, 155))(subject).mask.save(msk_path)


def main():
    parser = argparse.ArgumentParser(description="Med-DDPM with stride caching")
    parser.add_argument('-i', '--inputfolder', type=str, default="dataset/brats2021/seg")
    parser.add_argument('-e', '--exportfolder', type=str, default="exports_manifoldcache/")
    parser.add_argument('--input_size', type=int, default=192)
    parser.add_argument('--depth_size', type=int, default=144)
    parser.add_argument('--num_channels', type=int, default=64)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--timesteps', type=int, default=250)
    parser.add_argument('-w', '--weightfile', type=str, default="model/model_brats.pt")
    parser.add_argument('--shallow_stride', type=int, default=2, help="Stride for shallow (192^3) blocks")
    parser.add_argument('--deep_stride', type=int, default=4, help="Stride for deep (12^3) blocks")
    parser.add_argument('--warmup', type=str.lower, default="frac", choices=["frac", "tstar", "t*"],
                        help="'frac': no caching for the first --warmup_frac of steps (default). "
                             "'tstar': no caching for t > T* = max{t : alpha_bar_t >= 1/2}.")
    parser.add_argument('--warmup_frac', type=float, default=0.0)
    parser.add_argument('--cooldown_frac', type=float, default=0.0)
    parser.add_argument('--mid_boost', type=float, default=1.5, help="Stride multiplier at mid-denoising")
    parser.add_argument('--compare', action='store_true', help="Also run uncached baseline and report speedup")
    args = parser.parse_args()
    use_tstar = args.warmup in ("tstar", "t*")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels, out_channels = 8, 4

    for sub in ["t1", "t1ce", "t2", "flair", "seg"]:
        os.makedirs(f"{args.exportfolder}/{sub}", exist_ok=True)

    print("Loading model...")
    unet = create_model(args.input_size, args.num_channels, args.num_res_blocks,
                        in_channels=in_channels, out_channels=out_channels).to(device)
    diffusion = GaussianDiffusion(
        unet, image_size=args.input_size, depth_size=args.depth_size,
        timesteps=args.timesteps, loss_type='l1', with_condition=True, channels=out_channels
    ).to(device)
    weight = torch.load(args.weightfile, map_location="cpu")
    diffusion.load_state_dict(weight['ema'])
    diffusion.eval()

    t_star = compute_tstar(diffusion.alphas_cumprod) if use_tstar else None

    cached_unet = StrideCachedUNet(
        unet, total_timesteps=args.timesteps,
        shallow_stride=args.shallow_stride, deep_stride=args.deep_stride,
        warmup_frac=args.warmup_frac, cooldown_frac=args.cooldown_frac,
        mid_boost=args.mid_boost, t_star=t_star,
    ).to(device)
    cached_unet.eval()
    sampler = CachedDiffusionSampler(diffusion, cached_unet)

    warmup_desc = (f"T*={t_star} (caching for t<=T*, {t_star + 1}/{args.timesteps} steps)"
                   if use_tstar else f"warmup_frac={args.warmup_frac}")
    print(f"Caching config: shallow_stride={args.shallow_stride}, deep_stride={args.deep_stride}, "
          f"{warmup_desc}, cooldown={args.cooldown_frac}, mid_boost={args.mid_boost}")

    input_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: t.permute(3, 0, 1, 2)),
        Lambda(lambda t: t.unsqueeze(0)),
        Lambda(lambda t: t.transpose(4, 2))
    ])

    mask_list = sorted(glob.glob(f"{args.inputfolder}/*.nii.gz"))
    if args.compare:
        mask_list = mask_list[:2]
    print(f"Found {len(mask_list)} masks")

    counter = 0
    total_baseline_time = 0
    total_cached_time = 0

    for k, inputfile in enumerate(mask_list):
        print(f"\n[{k + 1}/{len(mask_list)}] {os.path.basename(inputfile)}")
        img = resize_img_4d(label2masks(nib.load(inputfile).get_fdata()), args.input_size, args.depth_size)
        input_tensor = input_transform(img)
        ref = nib.load(inputfile)
        name = os.path.basename(inputfile).split("_seg.nii.gz")[0]

        for bsize in num_to_groups(args.num_samples, args.batchsize):
            condition_tensors = torch.cat([input_tensor] * bsize, 0).to(device)

            if args.compare:
                torch.manual_seed(42)
                torch.cuda.manual_seed(42)
                t0 = time.time()
                baseline_images = diffusion.sample(batch_size=bsize, condition_tensors=condition_tensors)
                total_baseline_time += time.time() - t0

            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            t0 = time.time()
            cached_images = sampler.sample(bsize, condition_tensors)
            total_cached_time += time.time() - t0

            if args.compare:
                rel_diff = (cached_images - baseline_images).norm() / (baseline_images.norm() + 1e-8)
                print(f"  Relative L2 diff: {rel_diff.item():.6f}")
            cached_unet.print_stats()

            for b in range(bsize):
                counter += 1
                for ch, mod in enumerate(["t1", "t1ce", "t2", "flair"]):
                    sample = cached_images[b, ch, ...].transpose(2, 0).cpu().numpy()
                    sample = sample.reshape([args.input_size, args.input_size, args.depth_size])
                    save_path = f"{args.exportfolder}/{mod}/{counter}_{name}_{mod}.nii.gz"
                    nib.save(nib.Nifti1Image(sample, affine=ref.affine), save_path)
                    processImg(save_path)

                mask = ref.get_fdata()
                mask[mask == 4.] = 0.; mask[mask == 2.] = 5.
                mask[mask == 1.] = 2.; mask[mask == 5.] = 1.
                seg_path = f"{args.exportfolder}/seg/{counter}_{name}_seg.nii.gz"
                nib.save(nib.Nifti1Image(mask, affine=ref.affine), seg_path)
                processMsk(seg_path)

            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Sampling complete")
    print(f"  Samples: {counter}")
    print(f"  Cached time: {total_cached_time:.1f}s")
    if args.compare and total_baseline_time > 0:
        print(f"  Baseline time: {total_baseline_time:.1f}s")
        print(f"  Speedup: {total_baseline_time / total_cached_time:.2f}x")
    print(f"{'=' * 60}")

    cached_unet.print_flops_report()
    summ = {
        "label": "manifoldcache_med_ddpm",
        "num_samples_saved": counter,
        "timesteps": args.timesteps,
        "shallow_stride": args.shallow_stride,
        "deep_stride": args.deep_stride,
        "warmup": "tstar" if use_tstar else "frac",
        "t_star": t_star,
        "warmup_frac": args.warmup_frac,
        "cooldown_frac": args.cooldown_frac,
        "mid_boost": args.mid_boost,
        "cached_time_s": total_cached_time,
    }
    if args.compare and total_baseline_time > 0:
        summ["baseline_time_s"] = total_baseline_time
        summ["wallclock_speedup"] = total_baseline_time / total_cached_time
    summ.update(cached_unet.flops_summary())

    csv_path = os.path.join(args.exportfolder, "nfe_flops_summary.csv")
    try:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(summ.keys()))
            w.writerow(list(summ.values()))
        print(f"  Wrote NFE/FLOPs summary to {csv_path}")
    except Exception as e:
        print(f"  Could not write FLOPs CSV: {e}")


if __name__ == "__main__":
    main()
