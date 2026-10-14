# ManifoldCache on UniMoMo

Training-free, data-free feature caching for [UniMoMo](https://github.com/kxz18/UniMoMo) (unified 3D molecule generation for binder design).

## Setup

Follow the setup instructions in the official UniMoMo repository: **https://github.com/kxz18/UniMoMo**

1. Create the conda environment (e.g. `conda env create -f env_cuda117.yaml`) and activate it.
2. Download the trained checkpoint from the [UniMoMo release page](https://github.com/kxz18/UniMoMo/releases/tag/v1.0) and place it at `checkpoints/model.ckpt`.
3. Download and process the LNR peptide test set into `datasets/peptide/LNR/processed` (see "Datasets" in the UniMoMo README).

## Run

```bash
python generate_Manifoldcache.py --config configs/test/test_pep.yaml --ckpt checkpoints/model.ckpt --gpu 0 --save_dir ./results/fast --eps_stride <S_eps> --geo_stride <S_geo> --deep_stride <S_deep> --warmup tstar
```

`<S_eps>` is how many steps the whole denoising network output is reused, `<S_geo>` how many steps the edge and attention geometry is reused, and `<S_deep>` how many steps the deepest transformer layer is reused (defaults 2, 3 and 3).

**Warmup** is the first, noisiest part of sampling, where the network and its layers are recomputed and not reused. `--warmup tstar` ends it automatically at T* = max{t : ᾱ_t ≥ ½}, the last step where signal outweighs noise; without it, warmup is the first `--eps_warmup` / `--warmup_frac` of steps. **Cooldown** is the last `--eps_cooldown` / `--cooldown_frac` of steps (defaults 0.08 / 0.05), which are also recomputed in full so the final molecule is exact.

## Acknowledgements

Built on the official UniMoMo implementation. If you use this code, please also cite UniMoMo:

```bibtex
@inproceedings{
    kong2025unimomo,
    title={UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design},
    author={Kong, Xiangzhe and Zhang, Zishen and Zhang, Ziting and Jiao, Rui and Ma, Jianzhu and Liu, Kai and Huang, Wenbing and Liu, Yang},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025}
}
```
