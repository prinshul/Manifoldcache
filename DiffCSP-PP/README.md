# ManifoldCache on DiffCSP++

Training-free, data-free feature caching for [DiffCSP++](https://github.com/jiaor17/DiffCSP-PP) (space group constrained crystal generation).

## Setup

Follow the setup instructions in the official DiffCSP++ repository: **https://github.com/jiaor17/DiffCSP-PP**

1. Fill in `.env` (see `.env.template`) and create the environment with `micromamba env create -f environment.yml`.
2. Download the pre-trained `mp_csp` checkpoint from the [DiffCSP++ checkpoints link](https://drive.google.com/drive/folders/1FQ_b6CE09KtyGaU_r6uO8_I5JhrQmUFB?usp=sharing) and place it under `checkpoints/mp_csp/`.

## Run

```bash
python scripts/sample_Manifoldcache.py --model_path checkpoints/mp_csp/ --save_path output_manifoldcache --json_file queries.json --shallow_stride <S_shallow> --deep_stride <S_deep> --warmup tstar
```

`<S_shallow>` and `<S_deep>` are how many denoising steps the first and last CSPNet layers reuse their cached output (defaults 2 and 4, with `S_shallow <= S_deep`). `queries.json` lists the space group, Wyckoff letters and atom types of the structures to generate.

**Warmup** is the first, noisiest part of sampling, where every layer is recomputed and nothing is cached. `--warmup tstar` ends it automatically at T* = max{t : ᾱ_t ≥ ½}, the last step where signal outweighs noise; without it, warmup is the first `--warmup_frac` of steps. **Cooldown** is the last `--cooldown_frac` of steps (default 0.05), which are also recomputed in full so the final crystal is exact.

## Acknowledgements

Built on the official DiffCSP++ implementation. If you use this code, please also cite DiffCSP++:

```
@inproceedings{
jiao2024space,
title={Space Group Constrained Crystal Generation},
author={Rui Jiao and Wenbing Huang and Yu Liu and Deli Zhao and Yang Liu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=jkvZ7v4OmP}
}
```
