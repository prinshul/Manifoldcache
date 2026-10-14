# ManifoldCache on Med-DDPM

Training-free, data-free feature caching for [Med-DDPM](https://github.com/mobaidoctor/med-ddpm) (conditional 3D brain MRI synthesis on BraTS2021).

## Setup

Follow the setup instructions in the official Med-DDPM repository: **https://github.com/mobaidoctor/med-ddpm**

1. Install the requirements (`pip install -r requirements.txt`, plus PyTorch).
2. Download the pretrained BraTS weights from the [Med-DDPM model weights link](https://drive.google.com/drive/folders/1t6jk5MrKt34JYClgfijlbNYePIcTEQvJ?usp=sharing) and place `model_brats.pt` under `model/`.
3. Put the BraTS2021 segmentation masks under `dataset/brats2021/seg/` (two example masks are included).

## Run

```bash
python sample_brats_Manifoldcache.py -w model/model_brats.pt -i dataset/brats2021/seg --shallow_stride <S_shallow> --deep_stride <S_deep> --warmup tstar
```

`<S_shallow>` and `<S_deep>` are how many steps the shallowest and deepest blocks reuse their cached output (defaults 2 and 4, with `S_shallow <= S_deep`).

**Warmup** is the first, noisiest part of sampling, where every block is recomputed and nothing is cached. `--warmup tstar` ends it automatically at T* = max{t : ᾱ_t ≥ ½}, the last step where signal outweighs noise; without it, warmup is the first `--warmup_frac` of steps. **Cooldown** is the last `--cooldown_frac` of steps (default 0), which are also recomputed in full so the final details are exact.

## Acknowledgements

Built on the official Med-DDPM implementation. If you use this code, please also cite Med-DDPM:

```
@ARTICLE{10493074,
  author={Dorjsembe, Zolnamar and Pao, Hsing-Kuo and Odonchimed, Sodtavilan and Xiao, Furen},
  journal={IEEE Journal of Biomedical and Health Informatics},
  title={Conditional Diffusion Models for Semantic 3D Brain MRI Synthesis},
  year={2024},
  volume={28},
  number={7},
  pages={4084-4093},
  doi={10.1109/JBHI.2024.3385504}}
```
