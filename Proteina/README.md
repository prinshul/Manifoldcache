# ManifoldCache on Proteina

Training-free, data-free feature caching for [Proteina](https://github.com/NVIDIA-BioNeMo/proteina) (flow-based protein backbone generation).

## Setup

Follow the setup instructions in the official Proteina repository: **https://github.com/NVIDIA-BioNeMo/proteina**

1. Create the environment from `environment.yaml` and add a `.env` file with `DATA_PATH=...`.
2. Download the checkpoints `proteina_v1.4_D21M_400M_tri.ckpt` and `proteina_v1.8_D21M_400M_tri_autoguidance.ckpt` (links in the Proteina README). Set `ckpt_path` in `configs/experiment_config/inference_base.yaml` and `autoguidance_ckpt_path` in `configs/experiment_config/inference_cond_autoguidance.yaml`.
3. For designability, download the ProteinMPNN weights with `bash script_utils/download_pmpnn_weghts.sh`.

## Run

```bash
python proteinfoundation/inference_Manifoldcache.py --config_name inference_cond_autoguidance --shallow_stride <S_shallow> --deep_stride <S_deep> --warmup tstar
```

`<S_shallow>` and `<S_deep>` are how many ODE steps the first and last transformer layers reuse their cached output (defaults 2 and 4, with `S_shallow <= S_deep`).

**Warmup** is the first, noisiest part of sampling, where every layer is recomputed and nothing is cached. `--warmup tstar` ends it automatically at T*, the first step where log-SNR(t) ≥ 0 (signal outweighs noise) on the sampling schedule; without it, warmup is the first `--warmup_frac` of steps (default 0.10). **Cooldown** is the last `--cooldown_frac` of steps (default 0.05), which are also recomputed in full so the final structure is exact.

## License

The Proteina code is released under the NVIDIA License (non-commercial use only), see [LICENSE](LICENSE).

## Acknowledgements

Built on the official Proteina implementation. If you use this code, please also cite Proteina:

```
@inproceedings{geffner2025proteina,
    title={Proteina: Scaling Flow-based Protein Structure Generative Models},
    author={Tomas Geffner and Kieran Didi and Zuobai Zhang and Danny Reidenbach and Zhonglin Cao and Jason Yim and Mario Geiger and Christian Dallago and Emine Kucukbenli and Arash Vahdat and Karsten Kreis},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2025}
}
```
