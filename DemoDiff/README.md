# ManifoldCache on DemoDiff

Training-free, data-free feature caching for [DemoDiff](https://github.com/liugangcode/DemoDiff) (graph diffusion transformers for in-context molecular design).

## Setup

Follow the setup instructions in the official DemoDiff repository: **https://github.com/liugangcode/DemoDiff** (also in [`downstream/README.md`](downstream/README.md)). All commands run from `downstream/`.

1. Create the conda environment and install the dependencies with `bash seteup_env.sh`.
2. Download the pretrained model with `python prepare_model.py` (saves `pretrained/model.pt` and `pretrained/config.yaml`).
3. Download the benchmark tasks and oracles with `python prepare_data_and_oracle.py`.

## Run

```bash
cd downstream
python inference_Manifoldcache.py --task_list Albuterol_Similarity --num_samples 32 --shallow_stride <S_shallow> --deep_stride <S_deep> --warmup tstar
```

`<S_shallow>` and `<S_deep>` are how many diffusion steps the first and last transformer blocks reuse their cached output (defaults 2 and 3); the unconditional pass of classifier-free guidance reuses them twice as long. Results are saved to `log/exports_<warmup>_<S_shallow>_<S_deep>/<task>.csv` (e.g. `log/exports_tstar_2_3/Albuterol_Similarity.csv`); pass `--output_dir` to choose another folder, or `--no_cache` to run the uncached baseline.

**Warmup** is the first, noisiest part of sampling, where every block is recomputed and nothing is cached. `--warmup tstar` ends it automatically at T* = max{t : ᾱ_t ≥ ½}, the last step where signal outweighs noise; without it, warmup is the first `--warmup_frac` of steps (default 0.10). **Cooldown** is the last `--cooldown_frac` of steps (default 0.05), which are also recomputed in full so the final molecule is exact.

## Acknowledgements

Built on the official DemoDiff implementation. If you use this code, please also cite DemoDiff: *Graph Diffusion Transformers are In-Context Molecular Designers* ([arXiv:2510.08744](https://arxiv.org/abs/2510.08744)).
