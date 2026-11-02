# ManifoldCache on DiT-MC

Training-free, data-free feature caching for [DiT-MC](https://github.com/ML4MolSim/dit_mc) (diffusion transformers for 3D molecular conformer generation).

## Setup

Follow the setup instructions in the official DiT-MC repository: **https://github.com/ML4MolSim/dit_mc**

1. Install the package with `pip install .` (this also builds the Cython extension `dit_mc/algos.pyx`).
2. Download the data and checkpoints from the [DiT-MC Zenodo record](https://doi.org/10.5281/zenodo.15489212) and copy the GEOM-QM9 tfrecords to `./data` (see "Preparation" in the DiT-MC README).

## Run

```bash
cd dit_mc
python generate_confs_Manifoldcache.py --workdir <checkpoint_dir> --shallow_stride <S_shallow> --deep_stride <S_deep> --warmup tstar
```

`<checkpoint_dir>` is a downloaded DiT-MC checkpoint folder (e.g. `checkpoints/qm9/apeB`). DiT-MC caches the output of the whole network, which is reused for ⌊(`S_shallow` + `S_deep`) / 2⌋ ODE steps (defaults 2 and 4, so 3 steps). Results are saved to `<checkpoint_dir>/pred_mols_epoch<step>_<warmup>_<S_shallow>_<S_deep>_CC.pkl` (e.g. `pred_mols_epoch250_tstar_2_4_CC.pkl`); `--no_cache` runs the uncached baseline.

**Warmup** is the first, noisiest part of sampling, where the network is recomputed at every step and nothing is cached. `--warmup tstar` ends it automatically at T*, the first of the 50 ODE steps where log-SNR(τ) ≥ 0 (τ ≥ ½, signal outweighs noise); without it, warmup is the first `--warmup_frac` of steps (default 0.10). **Cooldown** is the last `--cooldown_frac` of steps (default 0.05), which are also recomputed in full so the final conformer is exact.

## Acknowledgements

Built on the official DiT-MC implementation. If you use this code, please also cite DiT-MC:

```bibtex
@inproceedings{frank2025sampling,
  title     = {Sampling 3D Molecular Conformers with Diffusion Transformers},
  author    = {Frank, J. Thorben and Ripken, Winfried and Lied, Gregor and M\"uller, Klaus-Robert and Unke, Oliver T. and Chmiela, Stefan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) 2025},
  year      = {2025},
  note      = {Poster},
  url       = {https://neurips.cc/virtual/2025/poster/115555}
}
```
