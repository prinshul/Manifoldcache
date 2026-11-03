# DemoDiff: Graph Diffusion Transformers are In-Context Molecular Designers
> DemoDiff is a diffusion-based molecular foundation model for **in-context inverse molecular design**. It leverages graph diffusion transformers to generate molecules based on contextual examples, enabling few-shot molecular design across diverse chemical tasks without task-specific fine-tuning.

[![arXiv](https://img.shields.io/badge/arXiv-2510.08744-b31b1b.svg)](https://arxiv.org/abs/2510.08744)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/liuganghuggingface/DemoDiff-0.7B)
[![HuggingFace Data (Downstream)](https://img.shields.io/badge/🤗%20Hugging%20Face-Data-blue)](https://huggingface.co/datasets/liuganghuggingface/demodiff_downstream)

## 🚀 Quick Start

### Prerequisites

### 1. Environment Setup

Create and activate a new conda environment:
```bash
conda create --name demodiff_inf python=3.11 -y
conda activate demodiff_inf
```

Install dependencies using the provided setup script:
```bash
cd demodiff/cleaned/downstream/
bash seteup_env.sh
```

### 2. Download Pretrained Model

Download the DemoDiff-0.7B model from HuggingFace:
```bash
python prepare_model.py
```

This will download:
- `pretrained/model.pt` (~1.8GB) - The pretrained model weights
- `pretrained/config.yaml` - Model configuration
- `data/processed/vocab3000ring300/pretrain.meta.json` - Pretraining data statistics after tokenization

### 3. Download Benchmark Datasets

Download all 30+ benchmark tasks and evaluation oracles:
```bash
python prepare_data_and_oracle.py
```

This will download:
- **Context Data**: 30+ molecular optimization tasks in `context_data/`
  - Each task contains `positive.csv`, `medium.csv`, `negative.csv`
  - Tasks include similarity, rediscovery, MPO, docking, and polymer design
- **Evaluation Oracles**: Property prediction models in `oracle/`
  - Polymer gas separation oracles (CO₂, CH₄, H₂, N₂, O₂)

### 4. Verify Installation

Test your setup by running a quick inference:
```bash
python inference.py --task_name Albuterol_Similarity
```

