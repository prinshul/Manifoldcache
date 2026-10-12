# Proteina Model Card <br>

# Overview

## Description: <br>
Proteina is a state-of-the-art generative model of protein structures that generates digital representations of protein backbone structures. It is trained with a flow matching objective and sampled iteratively starting from random noise, using either deterministic or stochastic sampling. It enables a protein designer to generate digital representation of new protein structures unconditionally, with fold class guidance or conditioning on motif structures. Fold class guidance is implemented through a classifier-free guidance scheme. 

This model is ready for non-commercial use and research and development.<br>

### License/Terms of Use: <br> 
Proteina is released under an NVIDIA license for non-commercial or research purposes only, please see [LICENSE](LICENSE).

### Deployment Geography:
Global

### Use Case: <br>
Proteina can be used by protein designers interested in generating novel protein backbone structures.

### Release Date:  <br>
February 28, 2025 <br>

## Reference(s):
The associated paper, *"Proteina: Scaling Flow-based Protein Structure Generative Models"*, can be found here https://openreview.net/forum?id=TVQLu34bdw.  <br> 

## Model Architecture: <br> 
**Architecture Type:** Flow model  <br>
**Network Architecture:** Transformer

We use a new non-equivariant transformer architecture with pair bias in the attention layers and optional triangle multiplicative layers for refining the pair representation. The architecture operates on the protein backbone’s three-dimensional carbon-alpha coordinates, which are iteratively updated during the generation process. The model parametrizes the flow that maps the noise distribution to the generated distribution.  <br>

## Input:<br> 
**Input Type(s):** 

- Text (time step schedule, noise schedule, sampling mode, motif coordinates) <br>

- Number (number of residues, noise scale, time step size, seed, noise schedule exponent, guidance weight, autoguidance weight) <br>

- Binary (use of self-conditioning, use of fold conditioning) <br>

**Input Format(s):** 

- Text: Strings (time step schedule, noise schedule, sampling mode), PDB file (motif coordinates) <br>

- Number: Integers (number of residues, seed), floats (noise scale, time step size, noise schedule exponent, guidance weight, autoguidance weight) <br>

- Binary: Booleans <br>

**Input Parameters:** 

- Text: 1D or text file (PDB file)

- Number: 1D

- Binary: 1D

**Other Properties Related to Input:** All inputs are handled and specified in the config yaml files, see README. 

## Output: <br>
**Output Type(s):** Text (generated protein backbone coordinates) <br>
**Output Format:** Text: PDB file (generated protein backbone coordinates) <br>

## Software Integration: <br>
**Runtime Engine(s):** Pytorch <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
NVIDIA Ampere (tested on A100) <br>

**[Preferred/Supported] Operating System(s):** <br>
Linux <br>

## Model Version(s): 
We release eight model checkpoints:
- Proteina v1.1 (trained on D_FS, with ~200M transformer and ~15M triangle layer parameters)
- Proteina v1.2 (trained on D_FS, with ~200M transformer parameters, no triangle layers)
- Proteina v1.3 (trained on D_FS, with ~60M transformer parameters, no triangle layers)
- Proteina v1.4 (trained on D_21M, with ~400M transformer and ~15M triangle layer parameters)
- Proteina v1.5 (v1.1, fine-tuned with LoRA on PDB subset)
- Proteina v1.6 (v1.2, fine-tuned for long protein generation)
- Proteina v1.7 (trained on D_FS for motif scaffolding, with ~60M transformer and ~12M triangle layer parameters)
- Proteina v1.8 (a "weak" checkpoint of Proteina v1.4 from early in training, after 10k steps. This checkpoint is used as a guidance model in the autoguidance experiments)

# Training and Evaluation Datasets: 

For additional information regarding the datasets, please see the paper here https://openreview.net/forum?id=TVQLu34bdw.

## Training Datasets:

AlphaFold Protein Structure Database (AFDB) 
- Link: https://alphafold.ebi.ac.uk/
- Data Collection Method by dataset: Synthetic (AlphaFold predictions)
- Labeling Method by dataset: N/A (no labels)
- Properties: The AlphaFold Protein Structure Database (AFDB) contains approx. 214M synthetic three-dimensional protein structures predicted by AlphaFold2, along with their corresponding sequences. We trained Proteina on two filtered subsets of the AFDB, one comprising 588,318 structures, the other one comprising 20,874,485 structures. 

Protein Data Bank (PDB)
- Link: https://www.rcsb.org/
- Data Collection Method by dataset: Automatic/Sensors/Human (experimental protein structure determination)
- Labeling Method by dataset: N/A (no labels)
- Properties: The Protein Data Bank (PDB) contains approx. 200K experimentally determined three-dimensional structures of large biological molecules, such as proteins and nucleic acids, along with auxiliary information such as the protein sequences. In one experiment, we used LoRA to fine-tune Proteina on a filtered subset of the PDB, comprising 90,423 proteins. 

The Encyclopedia of Domains (TED) structural domains assignments for AlphaFold Database
- Link: https://zenodo.org/records/13908086
- Data Collection Method by dataset: Synthetic 
- Labeling Method by dataset: Automated
- Properties: TED provides the CATH fold class labels for the majority of the structures in the AFDB. We use all available labels for our AFDB-based training sets, excluding the homologous superfamily level labels (for the 588,318-sized training set, 99.9% of the structures are labeled; for the 20,874,485-sized training set, 69.7% of the structures are labeled).

## Evaluation Datasets :
AlphaFold Protein Structure Database (AFDB) 
- Link: https://alphafold.ebi.ac.uk/
- Data Collection Method by dataset: Synthetic (AlphaFold predictions)
- Labeling Method by dataset: N/A  (no labels)
- Properties: The AFDB subset of 588,318 structures that was used to train Proteina was also used as a reference set in evaluations. 

Protein Data Bank (PDB)
- Link: https://www.rcsb.org/
- Data Collection Method by dataset: Automatic/Sensors/Human (experimental protein structure determination)
- Labeling Method by dataset: N/A (no labels)
- Properties: In different evaluations we used either the entire PDB or a subset of size 15,357 as reference set. 

## Evaluation Results
Extensive benchmarks and evaluations can be found in the associated paper, https://openreview.net/forum?id=TVQLu34bdw.

## Inference:
**Engine:** Pytorch <br>
**Test Hardware:** A100 <br>

## Ethical Considerations:
Users are responsible for ensuring the physical properties of model-generated molecules and proteins are appropriately evaluated and comply with applicable safety regulations and ethical standards.

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).



