## Template-based Refinement for DiffCSP++

This folder is inspired by [CSPML](https://github.com/Minoru938/CSPML), including

```
data : Fingerprints related to CSPML implementation. 
templated_models: Pre-trained CSPML models
training_scripts: Scripts for training a custom CSPML model
```

Training CSPML models

```
# Data extraction
python training_scripts/pair_generate.py --raw_data_dir <path containing raw data (train/val/test csv files)> --fp_data_dir <output path to save fingerprints>

# Training
python training_scripts/train_cspml.py --fp_data_dir <path containing fingerprints> --save_dir <CSPML model dir>
```

Running the code for generation

```
python script/csp_from_template.py --model_path <csp model> --csv_path <path containing *_comp_fps.csv (fp_data_dir)> --finder_model_path <CSPML model>
```
