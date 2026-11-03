import os
import yaml
from huggingface_hub import hf_hub_download

def download_demodiff_model():
    """
    Download model.pt and config.yaml from liuganghuggingface/DemoDiff-0.7B
    to the local ./pretrained folder, and pretrain.meta.json to the appropriate
    vocab{x}ring{y} folder based on config.yaml values
    """
    repo_id = "liuganghuggingface/DemoDiff-0.7B"
    local_dir = "./pretrained"
    
    # Create the pretrained directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # First, download config.yaml to read vocab_size and vocab_ring_len
    config_path = None
    try:
        print("Downloading config.yaml...")
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.yaml",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded config.yaml to {config_path}")
    except Exception as e:
        print(f"Error downloading config.yaml: {e}")
        return
    
    # Read config.yaml to get vocab_size and vocab_ring_len
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        vocab_size = config.get('vocab_size')
        vocab_ring_len = config.get('vocab_ring_len')
        
        if vocab_size is None or vocab_ring_len is None:
            print("Error: Could not find vocab_size or vocab_ring_len in config.yaml")
            return
            
        print(f"Found vocab_size: {vocab_size}, vocab_ring_len: {vocab_ring_len}")
    except Exception as e:
        print(f"Error reading config.yaml: {e}")
        return
    
    # Create the vocab{x}ring{y} directory path
    vocab_dir = f"../data/processed/vocab{vocab_size}ring{vocab_ring_len}"
    os.makedirs(vocab_dir, exist_ok=True)
    print(f"Created/verified directory: {vocab_dir}")
    
    # Files to download to pretrained folder
    pretrained_files = ["model.pt"]
    
    # Download files to pretrained folder
    for filename in pretrained_files:
        try:
            print(f"Downloading {filename} to pretrained folder...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {filename} to {downloaded_path}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    
    # Download pretrain.meta.json to vocab{x}ring{y} folder if it doesn't exist
    meta_json_path = os.path.join(vocab_dir, "pretrain.meta.json")
    if not os.path.exists(meta_json_path):
        try:
            print(f"Downloading pretrain.meta.json to {vocab_dir}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename="pretrain.meta.json",
                local_dir=vocab_dir,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded pretrain.meta.json to {downloaded_path}")
        except Exception as e:
            print(f"Error downloading pretrain.meta.json: {e}")
    else:
        print(f"pretrain.meta.json already exists at {meta_json_path}, skipping download")

if __name__ == "__main__":
    download_demodiff_model()