import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import argparse

def download_context_data(repo_name="liuganghuggingface/demodiff_downstream", local_path="./context_data"):
    """
    Download all task data from Hugging Face repository to local context_data folder,
    organized by task name folders and CSV files.
    
    Args:
        repo_name (str): Hugging Face repository name
        local_path (str): Local path to download data (default: ./context_data)
    """
    local_root = Path(local_path)
    
    try:
        print(f"Downloading data from Hugging Face repository: {repo_name}")
        print(f"Local destination: {local_root.absolute()}")
        
        # Create local directory if it doesn't exist
        local_root.mkdir(parents=True, exist_ok=True)
        
        # List all files in the repository
        print("Listing repository files...")
        files = list_repo_files(repo_id=repo_name, repo_type="dataset")
        
        # Filter for CSV files (they should be in task folders directly)
        csv_files = [f for f in files if f.endswith(".csv") and "/" in f and not f.startswith(".")]
        
        if not csv_files:
            print("✗ No CSV files found in repository")
            return False
        
        print(f"Found {len(csv_files)} CSV files to download")
        
        downloaded_count = 0
        task_folders = set()
        
        for file_path in csv_files:
            try:
                # Parse task name from path: TaskName/file.csv
                parts = file_path.split('/')
                if len(parts) >= 2:
                    task_name = parts[0]
                    csv_filename = parts[1]
                    
                    # Track task folders
                    task_folders.add(task_name)
                    
                    # Create local task directory
                    local_task_dir = local_root / task_name
                    local_task_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    local_file_path = hf_hub_download(
                        repo_id=repo_name,
                        filename=file_path,
                        repo_type="dataset",
                        local_dir=str(local_root),
                        local_dir_use_symlinks=False
                    )
                    
                    print(f"  ✓ Downloaded: {task_name}/{csv_filename}")
                    downloaded_count += 1
                    
            except Exception as e:
                print(f"  ✗ Failed to download {file_path}: {e}")
        
        print(f"\n✓ Downloaded {downloaded_count} files from {len(task_folders)} task folders")
        return downloaded_count > 0
        
    except Exception as e:
        print(f"✗ Error downloading from Hugging Face: {e}")
        print("Make sure the repository exists and is accessible")
        return False

def download_oracle_pickles(repo_name="liuganghuggingface/demodiff_downstream", local_oracle_path="./oracle"):
    """
    Download oracle pickle files from HuggingFace dataset repository to local oracle folder
    
    Args:
        repo_name (str): Hugging Face repository name
        local_oracle_path (str): Local path to download oracle files (default: ./oracle)
    """
    # Create local oracle directory if it doesn't exist
    oracle_dir = Path(local_oracle_path)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading oracle files from {repo_name}...")
    print(f"Local destination: {oracle_dir.absolute()}")
    
    try:
        # List all files in the repository
        print("Listing repository files...")
        files = list_repo_files(repo_id=repo_name, repo_type="dataset")
        
        # Filter for pickle files in the root directory (not in subfolders)
        oracle_files = [f for f in files if f.endswith(".pkl") and "/" not in f and not f.startswith(".")]
        
        if not oracle_files:
            print("✗ No oracle pickle files found in repository root")
            return False
        
        print(f"Found {len(oracle_files)} oracle pickle files to download:")
        for f in oracle_files:
            print(f"  - {f}")
        
        downloaded_count = 0
        
        for filename in oracle_files:
            try:
                print(f"Downloading {filename}...")
                
                # Download file to local oracle directory
                local_file_path = hf_hub_download(
                    repo_id=repo_name,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(oracle_dir),
                    local_dir_use_symlinks=False
                )
                
                print(f"✓ Successfully downloaded {filename}")
                downloaded_count += 1
                
            except Exception as e:
                print(f"✗ Error downloading {filename}: {str(e)}")
        
        print(f"\n✓ Downloaded {downloaded_count} oracle files to {oracle_dir.absolute()}")
        return downloaded_count > 0
        
    except Exception as e:
        print(f"✗ Error accessing Hugging Face repository: {e}")
        print("Make sure the repository exists and is accessible")
        return False

def verify_context_data(local_path="./context_data"):
    """
    Verify that the downloaded data has the correct structure.
    
    Args:
        local_path (str): Local path to verify (default: ./context_data)
    """
    local_root = Path(local_path)
    
    print(f"\nVerifying downloaded data in: {local_root.absolute()}")
    
    if not local_root.exists():
        print("✗ Context data directory does not exist")
        return False
    
    task_count = 0
    file_count = 0
    csv_types = {"positive.csv": 0, "medium.csv": 0, "negative.csv": 0}
    
    for item in local_root.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            csv_files = list(item.glob("*.csv"))
            if csv_files:  # Only count directories with CSV files
                task_count += 1
                file_count += len(csv_files)
                
                print(f"  {item.name}: {len(csv_files)} CSV files")
                for csv_file in csv_files:
                    print(f"    - {csv_file.name}")
                    if csv_file.name in csv_types:
                        csv_types[csv_file.name] += 1
    
    print(f"\nSummary:")
    print(f"Tasks found: {task_count}")
    print(f"CSV files found: {file_count}")
    print(f"File distribution:")
    for csv_type, count in csv_types.items():
        print(f"  {csv_type}: {count} tasks")
    
    return task_count > 0 and file_count > 0

def verify_oracle_files(repo_name="liuganghuggingface/demodiff_downstream", local_oracle_path="./oracle"):
    """
    Verify that the local oracle folder contains all pickle files from the remote repository
    
    Args:
        repo_name (str): Hugging Face repository name
        local_oracle_path (str): Local path to verify (default: ./oracle)
    
    Returns:
        bool: True if all remote files are present locally, False otherwise
    """
    oracle_dir = Path(local_oracle_path)
    
    print(f"Verifying oracle files in: {oracle_dir.absolute()}")
    print(f"Against remote repository: {repo_name}")
    
    try:
        # Get remote pickle files
        print("Checking remote repository...")
        remote_files = list_repo_files(repo_id=repo_name, repo_type="dataset")
        remote_pickles = [f for f in remote_files if f.endswith(".pkl") and "/" not in f and not f.startswith(".")]
        
        if not remote_pickles:
            print("✗ No oracle pickle files found in remote repository")
            return False
        
        print(f"Remote repository contains {len(remote_pickles)} oracle files:")
        for f in sorted(remote_pickles):
            print(f"  - {f}")
        
        # Check local oracle directory
        if not oracle_dir.exists():
            print(f"\n✗ Local oracle directory does not exist: {oracle_dir.absolute()}")
            print("Run with --download-oracle to create and populate the oracle directory")
            return False
        
        # Get local pickle files
        local_pickles = list(oracle_dir.glob("*.pkl"))
        local_pickle_names = [f.name for f in local_pickles]
        
        print(f"\nLocal oracle directory contains {len(local_pickles)} files:")
        for pkl_file in sorted(local_pickles):
            file_size = pkl_file.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"  - {pkl_file.name} ({file_size:.2f} MB)")
        
        # Compare remote vs local
        missing_files = set(remote_pickles) - set(local_pickle_names)
        extra_files = set(local_pickle_names) - set(remote_pickles)
        
        print(f"\nVerification Results:")
        
        if not missing_files and not extra_files:
            print("✓ Perfect match! All remote oracle files are present locally")
            print(f"✓ Total files verified: {len(remote_pickles)}")
            return True
        
        if missing_files:
            print(f"✗ Missing files ({len(missing_files)} files not downloaded):")
            for f in sorted(missing_files):
                print(f"    - {f}")
        
        if extra_files:
            print(f"⚠ Extra files ({len(extra_files)} files not in remote):")
            for f in sorted(extra_files):
                print(f"    - {f}")
        
        if missing_files:
            print("\nTo download missing files, run with --download-oracle")
        
        return len(missing_files) == 0
        
    except Exception as e:
        print(f"✗ Error during verification: {e}")
        return False

def list_available_tasks(local_path="./context_data"):
    """
    List all available tasks in the context data folder.
    
    Args:
        local_path (str): Local path to check (default: ./context_data)
    
    Returns:
        list: List of task names
    """
    local_root = Path(local_path)
    
    if not local_root.exists():
        print(f"✗ Context data directory does not exist: {local_root.absolute()}")
        return []
    
    tasks = []
    for item in local_root.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            csv_files = list(item.glob("*.csv"))
            if csv_files:  # Only include directories with CSV files
                tasks.append(item.name)
    
    return sorted(tasks)

def list_oracle_files(local_oracle_path="./oracle"):
    """
    List oracle pickle files in the local oracle directory
    
    Args:
        local_oracle_path (str): Local path to check (default: ./oracle)
    
    Returns:
        list: List of oracle pickle file names
    """
    oracle_dir = Path(local_oracle_path)
    
    if not oracle_dir.exists():
        print(f"✗ Oracle directory does not exist: {oracle_dir.absolute()}")
        return []
    
    pickle_files = list(oracle_dir.glob("*.pkl"))
    
    if pickle_files:
        print(f"Oracle files found in {oracle_dir.absolute()}:")
        for pkl_file in sorted(pickle_files):
            file_size = pkl_file.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"  - {pkl_file.name} ({file_size:.2f} MB)")
    else:
        print(f"No oracle pickle files found in {oracle_dir.absolute()}")
    
    return [f.name for f in pickle_files]

def get_task_data_path(task_name, local_path="./context_data"):
    """
    Get the path to a specific task's data folder.
    
    Args:
        task_name (str): Name of the task
        local_path (str): Local path to context data (default: ./context_data)
    
    Returns:
        Path: Path to the task folder, or None if not found
    """
    local_root = Path(local_path)
    task_path = local_root / task_name
    
    if task_path.exists() and task_path.is_dir():
        csv_files = list(task_path.glob("*.csv"))
        if csv_files:
            return task_path
    
    return None

def main():
    """
    Main function to handle command line arguments and execute data preparation.
    """
    parser = argparse.ArgumentParser(description="Prepare DemoDiff context data and oracle files for inference")
    
    # Context data arguments
    parser.add_argument("--download", action="store_true", help="Download context data from Hugging Face")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded context data structure")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument("--path", default="./context_data", help="Local path for context data (default: ./context_data)")
    
    # Oracle data arguments
    parser.add_argument("--download-oracle", action="store_true", help="Download oracle pickle files from Hugging Face")
    parser.add_argument("--verify-oracle", action="store_true", help="Verify oracle files against remote repository")
    parser.add_argument("--list-oracle", action="store_true", help="List local oracle files")
    parser.add_argument("--oracle-path", default="./oracle", help="Local oracle directory path (default: ./oracle)")
    
    # Combined arguments
    parser.add_argument("--download-all", action="store_true", help="Download both context data and oracle files")
    parser.add_argument("--verify-all", action="store_true", help="Verify both context data and oracle files")
    
    # Repository argument
    parser.add_argument("--repo", default="liuganghuggingface/demodiff_downstream", 
                       help="Hugging Face repository (default: liuganghuggingface/demodiff_downstream)")
    
    args = parser.parse_args()
    
    success = True
    
    # Handle combined operations first
    if args.download_all:
        print("=== Downloading Context Data ===")
        context_success = download_context_data(args.repo, args.path)
        if context_success:
            verify_context_data(args.path)
        
        print("\n=== Downloading Oracle Files ===")
        oracle_success = download_oracle_pickles(args.repo, args.oracle_path)
        if oracle_success:
            verify_oracle_files(args.repo, args.oracle_path)
        
        return context_success and oracle_success
    
    elif args.verify_all:
        print("=== Verifying Context Data ===")
        context_success = verify_context_data(args.path)
        
        print("\n=== Verifying Oracle Files ===")
        oracle_success = verify_oracle_files(args.repo, args.oracle_path)
        
        return context_success and oracle_success
    
    # Handle individual operations
    elif args.download:
        print("Downloading context data from Hugging Face...")
        success = download_context_data(args.repo, args.path)
        if success:
            verify_context_data(args.path)
    
    elif args.verify:
        success = verify_context_data(args.path)
    
    elif args.list:
        tasks = list_available_tasks(args.path)
        if tasks:
            print(f"Available tasks in {args.path}:")
            for i, task in enumerate(tasks, 1):
                print(f"  {i:2d}. {task}")
            print(f"\nTotal: {len(tasks)} tasks")
        else:
            print(f"No tasks found in {args.path}")
        success = len(tasks) > 0
    
    elif args.download_oracle:
        success = download_oracle_pickles(args.repo, args.oracle_path)
        if success:
            verify_oracle_files(args.repo, args.oracle_path)
    
    elif args.verify_oracle:
        success = verify_oracle_files(args.repo, args.oracle_path)
    
    elif args.list_oracle:
        oracle_files = list_oracle_files(args.oracle_path)
        success = len(oracle_files) > 0
    
    else:
        # Default behavior: download if not exists, otherwise verify
        context_path = Path(args.path)
        oracle_path = Path(args.oracle_path)
        
        context_exists = context_path.exists() and list_available_tasks(args.path)
        oracle_exists = oracle_path.exists() and list_oracle_files(args.oracle_path)
        
        if not context_exists or not oracle_exists:
            print("Missing data detected. Downloading from Hugging Face...")
            
            if not context_exists:
                print("\n=== Downloading Context Data ===")
                context_success = download_context_data(args.repo, args.path)
                if context_success:
                    verify_context_data(args.path)
            else:
                context_success = True
            
            if not oracle_exists:
                print("\n=== Downloading Oracle Files ===")
                oracle_success = download_oracle_pickles(args.repo, args.oracle_path)
                if oracle_success:
                    verify_oracle_files(args.repo, args.oracle_path)
            else:
                oracle_success = True
            
            success = context_success and oracle_success
        else:
            print("All data already exists. Verifying...")
            print("\n=== Verifying Context Data ===")
            context_success = verify_context_data(args.path)
            print("\n=== Verifying Oracle Files ===")
            oracle_success = verify_oracle_files(args.repo, args.oracle_path)
            success = context_success and oracle_success
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
