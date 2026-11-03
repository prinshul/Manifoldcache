import torch
import pandas as pd
import numpy as np
import os
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import argparse
import time
import multiprocessing
from functools import partial

from docking_oracle import DockingOracle
from polymer_oracle import PolymerOracle
from diffusion_model import GraphDiffusionTransformer

def standardize_smiles(smiles):
    """Remove stereochemistry from a SMILES string and return the modified SMILES."""
    if len(smiles) > 1:
        if smiles is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.rdmolops.RemoveStereochemistry(mol)
        smiles_no_stereo = Chem.MolToSmiles(mol)
        
        # Get the largest connected component by selecting the longest SMILES string
        if "." in smiles_no_stereo:
            components = smiles_no_stereo.split(".")
            largest_component = max(components, key=len)
            return largest_component
        return smiles_no_stereo
    else:
        return smiles

def find_most_similar(target_smiles, smiles_list):
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2)

    max_similarity = 0
    most_similar_smiles = None

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        similarity = DataStructs.TanimotoSimilarity(target_fp, fp)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_smiles = smiles

    return most_similar_smiles, max_similarity

def calculate_relationship_score(pos_sim, med_sim, neg_sim):
    # Check if we have all three similarity values
    if pos_sim is None or med_sim is None or neg_sim is None:
        return None, None
    
    # Calculate continuous score based on pairwise differences
    # Calculate pairwise differences in the expected direction
    pos_med_diff = pos_sim - med_sim  # Should be positive
    med_neg_diff = med_sim - neg_sim  # Should be positive
    pos_neg_diff = pos_sim - neg_sim  # Should be positive and largest
    
    # Calculate a continuous score component
    # If differences are in the wrong direction, penalize
    diff_score = 0.0
    if pos_med_diff > 0:
        diff_score += pos_med_diff
    if med_neg_diff > 0:
        diff_score += med_neg_diff
    if pos_neg_diff > 0:
        diff_score += pos_neg_diff
    
    # Normalize the difference score to [0,1] range
    # Maximum possible difference sum would be 3 (if all similarities are in [0,1])
    max_possible_diff = 3.0
    diff_score = min(diff_score / max_possible_diff, 1.0)
    
    # Since weight_rank = 0, final_score = diff_score
    final_score = diff_score
    
    return final_score, None  # Return None for rank_score since it's not used

def calculate_context_similarities(target_smiles, context):
    """Calculate similarity between target molecule and each context type"""
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2)
    
    similarities = {}
    
    for context_type in ['pos', 'med', 'neg']:
        context_smiles_list = [item[0] for item in context[context_type]]
        if not context_smiles_list:
            similarities[context_type] = None
            continue
            
        sim_values = []
        for smiles in context_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                sim_values.append(sim)
        
        # Calculate mean similarity instead of max
        if sim_values:
            similarities[context_type] = sum(sim_values) / len(sim_values)
        else:
            similarities[context_type] = 0
    
    return similarities

def process_molecule(target_context, context_smiles_dict, is_polymer=False):
    target, context = target_context
    if target is None or 'None' in target or not Chem.MolFromSmiles(target):
        return None
            
    # Only check for polymer-specific validation if is_polymer is True
    if is_polymer and target.count('*') % 2 != 0:
        return None
    
    # Skip if the target molecule is already in the context
    all_context_smiles = []
    for smiles_list in context_smiles_dict.values():
        all_context_smiles.extend(smiles_list)
    if target in all_context_smiles:
        return None
    # Calculate similarity to each context type
    context_similarities = calculate_context_similarities(target, context)
    
    # Calculate consistency score
    consistency_score, rank_score = calculate_relationship_score(
        context_similarities.get('pos'), 
        context_similarities.get('med'), 
        context_similarities.get('neg')
    )
    
    # Skip molecules with consistency score below threshold if all similarities are available
    if None not in context_similarities.values() and consistency_score < 0:
        return None
    
    # max score in pos
    if len(context['pos']) > 0:
        max_context_score = max([item[1] for item in context['pos']])
        max_context_smiles = [item[0] for item in context['pos'] if item[1] == max_context_score]
    else:
        max_context_score = -1
        max_context_smiles = ""
    
    return {
        'smiles': target,
        'consistency_score': consistency_score if consistency_score is not None else 0,
        'pos_similarity': context_similarities.get('pos'),
        'med_similarity': context_similarities.get('med'),
        'neg_similarity': context_similarities.get('neg'),
        'pos_context': context['pos'],
        'med_context': context['med'],
        'neg_context': context['neg'],
        'max_score_context': (max_context_smiles, max_context_score)
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference tasks with Demodiff')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--num_samples', type=int, default=32, help='Number of samples to generate')
    parser.add_argument('--task_list', nargs='+', default=['Albuterol_Similarity'], 
                        help='List of tasks to run')
    parser.add_argument('--sample_steps', type=int, default=500,
                        help='Number of timesteps for sampling')
    parser.add_argument('--deterministic', action='store_true', 
                        help='Use deterministic sampling')
    parser.add_argument('--output_dir', type=str, default='log/tmp_output', 
                        help='Directory to save output files')
    parser.add_argument('--model_path', type=str, default='pretrained/model.pt',
                        help='Path to the pretrained model')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of top molecules to consider for each generated molecule')
    parser.add_argument('--data_dir', type=str, default='context_data',
                        help='Directory to save context data')

    args = parser.parse_args()
    
    # Use arguments instead of hardcoded values
    path_to_model = args.model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    graph_dit = GraphDiffusionTransformer(path_to_model, device, guide_scale=2, sample_timesteps=args.sample_steps)

    for task in args.task_list:
        if task == 'Median_1':
            task = 'Median 1'
        elif task == 'Median_2':
            task = 'Median 2'
        print(f"\n=== Processing task: {task} ===")
        try:
            pos_df = pd.read_csv(f'{args.data_dir}/{task}/positive.csv')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'No positive data for {task}')
            pos_df = pd.DataFrame(columns=['smiles', 'score'])
            
        try:
            neg_df = pd.read_csv(f'{args.data_dir}/{task}/negative.csv')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'No negative data for {task}')
            neg_df = pd.DataFrame(columns=['smiles', 'score'])
        
        try:
            med_df = pd.read_csv(f'{args.data_dir}/{task}/medium.csv')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'No medium data for {task}')
            med_df = pd.DataFrame(columns=['smiles', 'score'])
        
        if med_df.empty:
            med_df = neg_df.head(len(neg_df) // 2)
            neg_df = neg_df.tail(len(neg_df) // 2)
        if pos_df.empty:
            pos_df = med_df.head(len(med_df) // 2)
            med_df = med_df.tail(len(med_df) // 2)

        # Add frequency column to each dataframe with initial value of 1
        pos_df['frequency'] = 1
        neg_df['frequency'] = 1
        med_df['frequency'] = 1

        # Generate molecules
        all_valid_results = []
        unique_list = set()  # Track unique SMILES strings generated

        # Create a list of training SMILES to exclude from results
        train_smiles_list = set()
        for df in [pos_df, med_df, neg_df]:
            for smiles in df['smiles']:
                std_smiles = standardize_smiles(smiles)
                if std_smiles:
                    train_smiles_list.add(std_smiles)

        pos_df = pos_df.sort_values(by='score', ascending=False)
        neg_df = neg_df.sort_values(by='score', ascending=False)
        med_df = med_df.sort_values(by='score', ascending=False)

        context_smiles_standardized = {}
        context_smiles_standardized['pos'] = [standardize_smiles(smiles) for smiles in list(pos_df['smiles'])]
        context_smiles_standardized['med'] = [standardize_smiles(smiles) for smiles in list(med_df['smiles'])]
        context_smiles_standardized['neg'] = [standardize_smiles(smiles) for smiles in list(neg_df['smiles'])]

        graph_dit.prepare_data(pos_df, neg_df, med_df, batch_size=args.batch_size)    
        print(f"Phase 1: Generating {args.num_samples} valid molecules...")
        time_start = time.time()
        while len(all_valid_results) < args.num_samples:
            graph_dit.prepare_data(pos_df, neg_df, med_df, batch_size=args.batch_size)
            context_target_map = graph_dit.generate(task_name=task, deterministic=args.deterministic)
            
            # Process generated molecules in parallel
            num_cores = 5
            with multiprocessing.Pool(processes=num_cores) as pool:
                # Check if this is a polymer task
                is_polymer_task = task.startswith("polymer_")
                process_func = partial(process_molecule, 
                                      context_smiles_dict=context_smiles_standardized,
                                      is_polymer=is_polymer_task)
                
                # Process molecules in order
                results = pool.map(process_func, context_target_map.items())                
                new_results = [r for r in results if r is not None]
                if len(new_results) == 0:
                    continue

                filtered_results = []

                for result in new_results:
                    smiles = result['smiles']
                    std_smiles = standardize_smiles(smiles)
                    
                    # Skip if molecule is in training data or already generated
                    if std_smiles in train_smiles_list or std_smiles in unique_list:
                        continue
                    
                    unique_list.add(std_smiles)
                    filtered_results.append(result)
                
                # Add unique molecules to valid results
                all_valid_results.extend(filtered_results)
                print(f"Added {len(filtered_results)} unique molecules")

            print(f"Generated {len(all_valid_results)}/{args.num_samples} valid molecules, time taken: {time.time() - time_start:.4f} seconds")
            
            if len(all_valid_results) >= args.num_samples:
                all_valid_results = all_valid_results[:args.num_samples]
                break
        
        # Second phase: Select top-k molecules by consistency score and evaluate with oracle
        print(f"Phase 2: Selecting top {args.top_k} molecules by consistency score and evaluating...")
        all_valid_results = sorted(all_valid_results, key=lambda x: x['consistency_score'], reverse=True)[:args.top_k]
        
        # Initialize oracle for evaluation
        if task.startswith("polymer_"):
            oracle = PolymerOracle(task)
        elif task.startswith("docking_"):
            oracle = DockingOracle(task)
        else:
            oracle = Oracle(name=task)
        
        # Calculate oracle scores for selected molecules
        for result in all_valid_results:
            result['score'] = oracle([result['smiles']])[0]
    
        # Sort results by score in descending order
        results_df = pd.DataFrame(all_valid_results)        
        results_df = results_df.sort_values(by='score', ascending=False)
        results_df = results_df[['smiles', 'score', 'consistency_score']]
        
        # Save results
        results_df.to_csv(f'{output_dir}/{task}.csv', index=False)
        print(f"Results saved to {output_dir}/{task}.csv")

    print(f"\nAll tasks completed.")

if __name__ == '__main__':
    main()
