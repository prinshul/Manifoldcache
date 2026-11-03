import warnings
warnings.filterwarnings('ignore')
from rdkit import Chem

import sys
import os
current_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(current_file) + "/docking")
from scorer import get_docking_scores

class DockingOracle:
    def __init__(self, task_name):
        """
        Initialize DockingOracle for a specific protein.
        
        Args:
            task_name (str): Task name like "docking_protein"
        """
        self.task_name = task_name
        
        # Extract gas names from task name
        # Expected format: "docking_protein"
        parts = task_name.split('_')
        if len(parts) != 2 or parts[0] != 'docking':
            raise ValueError(f"Invalid task name format. Expected 'docking_protein', got {task_name}")
        protein_name = parts[1]
        assert protein_name in ['jak2', 'braf', 'fa7', 'parp1', '5ht1b']
        self.protein_name = protein_name

    def __call__(self, smiles_list):
        mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        scores = get_docking_scores(self.protein_name, mol_list, True)
        # get_docking_scores('jak2', mol_list, True)
        # print(f"docking scores: {scores}")
        return scores

# Example usage and testing
if __name__ == "__main__":
    # Test the DockingOracle
    oracle = DockingOracle("docking_jak2")
    
    SMILES_list = [
        'B1OCC2=CC=CC=C12.c1ncnc2[nH]ccc12',
        'CCC1=CCCCC1.N1=CNC2[nH]CNC(=O)C12',
        'CC(=O)N.OC(c1ccccc1)c2ccccc2',
        'NCc1ccccc1.CC1(C)CCCCC1',
        'CC(C)NCCO.Oc1c(O)cccc1',
        "CC(C)NCC(O)C(C)(O)c1ccc(O)cc1"
    ]
    
    scores = oracle(SMILES_list)
    print(f"docking scores: {scores}")
