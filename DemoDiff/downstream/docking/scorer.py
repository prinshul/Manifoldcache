# modifed from: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py

from rdkit import Chem
import numpy as np

from docking import get_dockingvina

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def standardize_mols(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        print('standardize_mols error')
        return None


def standardize_smiles(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception:
        print('standardize_smiles error')
        return None


def get_docking_scores(target, mols, verbose=False):
    dockingvina = get_dockingvina(target)

    smiles = [standardize_smiles(mol) for mol in mols]
    smiles_valid = [smi for smi in smiles if smi is not None]
    
    scores = - np.array(dockingvina.predict_single(smiles_valid)) / 14
    scores = list(np.clip(scores, 0, 1))

    if None in smiles:
        scores = [scores.pop(0) if smi is not None else 0. for smi in smiles]

    return scores

# ['jak2', 'braf', 'fa7', 'parp1', '5ht1b']:
if __name__ == '__main__':
    SMILES_list = ['CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1', 'C1=CC=C(C(=C1)C(=O)O)C(=O)O']
    mol_list = [Chem.MolFromSmiles(smi) for smi in SMILES_list]
    scores = get_docking_scores('jak2', mol_list, True)
    print(scores)