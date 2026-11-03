import os
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class PolymerOracle:
    def __init__(self, task_name):
        """
        Initialize PolymerOracle for a specific gas selectivity task.
        
        Args:
            task_name (str): Task name like "polymer_CO2_CH4"
        """
        self.task_name = task_name
        
        # Extract gas names from task name
        # Expected format: "polymer_GAS1_GAS2"
        parts = task_name.split('_')
        if len(parts) != 3 or parts[0] != 'polymer':
            raise ValueError(f"Invalid task name format. Expected 'polymer_GAS1_GAS2', got {task_name}")
        
        self.gas1 = parts[1]  # First gas (numerator in selectivity)
        self.gas2 = parts[2]  # Second gas (denominator in selectivity)
        
        # Define selectivity parameters (from 3_create_polymer_oracle.py)
        self.bd_dict_x = {
            'CO2_CH4': [1.00E+05, 1.00E-02], 
            'H2_CH4': [5.00E+04, 2.50E+00], 
            'O2_N2': [5.00E+04, 1.00E-03], 
            'H2_N2': [1.00E+05, 1.00E-01], 
            'CO2_N2': [1.00E+06, 1.00E-04]
        }
        
        self.bd_dict_y = {
            'CO2_CH4': [1.00E+05/2.21E+04, 1.00E-02/4.88E-06], 
            'H2_CH4': [5.00E+04/8.67E+04, 2.50E+00/5.64E-04], 
            'O2_N2': [5.00E+04/2.78E+04, 1.00E-03/2.43E-05], 
            'H2_N2': [1.00E+05/1.02E+05, 1.00E-01/9.21E-06], 
            'CO2_N2': [1.00E+06/3.05E+05, 1.00E-04/1.05E-08]
        }
        
        # Calculate selectivity parameters
        self.selectivity_key = f"{self.gas1}_{self.gas2}"
        if self.selectivity_key not in self.bd_dict_x:
            raise ValueError(f"Selectivity pair {self.selectivity_key} not supported")
        
        x1, x2 = np.log10(self.bd_dict_x[self.selectivity_key][0]), np.log10(self.bd_dict_x[self.selectivity_key][1])
        y1, y2 = np.log10(self.bd_dict_y[self.selectivity_key][0]), np.log10(self.bd_dict_y[self.selectivity_key][1])
        self.a = (y1-y2)/(x1-x2)
        self.b = y1 - self.a * x1
        
        # Paths
        abspath = os.path.dirname(os.path.abspath(__file__))
        self.oracle_dir = os.path.join(abspath, "oracle")
        self.data_dir = os.path.join(abspath, "preprocess_polymer")
        
        # Create oracle directory if it doesn't exist
        os.makedirs(self.oracle_dir, exist_ok=True)
        
        # Load or train models
        self.gas1_model = self._load_or_train_model(self.gas1)
        self.gas2_model = self._load_or_train_model(self.gas2)
        
    def _load_or_train_model(self, gas_name):
        """Load existing model or train new one for a specific gas."""
        model_path = os.path.join(self.oracle_dir, f"{gas_name}.pkl")
        
        if os.path.exists(model_path):
            print(f"Loading existing model for {gas_name}...")
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Training new model for {gas_name}...")
            return self._train_model(gas_name, model_path)
    
    def _train_model(self, gas_name, model_path):
        """Train a Random Forest model for gas permeability prediction."""
        # Load training data
        data_path = os.path.join(self.data_dir, f"{gas_name}_raw.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        df = pd.read_csv(data_path)
        smiles_list = df['SMILES'].tolist()
        labels = df[gas_name].values
        
        # Generate Morgan fingerprints
        print(f"Generating Morgan fingerprints for {len(smiles_list)} molecules...")
        fingerprints = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints.append(list(fp))
                valid_indices.append(i)
            else:
                print(f"Warning: Invalid SMILES {smiles}")
        
        if len(fingerprints) == 0:
            raise ValueError(f"No valid SMILES found in {data_path}")
        
        X = np.array(fingerprints)
        y = labels[valid_indices]
        
        # Train Random Forest on all data
        print(f"Training Random Forest for {gas_name} on {len(X)} samples...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Evaluate on the same training data
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"Model performance for {gas_name}: MAE={mae:.4f}, R2={r2:.4f}")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
        
        return model
    
    def _compute_fingerprints(self, smiles_list):
        """Compute Morgan fingerprints for a list of SMILES."""
        fingerprints = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints.append(list(fp))
                valid_indices.append(i)
            else:
                fingerprints.append(None)
                valid_indices.append(None)
        
        return fingerprints, valid_indices
    
    def _selectivity_score(self, gas1_values, gas2_values):
        """
        Compute selectivity score based on gas permeability values.
        
        Args:
            gas1_values: Array of gas1 permeability values
            gas2_values: Array of gas2 permeability values
            
        Returns:
            Array of selectivity scores (0-1)
        """
        gas1_values = np.array(gas1_values)
        gas2_values = np.array(gas2_values)
        
        # Avoid division by zero
        gas2_values = np.where(gas2_values == 0, 1e-10, gas2_values)
        
        x = np.log10(gas1_values)
        y = np.log10(gas1_values / gas2_values)
        
        # Calculate distance from boundary
        upper = y - (self.a * x + self.b)
        
        # Convert to score: shift by 0.5 and clip to [0, 1]
        scores = np.clip(upper + 0.5, 0, 1)
        
        return scores
    
    def __call__(self, smiles_list):
        """
        Compute selectivity scores for a list of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of selectivity scores (0-1)
        """
        if not isinstance(smiles_list, list):
            smiles_list = [smiles_list]
        
        # Compute fingerprints
        fingerprints, valid_indices = self._compute_fingerprints(smiles_list)
        
        scores = []
        
        for i, (fp, valid_idx) in enumerate(zip(fingerprints, valid_indices)):
            if fp is None:
                # Invalid SMILES
                scores.append(0.0)
            else:
                # Predict gas permeabilities
                fp_array = np.array(fp).reshape(1, -1)
                gas1_pred = self.gas1_model.predict(fp_array)[0]
                gas2_pred = self.gas2_model.predict(fp_array)[0]
                
                # Ensure positive predictions
                gas1_pred = max(gas1_pred, 1e-10)
                gas2_pred = max(gas2_pred, 1e-10)
                
                # Compute selectivity score
                score = self._selectivity_score([gas1_pred], [gas2_pred])[0]
                scores.append(float(score))
        
        return scores

# Example usage and testing
if __name__ == "__main__":
    # Test the PolymerOracle
    oracle = PolymerOracle("polymer_CO2_CH4")
    
    # Test with some example SMILES
    test_smiles = [
        "*=CC1CCC(C1)C=*",
        "*C(=C(*)C1=CC2=C(C=C1)C1=C(C2)C=CC=C1)C1=CC=CC=C1",
        "*c1ccc2c(c1)-c1nc3ccc(-c4ccc5c(c4)nc4n5C(=O)c5cc(C(c6ccc(-c7nc8cc(C(c9ccc%10oc(-c%11ccc(C(*)(C(F)(F)F)C(F)(F)F)cc%11)nc%10c9)(C(F)(F)F)C(F)(F)F)ccc8o7)cc6)(C(F)(F)F)C(F)(F)F)ccc5-4)cc3n1C2=O"
    ]
    
    scores = oracle(test_smiles)
    print(f"Selectivity scores: {scores}")
