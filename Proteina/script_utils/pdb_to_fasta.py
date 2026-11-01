# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import argparse
import biotite.structure.io.pdb as pdb
from biotite.structure import to_sequence

def pdb_to_fasta(pdb_dir, output_fasta):
    with open(output_fasta, 'w') as fasta_file:
        for filename in os.listdir(pdb_dir):
            if filename.endswith('.pdb'):
                pdb_path = os.path.join(pdb_dir, filename)
                name = os.path.splitext(filename)[0]
                structure = pdb.PDBFile.read(pdb_path)
                array = structure.get_structure()
                sequences, chain_starts = to_sequence(array)
                sequence = str(sequences[0])
                
                fasta_file.write(f">{name}\n{str(sequence)}\n")

def main():
    parser = argparse.ArgumentParser(description='Convert PDB files to FASTA format with filenames as headers')
    parser.add_argument('input_dir', help='Directory containing PDB files')
    parser.add_argument('output_file', help='Output FASTA file path')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        parser.error('Input directory does not exist')
        
    pdb_to_fasta(args.input_dir, args.output_file)

if __name__ == '__main__':
    main()
