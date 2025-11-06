#!/usr/bin/env python3

import sys
from rdkit import Chem
from rdkit.Chem import QED

def calculate_qed_from_file(input_path, output_path):
    print(f"[*] Start of the process...")
    print(f"[*] Input file: {input_path}")
    print(f"[*] Output file: {output_path}")

    try:
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            outfile.write("SMILES\QED\n")

            for line in infile:
                smiles = line.strip()
                if not smiles:
                    continue

                mol = Chem.MolFromSmiles(smiles)

                if mol is None:
                    qed_value = "Not valid"
                    print(f"[!] Error: Invalid SMILES -> {smiles}")
                else:
                    qed_value = QED.qed(mol)
                    qed_value = f"{qed_value:.4f}"

                outfile.write(f"{smiles}\t{qed_value}\n")

        print(f"[*] Process completed successfully. Results saved in '{output_path}'.")

    except FileNotFoundError:
        print(f"[!] Error: The input file '{input_path}' was not found.")
    except Exception as e:
        print(f"[!] An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python qed.py <file input.text> <file output.txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    calculate_qed_from_file(input_file, output_file)