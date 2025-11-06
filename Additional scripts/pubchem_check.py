import pubchempy as pcp
import os
import requests
import json
import csv
from typing import List, Tuple, Union
from urllib.parse import quote

PUG_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

def get_pubchem_results(smiles: str, min_sim: float, max_sim: float) -> List[Tuple[Union[int, str], Union[float, str]]]:
    smiles = smiles.strip()
    if not smiles:
        return [('NEW', 'N/A')] 
    
    if min_sim == 100.0 and max_sim == 100.0:
        try:
            results = pcp.get_compounds(smiles, 'smiles')
            if results:
                return [(results[0].cid, 100.0)]
            else:
                return [('NEW', 'N/A')]
        except Exception as e:
            print(f"Error while matching exactly for '{smiles}': {e}")
            return [('ERROR', 'N/A')]
            
    if min_sim < 100.0:
        if min_sim >= max_sim:
             print(f"Attention: min_sim ({min_sim}) must be < max_sim ({max_sim}).")
             max_sim = 100.0 
             
        smiles_encoded = quote(smiles)

        try:
            url = (
                f"{PUG_REST_BASE}/compound/smiles/{smiles_encoded}/property/TanimotoSimilarity/json"
                f"?Threshold={min_sim}&MaxRecords=200"
            )
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                
                all_results = data['PropertyTable']['Properties']
                filtered_results = []
                
                for prop in all_results:
                    cid = prop.get('CID')
                    sim_score = prop.get('TanimotoSimilarity')
                    
                    if cid is not None and sim_score is not None:
                        if sim_score < max_sim: 
                            filtered_results.append((cid, sim_score))

                if filtered_results:
                    filtered_results.sort(key=lambda x: x[1], reverse=True)
                    return filtered_results
                else:
                    return [('NONE_FOUND', 'N/A')]
            
            return [('NONE_FOUND', 'N/A')] 
            
        except requests.exceptions.RequestException as e:
            if '404' in str(e):
                return [('NONE_FOUND', 'N/A')]
            print(f"Request error for '{smiles}': {e}")
            return [('ERROR', 'N/A')]
        except Exception as e:
            print(f"Unexpected error for '{smiles}': {e}")
            return [('ERROR', 'N/A')]
            
    return [('ERROR', 'N/A')]

def process_smiles_file(input_filename: str):
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}_pubchem_similarity_results.txt"

    is_exact_match = True

    mode = "EXACT_MATCH (100%)"
    header = "SMILES\tCID\n"

    print(f"--- PubChem Search Started ---")
    print(f"Mode: {mode}")
    print(f"Input file (CSV): {input_filename}")
    print(f"Output file (TAB delimited TXT): {output_filename}")
    print(f"------------------------------")

    try:
        with open(input_filename, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            
            rows = list(reader)
            total_smiles = len(rows)

            if 'SMILES' not in reader.fieldnames:
                print(f"ERROR: Column 'SMILES' not found in file '{input_filename}'.")
                print(f"Headings found: {reader.fieldnames}")
                return

        with open(output_filename, 'w', encoding='utf-8') as outfile:
            outfile.write(header)
            
            for i, row in enumerate(rows):
                smiles_string = row.get('SMILES', '').strip()
                
                if not smiles_string:
                    continue

                results = get_pubchem_results(smiles_string, 100.0, 100.0)
                
                cids = [str(r[0]) for r in results]
                scores = [str(r[1]) for r in results]

                cids_str = ";".join(cids)
                scores_str = ";".join(scores)
                
                
                if is_exact_match:
                    if cids_str == 'NEW':
                        final_cids_str = 'NEW'
                        output_line = f"{smiles_string}\t{final_cids_str}\n"
                        status = "NEW (Not Found in PubChem)"
                    elif cids_str == 'ERROR':
                        final_cids_str = 'ERROR'
                        output_line = f"{smiles_string}\t{final_cids_str}\n"
                        status = "ERROR"
                    else:
                        final_cids_str = cids[0]
                        output_line = f"{smiles_string}\t{final_cids_str}\n"
                        status = f"CID={final_cids_str}"
                        
                    outfile.write(output_line)
                    
                else:
                    results_count = len(results) if cids[0] not in ('NEW', 'NONE_FOUND', 'ERROR') else 0

                    outfile.write(f"{smiles_string}\t{cids_str}\t{scores_str}\t{results_count}\n")

                    if results_count > 0:
                        status = f"Found {results_count} hit(s): CID(s)={cids_str}, Score(s)={scores_str}"
                    elif cids_str == 'NEW':
                        status = "NEW (Exact Match Not Found)"
                    elif cids_str == 'NONE_FOUND':
                        status = "NO RESULTS (Similarity Out of Range)"
                    else:
                        status = "ERROR"

                
                print(f"Processed {i + 1}/{total_smiles}: {smiles_string} -> {status}")

        print("\nProcess completed successfully!")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # --- MAIN CONFIGURATION ---
    # ----------------------------------------------------------------------
    
    # Change with your input file name
    input_file = "input_file.csv"

    # ----------------------------------------------------------------------
    
    if not os.path.exists(input_file):
        print(f"WARNING: File '{input_file}' not found.")
        
    process_smiles_file(input_file)
