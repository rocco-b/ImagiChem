import numpy as np
import random
import cv2
import math
import sys
import os
import hashlib
import re
from collections import defaultdict, Counter, deque
from cores import CORE_LIBRARY
from groups import GROUP_LIBRARY
from graph_utils import MoleculeGraph, VALENCES, ALLOWED_ATOMS, HETERO, HALOGENS, is_bond_allowed, RNG
try:
    from rdkit import Chem
    import rdkit
    from rdkit.Chem import AllChem, Draw, Descriptors
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from rdkit.Contrib.SA_Score import sascorer
except Exception as e:
    raise ImportError("RDKit not found. Install RDKit (conda-forge). Error: " + str(e))

def image_to_seed(path: str) -> int:
    try:
        with open(path, 'rb') as f:
            file_bytes = np.fromfile(f, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    except IOError:
        img = None
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    h = hashlib.sha256(img.tobytes()).hexdigest()
    return int(h, 16) % (2**32)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    RNG.seed(seed)

def split_pixel_rows(image_path):
    try:
        try:
            with open(image_path, 'rb') as f:
                file_bytes = np.fromfile(f, dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        except IOError:
            image = None
        
        if image is None:
            raise FileNotFoundError(f"Impossible to read the file '{image_path}'. Check the file path.")
            
        pixel_rows = image.tolist()
        return pixel_rows
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

class PixelLineAnalyzer:
    def __init__(self, pixel_line):
        self.pixel_line = np.array(pixel_line, dtype=float)
        self.length = len(self.pixel_line)
        if self.length < 3:
            raise ValueError("Pixel line must contain 3 values or more.")

    def analyze_pixel_pattern(self):
        mean_val = np.mean(self.pixel_line)
        std_val = np.std(self.pixel_line)
        gradient = np.diff(self.pixel_line)
        
        peaks = self._find_peaks()
        valleys = self._find_valleys()
        
        return {
            'mean': mean_val,
            'std': std_val,
            'num_peaks': len(peaks),
            'contrast': np.max(self.pixel_line) - np.min(self.pixel_line)
        }
    
    def _find_peaks(self, threshold=0.3):
        peaks = []
        if np.ptp(self.pixel_line) == 0:
            return []
        
        max_val = np.max(self.pixel_line)
        min_val = np.min(self.pixel_line)
        threshold_val = min_val + threshold * (max_val - min_val)
        
        for i in range(1, self.length - 1):
            if (self.pixel_line[i] > self.pixel_line[i-1] and 
                self.pixel_line[i] > self.pixel_line[i+1] and 
                self.pixel_line[i] > threshold_val):
                peaks.append(i)
                
        return peaks
    
    def _find_valleys(self, threshold=0.3):
        valleys = []
        if np.ptp(self.pixel_line) == 0:
            return []

        max_val = np.max(self.pixel_line)
        min_val = np.min(self.pixel_line)
        threshold_val = max_val - threshold * (max_val - min_val)
        
        for i in range(1, self.length - 1):
            if (self.pixel_line[i] < self.pixel_line[i-1] and 
                self.pixel_line[i] < self.pixel_line[i+1] and 
                self.pixel_line[i] < threshold_val):
                valleys.append(i)
                
        return valleys
    
    def generate_molecular_formula_string(self, rng, allow_halogens=False):
        features = self.analyze_pixel_pattern()
        
        c_range = (15, 40)
        c_count = int(self._map_value(features['mean'], 0, 255, c_range[0], c_range[1]))
        c_count = int(np.clip(c_count, c_range[0], c_range[1]))
        
        total_atoms_approx = int(self._map_value(features['contrast'], 0, 255, 10, 80))
        
        other_atoms_max = total_atoms_approx - c_count
        if other_atoms_max < 5:
            c_count = max(c_range[0], int(total_atoms_approx * 0.5))

        atoms = {'C': c_count}
        atom_ranges = {
            'O': (1, 10), 'N': (1, 8), 'S': (0, 2),
            'F': (0, 1), 'Cl': (0, 1), 'Br': (0, 1), 'I': (0, 1)
        }
        
        for atom, (min_val, max_val) in atom_ranges.items():
            if atom == 'O':
                count = int(self._map_value(features['mean'], 0, 255, min_val, max_val))
            elif atom == 'N':
                count = int(self._map_value(features['std'], 0, 100, min_val, max_val))
            elif atom == 'S':
                if features['num_peaks'] > 250:
                    count = 1
                else:
                    count = 0
            elif atom in HALOGENS: 
                count = 0
                if allow_halogens:
                    count = rng.randint(min_val, max_val) 
            else: 
                count = 0
            
            count = np.clip(count, min_val, max_val)
            
            current_total = sum(atoms.values()) + count
            if current_total > total_atoms_approx:
                count = max(min_val, total_atoms_approx - sum(atoms.values()))
            
            if count > 0:
                atoms[atom] = count
        
        formula_parts = []
        for atom, count in atoms.items():
            formula_parts.extend([atom] * count)
        
        rng.shuffle(formula_parts)
        extended_formula = ''.join(formula_parts)
        
        return extended_formula
    
    def _map_value(self, value, in_min, in_max, out_min, out_max):
        if in_max == in_min:
            return out_min
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def merge_graph_into(target: MoleculeGraph, source: MoleculeGraph, attach_to_target=None, attach_from_source_idx=None):
    mapping = {}
    for old_idx in sorted(source.elements.keys()):
        el = source.elements[old_idx]
        coords = source.coords.get(old_idx, (0.0,0.0))
        new_idx = target.new_atom(el, coords=coords)
        mapping[old_idx] = new_idx
    for a,b,order in source.bonds():
        target.adj[mapping[a]][mapping[b]] = order
        target.adj[mapping[b]][mapping[a]] = order
        target.valence_rem[mapping[a]] -= order
        target.valence_rem[mapping[b]] -= order
    if attach_to_target is not None and attach_from_source_idx is not None:
        target.add_bond(attach_to_target, mapping[attach_from_source_idx], order=1)
    return mapping

def parse_atoms(input_str: str):
    tokens = re.findall(r'Cl|Br|C|N|O|S|I|F', input_str)
    if not tokens:
        raise ValueError("Input string not valid. No atoms found.")
    for t in tokens:
        if t not in ALLOWED_ATOMS:
            raise ValueError("Atom not allowed: " + t)
    return Counter(tokens)

def count_graph_atoms(g: MoleculeGraph):
    return g.atom_counts()

def consume_from_pool(pool: Counter, needed: Counter):
    for el, cnt in needed.items():
        if pool[el] < cnt:
            raise ValueError("Not enough atoms in the pool.")
        pool[el] -= cnt

def demote_some_double_bonds_until_capacity(g: MoleculeGraph, needed_hosts: int):
    def count_hosts():
        return sum(1 for v in g.valence_rem.values() if v >= 1)
    if count_hosts() >= needed_hosts:
        return True
    bonds = [(a, b, order) for a, b, order in g.bonds() if order >= 2]
    bonds.sort(key=lambda x: (0 if (g.elements[x[0]] == "C" or g.elements[x[1]] == "C") else 1))
    for a, b, _ in bonds:
        g.demote_bond(a, b)
        if count_hosts() >= needed_hosts:
            return True
    return count_hosts() >= needed_hosts

def choose_core_by_pool(pool: Counter):
    candidates = []
    for core_info in CORE_LIBRARY:
        can_build = all(pool.get(atom, 0) >= count for atom, count in core_info["requirements"].items())

        if can_build:
            candidates.append(core_info)

    if not candidates:
        return None, None

    chosen_core = RNG.choice(candidates)
    return chosen_core['name'], chosen_core['factory']


def choose_host(g: MoleculeGraph, max_candidates=6):
    cand = [n for n, v in g.valence_rem.items() if v >= 1]
    if not cand:
        return None
    cand.sort(key=lambda n: (len(g.neighbors(n)), -g.valence_rem[n]))
    top = cand[:max(1, min(max_candidates, max(1, len(cand)//2)))]
    return RNG.choice(top)

def choose_sinton_or_atom(pool: Counter):
    total = sum(pool.values())
    if total == 0:
        return None
    if pool["O"] >= 2 and pool["C"] >= 1 and RNG.random() < 0.45:
        return "ester"
    if pool["N"] >= 1 and pool["O"] >= 1 and pool["C"] >= 1 and RNG.random() < 0.45:
        return "amide"
    if pool["N"] >= 1 and pool["C"] >= 2 and RNG.random() < 0.25:
        return "amine"
    if pool["O"] >= 1 and RNG.random() < 0.2:
        return "alcohol"
    if pool["N"] >= 1 and RNG.random() < 0.15:
        return "nitro"
    if pool["C"] >= 1 and RNG.random() < 0.55:
        most = pool.most_common()
        return most[0][0]
    most = pool.most_common()
    return most[0][0]

def choose_next_elem_for_chain(pool: Counter):
    candidates = [(VALENCES[el], el) for el, cnt in pool.items() if cnt > 0]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    chosen = candidates[0][1]
    pool[chosen] -= 1
    return chosen

def assemble_from_input_string(input_str: str):
    pool = parse_atoms(input_str)
    g = MoleculeGraph()

    n_cores = 1
    if sum(pool.values()) >= 12:
        n_cores = RNG.choice([1, 2])

    atoms_in_first_core = set()

    for i in range(n_cores):
        core_name, core_factory = choose_core_by_pool(pool)
        if core_factory is None:
            continue
        
        try:
            core_info = next(item for item in CORE_LIBRARY if item["name"] == core_name)
            needed_atoms = core_info["requirements"]
        except StopIteration:
            print(f"Attention: core '{core_name}' not found in the library.", file=sys.stderr)
            continue
            
        core_graph = core_factory()
        host_for_second_core = None

        if g.elements and i > 0:
            possible_hosts = [idx for idx in g.elements if idx not in atoms_in_first_core and g.valence_rem[idx] >= 1]
            
            if possible_hosts:
                host_for_second_core = RNG.choice(possible_hosts)
            else:
                if pool.get('C', 0) > 0:
                    needed_atoms['C'] = needed_atoms.get('C', 0) + 1
                else:
                    continue

        if not all(pool.get(el, 0) >= count for el, count in needed_atoms.items()):
            continue

        consume_from_pool(pool, needed_atoms)

        if not g.elements:
            mapping = merge_graph_into(g, core_graph)
            atoms_in_first_core = set(mapping.values())
        else:
            attach_from_source_idx = RNG.choice(list(core_graph.elements.keys()))
            if host_for_second_core:
                merge_graph_into(g, core_graph, attach_to_target=host_for_second_core, attach_from_source_idx=attach_from_source_idx)
            else:
                linker_attach_point = RNG.choice(list(atoms_in_first_core)) if atoms_in_first_core else RNG.choice(list(g.elements.keys()))

                if g.valence_rem.get(linker_attach_point, 0) >= 1:
                    linker_atom_idx = g.new_atom("C")
                    g.add_bond(linker_attach_point, linker_atom_idx, order=1)
                    merge_graph_into(g, core_graph, attach_to_target=linker_atom_idx, attach_from_source_idx=attach_from_source_idx)

    if not g.elements:
        if not pool:
            raise ValueError("Input string not valid. No atoms found to start.")
        most_common_atom = pool.most_common(1)[0][0]
        pool[most_common_atom] -= 1
        g.new_atom(most_common_atom)

    while sum(pool.values()) > 0:
        possible_groups = []
        for group_info in GROUP_LIBRARY:
            if all(pool.get(el, 0) >= cnt for el, cnt in group_info["requirements"].items()):
                if RNG.random() < group_info.get("weight", 0.5):
                    possible_groups.append(group_info)
        
        group_added = False
        if possible_groups:
            chosen_group = RNG.choice(possible_groups)
            factory = chosen_group['factory']
            need = chosen_group['requirements']
            
            host = choose_host(g)
            if host is None:
                demote_some_double_bonds_until_capacity(g, 1)
                host = choose_host(g)

            if host is not None:
                consume_from_pool(pool, need)
                merge_graph_into(g, factory(), attach_to_target=host, attach_from_source_idx=0)
                group_added = True
        
        if group_added:
            continue

        elem_to_add = choose_next_elem_for_chain(pool)
        if elem_to_add is None:
            break
        
        new_idx = g.new_atom(elem_to_add)
        connected = False

        all_possible_hosts = [idx for idx, rem in g.valence_rem.items() if rem >= 1 and idx != new_idx]
        RNG.shuffle(all_possible_hosts)

        for host in all_possible_hosts:
            if g.add_bond(host, new_idx, order=1):
                connected = True
                break
        
        if not connected:
            demote_some_double_bonds_until_capacity(g, 2)
            all_possible_hosts = [idx for idx, rem in g.valence_rem.items() if rem >= 1 and idx != new_idx]
            for host in all_possible_hosts:
                if g.add_bond(host, new_idx, order=1):
                    connected = True
                    break

    def components(graph: MoleculeGraph):
        seen = set()
        comps = []
        for node in graph.elements.keys():
            if node in seen: continue
            q = deque([node])
            comp = set()
            while q:
                u = q.popleft()
                if u in comp: continue
                comp.add(u); seen.add(u)
                for v in graph.adj[u]:
                    if v not in comp: q.append(v)
            comps.append(comp)
        return comps

    comps = components(g)
    attempts = 0
    while len(comps) > 1 and attempts < 200:
        comps.sort(key=len)
        comp_a, comp_b = comps[0], comps[1]
        
        node_a = next((n for n in comp_a if g.valence_rem[n] >= 1), None)
        if node_a is None:
            demote_some_double_bonds_until_capacity(g, g.total_capacity() + 1)
            node_a = next((n for n in comp_a if g.valence_rem[n] >= 1), None)

        node_b = next((n for n in comp_b if g.valence_rem[n] >= 1), None)
        if node_b is None:
            demote_some_double_bonds_until_capacity(g, g.total_capacity() + 1)
            node_b = next((n for n in comp_b if g.valence_rem[n] >= 1), None)

        if node_a is None or node_b is None:
            print("Attention: impossible connect all molecular fragments.", file=sys.stderr)
            break
            
        if g.add_bond(node_a, node_b, order=1):
            comps = components(g)
        
        attempts += 1


    rwmol = graph_to_rwmol(g)
    mol = rwmol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        for a, b, order in list(g.bonds()):
            while g.adj[a][b] >= 2:
                g.demote_bond(a, b)
        rwmol = graph_to_rwmol(g)
        mol = rwmol.GetMol()
        Chem.SanitizeMol(mol)

    AllChem.Compute2DCoords(mol)
    return mol, g

def graph_to_rwmol(g: MoleculeGraph):
    rwmol = Chem.RWMol()
    idx_map = {}
    for idx, elem in sorted(g.elements.items()):
        atom = Chem.Atom(elem)
        ai = rwmol.AddAtom(atom)
        idx_map[idx] = ai
    from rdkit.Chem import BondType
    for a, b, order in g.bonds():
        bt = BondType.SINGLE
        if order == 2:
            bt = BondType.DOUBLE
        elif order >= 3:
            bt = BondType.TRIPLE
        if a in idx_map and b in idx_map:
            rwmol.AddBond(idx_map[a], idx_map[b], bt)
    return rwmol

def refine_smiles(smiles: str) -> str:
    frammenti = smiles.split('.')

    if len(frammenti) == 1:
        return smiles

    frammento_piu_grande = max(frammenti, key=len)

    return frammento_piu_grande

_pains_catalog = None

def initialize_pains_filters():
    global _pains_catalog
    if _pains_catalog is not None:
        return

    rdkit_data_dir = os.environ.get("RDKIT_DATA")
    if not rdkit_data_dir:
        try:
            rdkit_base = os.path.dirname(rdkit.__file__)
            rdkit_data_dir = os.path.join(rdkit_base, 'Data')
        except Exception:
            raise RuntimeError("Impossible to found RDKit's 'Data' directory. Set the 'RDKIT_DATA' environment variable.")

    pains_dir = os.path.join(rdkit_data_dir, 'Pains')
    if not os.path.isdir(pains_dir):
        raise RuntimeError(f"Directory PAINS not found in '{pains_dir}'. RDKit installing may be incomplete.")

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    
    _pains_catalog = FilterCatalog(params)

def check_pains(mol: Chem.Mol) -> str:
    if _pains_catalog is None:
        initialize_pains_filters()
    
    if not mol:
        return "SMILES_error"
    
    try:
        entry = _pains_catalog.GetFirstMatch(mol)
        return entry.GetDescription() if entry else "None"
    except Exception:
        return "PAINS_error"

def calculate_sa_score(mol: Chem.Mol) -> float:
    if not mol:
        return -1.0

    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return -1.0

def run_imagichem_processing(image_path, progress_callback):
    try:
        image_seed = image_to_seed(image_path)
        set_seed(image_seed)
        pixel_rows = split_pixel_rows(image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Fatal error reading image file: {e}")
        return []

    if not pixel_rows:
        return []

    lista_formule = []
    for i, pixel_row in enumerate(pixel_rows):
        try:
            analyzer = PixelLineAnalyzer(pixel_row)
            can_add_halogens = (i % 20 == 0)
            formula_string = analyzer.generate_molecular_formula_string(RNG, allow_halogens=can_add_halogens)
            lista_formule.append(formula_string)
        except ValueError:
            lista_formule.append(None)
    
    lista_smiles = []
    total_formulas = len(lista_formule)
    for i, formula in enumerate(lista_formule):
        if formula:
            try:
                mol, graph = assemble_from_input_string(formula)
                smiles = Chem.MolToSmiles(mol, canonical=True)
                lista_smiles.append(smiles)
            except Exception:
                pass
        
        progress = int((i + 1) / total_formulas * 100)
        progress_callback(progress)

    results_with_scores = []
    for smiles in lista_smiles:
        if smiles:
            frammento_principale = refine_smiles(smiles)
            mol = Chem.MolFromSmiles(frammento_principale)

            sa_score = calculate_sa_score(mol)
            pains_match = check_pains(mol)
            
            results_with_scores.append((frammento_principale, sa_score, pains_match))

    results_with_scores.sort(key=lambda x: x[1]) 
    
    return results_with_scores