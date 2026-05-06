import numpy as np
import random
import cv2
import math
import os
import sys
import hashlib
import re
from collections import defaultdict, Counter, deque
from typing import Optional
from cores import CORE_LIBRARY
from groups import GROUP_LIBRARY
from graph_utils import MoleculeGraph, VALENCES, ALLOWED_ATOMS, HETERO, HALOGENS, is_bond_allowed, RNG
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, Descriptors, QED, rdMolDescriptors, Crippen
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception as e:
    raise ImportError("RDKit was not found. Install RDKit, preferably from conda-forge. Error: " + str(e))
try:
    import pubchempy as pcp
except ImportError:
    pcp = None

CURRENT_IMAGE_PROFILE = {}
CURRENT_CORE_USAGE = Counter()
CURRENT_FAMILY_USAGE = Counter()
CURRENT_GROUP_USAGE = Counter()
CURRENT_SCAFFOLD_USAGE = Counter()
CURRENT_SIGNATURE_USAGE = Counter()
CURRENT_MOTIF_USAGE = Counter()
CURRENT_FINAL_SMILES = Counter()


def imread_unicode_safe(path: str, flags=cv2.IMREAD_COLOR):
    """Robust image reader for Windows/OneDrive/non-ASCII paths."""
    if not path:
        raise FileNotFoundError('Image path was not provided.')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Image not found: {path}')
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise ValueError(f'Image file is empty or not readable: {path}')
        img = cv2.imdecode(data, flags)
        if img is None:
            raise ValueError(f"Unable to decode image: {path}")
        return img
    except Exception:
        img = cv2.imread(path, flags)
        if img is None:
            raise FileNotFoundError(f'Image not found: {path}')
        return img


def _deterministic_unit(value: str) -> float:
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16**12 - 1)


def _classify_core_topology(mol):
    ring_info = mol.GetRingInfo().AtomRings()
    ring_sizes = sorted(len(r) for r in ring_info)
    spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    aromatic_fraction = aromatic_atoms / max(1, mol.GetNumAtoms())
    hetero_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    sulfur_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
    has_small_ring = any(size <= 3 for size in ring_sizes)
    num_rings = len(ring_sizes)

    if spiro_atoms > 0 or bridgeheads > 0:
        family = "spiro_bridged"
    elif has_small_ring:
        family = "small_ring"
    elif num_rings >= 2 and aromatic_fraction >= 0.55:
        family = "aromatic_fused"
    elif aromatic_fraction >= 0.45:
        family = "aromatic_single"
    elif sulfur_atoms > 0 and aromatic_fraction < 0.35:
        family = "sulfurized_aliphatic"
    elif num_rings >= 2 and aromatic_fraction >= 0.18:
        family = "mixed_polycyclic"
    elif num_rings >= 2:
        family = "saturated_polycyclic"
    elif num_rings == 1 and hetero_atoms >= 1:
        family = "simple_heterocycle"
    elif num_rings == 1:
        family = "simple_carbocycle"
    else:
        family = "acyclic"
    return {
        "family": family,
        "ring_sizes": ring_sizes,
        "spiro_atoms": spiro_atoms,
        "bridgeheads": bridgeheads,
        "aromatic_fraction": aromatic_fraction,
        "hetero_atoms": hetero_atoms,
        "sulfur_atoms": sulfur_atoms,
        "num_rings": num_rings,
        "num_atoms": mol.GetNumAtoms(),
    }


def _build_core_metadata():
    metadata = []
    for core in CORE_LIBRARY:
        try:
            g = core["factory"]()
            mol = graph_to_rwmol(g).GetMol()
            mol.UpdatePropertyCache(strict=False)
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                      Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                                      Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                      Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                      Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                                 catchErrors=True)
            except Exception:
                pass
            try:
                Chem.GetSymmSSSR(mol)
            except Exception:
                pass
            topo = _classify_core_topology(mol)
            topo["name"] = core["name"]
            topo["requirements"] = dict(core["requirements"])
            metadata.append(topo)
        except Exception:
            metadata.append({
                "name": core["name"],
                "requirements": dict(core["requirements"]),
                "family": "unknown",
                "ring_sizes": [],
                "spiro_atoms": 0,
                "bridgeheads": 0,
                "aromatic_fraction": 0.0,
                "hetero_atoms": sum(v for k,v in core["requirements"].items() if k != "C"),
                "num_rings": 0,
                "num_atoms": sum(core["requirements"].values()),
            })
    return metadata


CORE_METADATA = []
CORE_METADATA_BY_NAME = {}


def _target_core_families(profile: Optional[dict]):
    profile = profile or {}
    art = profile.get("artistic_bias", 0.45)
    coh = profile.get("spatial_coherence", 0.5)
    comp = profile.get("complexity", 0.5)
    sat = profile.get("saturation_mean", 0.4)
    hue = profile.get("hue_std", 0.2)

    families = []
    if art >= 0.62:
        families.extend(["aromatic_single", "aromatic_fused", "simple_ring"])
        if comp > 0.52:
            families.append("polycyclic_nonaromatic")
    elif coh >= 0.58:
        families.extend(["simple_ring", "aromatic_single", "polycyclic_nonaromatic"])
    else:
        families.extend(["simple_ring", "polycyclic_nonaromatic", "acyclic"])

    if sat > 0.55 or hue > 0.18:
        families.insert(0, "aromatic_single")
    if comp < 0.35:
        families.insert(0, "simple_ring")

    # Per input artistici evita esplicitamente motivi spiro/ciclopropanici dominanti.
    if art < 0.42:
        families.extend(["small_ring", "spiro_bridged"])

    seen = set()
    ordered = []
    for fam in families:
        if fam not in seen:
            ordered.append(fam); seen.add(fam)
    return ordered


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _bucket_int(value: int, step: int = 2) -> int:
    return step * int(value / max(1, step))


def _safe_mol_from_graph(g: 'MoleculeGraph'):
    """Build an RDKit Mol from the current graph and initialize caches/ring info as safely as possible."""
    mol = graph_to_rwmol(g).GetMol()
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        Chem.FastFindRings(mol)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        pass
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return mol


def _motif_counts_from_mol(mol) -> dict:
    motifs = {}
    try:
        patt_amide = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]")
        patt_carbamate = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[OX2][#6]")
        patt_carboxyl = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1,OX1-]")
        patt_aryl_ether = Chem.MolFromSmarts("a-O-[#6]")
        patt_tertiary_amine = Chem.MolFromSmarts("[NX3]([#6])([#6])[#6]")
        motifs['amide'] = len(mol.GetSubstructMatches(patt_amide)) if patt_amide else 0
        motifs['carbamate'] = len(mol.GetSubstructMatches(patt_carbamate)) if patt_carbamate else 0
        motifs['carboxyl'] = len(mol.GetSubstructMatches(patt_carboxyl)) if patt_carboxyl else 0
        motifs['aryl_ether'] = len(mol.GetSubstructMatches(patt_aryl_ether)) if patt_aryl_ether else 0
        motifs['tertiary_amine'] = len(mol.GetSubstructMatches(patt_tertiary_amine)) if patt_tertiary_amine else 0
    except Exception:
        motifs = {'amide': 0, 'carbamate': 0, 'carboxyl': 0, 'aryl_ether': 0, 'tertiary_amine': 0}
    return motifs


def _scaffold_key_from_mol(mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is not None and scaf.GetNumAtoms() > 0:
            return Chem.MolToSmiles(scaf, canonical=True)
    except Exception:
        pass
    try:
        rings = rdMolDescriptors.CalcNumRings(mol)
        arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        het = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        return f"fallback|r{rings}|a{arom}|h{het}|n{mol.GetNumAtoms()}"
    except Exception:
        return "fallback|unknown"


def _signature_key_from_mol(mol) -> str:
    try:
        rings = rdMolDescriptors.CalcNumRings(mol)
        arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        het = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        heavy = mol.GetNumAtoms()
        sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        sp3_bin = int(round(sp3 * 5))
        return f"sig|r{rings}|a{_bucket_int(arom,2)}|h{_bucket_int(het,1)}|n{_bucket_int(heavy,3)}|s{sp3_bin}"
    except Exception:
        return "sig|unknown"


def _novelty_penalty_from_mol(mol, profile: Optional[dict], proposed_group: Optional[str] = None) -> float:
    profile = profile or {}
    penalty = 0.0
    try:
        scaffold_key = _scaffold_key_from_mol(mol)
        scaffold_use = CURRENT_SCAFFOLD_USAGE.get(scaffold_key, 0)
        penalty += min(0.55, 0.06 * scaffold_use)
        sig = _signature_key_from_mol(mol)
        sig_use = CURRENT_SIGNATURE_USAGE.get(sig, 0)
        penalty += min(0.30, 0.035 * sig_use)
        motifs = _motif_counts_from_mol(mol)
        for name, count in motifs.items():
            if count <= 0:
                continue
            penalty += min(0.24, 0.018 * CURRENT_MOTIF_USAGE.get(name, 0) * count)
        if proposed_group:
            penalty += min(0.16, 0.012 * CURRENT_GROUP_USAGE.get(proposed_group, 0))
        # discourage repeated medicinal over-regularization motifs
        penalty += 0.06 * max(0, motifs.get('amide', 0) - 1)
        penalty += 0.08 * max(0, motifs.get('carbamate', 0) - 1)
        penalty += 0.06 * max(0, motifs.get('carboxyl', 0) - 1)
        penalty += 0.05 * max(0, motifs.get('tertiary_amine', 0) - 1)
        art = profile.get('artistic_bias', 0.45)
        if art > 0.58 and motifs.get('amide', 0) > 0 and motifs.get('carbamate', 0) > 0:
            penalty += 0.05
    except Exception:
        pass
    return float(penalty)


def _register_molecule_signature(mol, chosen_group: Optional[str] = None):
    try:
        scaffold_key = _scaffold_key_from_mol(mol)
        CURRENT_SCAFFOLD_USAGE[scaffold_key] += 1
        CURRENT_SIGNATURE_USAGE[_signature_key_from_mol(mol)] += 1
        for name, count in _motif_counts_from_mol(mol).items():
            if count > 0:
                CURRENT_MOTIF_USAGE[name] += count
        if chosen_group:
            CURRENT_GROUP_USAGE[chosen_group] += 1
    except Exception:
        if chosen_group:
            CURRENT_GROUP_USAGE[chosen_group] += 1


def _register_final_smiles(smiles: str):
    CURRENT_FINAL_SMILES[smiles] += 1


def compute_artistic_profile(image_path: str) -> dict:
    """
    Extracts global 2D and color descriptors from the image.
    L'obiettivo è preservare informazioni che distinguono immagini artistiche
    da rumore puro: coerenza spaziale, armonia cromatica e struttura multi-scala.
    """
    img_bgr = imread_unicode_safe(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # downsample per rendere robuste le statistiche e meno sensibili al rumore fine
    small_gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small_gray, (0, 0), sigmaX=3.0)

    diff = np.abs(small_gray - blur)
    fine_noise = float(np.mean(diff) / 255.0)
    spatial_coherence = 1.0 - _clip01(fine_noise * 2.2)

    gx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    edge_density = float(np.mean(grad_mag > np.percentile(grad_mag, 70)))
    structure_strength = _clip01(float(np.std(blur) / 64.0))

    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0
    hue = hsv[:, :, 0] / 179.0

    sat_mean = float(np.mean(sat))
    sat_std = float(np.std(sat))
    brightness_mean = float(np.mean(val))
    brightness_std = float(np.std(val))
    hue_std = float(np.std(hue))

    # armonia cromatica: palette moderata e coerenza di saturazione
    palette_harmony = _clip01(1.0 - abs(sat_mean - 0.45) / 0.45)
    palette_harmony *= _clip01(1.0 - sat_std / 0.35)

    # bias artistico complessivo: favorisce struttura + coerenza + palette non estrema
    artistic_bias = (
        0.40 * spatial_coherence +
        0.25 * palette_harmony +
        0.20 * structure_strength +
        0.15 * _clip01(1.0 - abs(brightness_mean - 0.55) / 0.55)
    )
    artistic_bias = _clip01(artistic_bias)

    # complessità utile ma non caotica
    complexity = _clip01(0.55 * structure_strength + 0.20 * edge_density + 0.25 * brightness_std)

    profile = {
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "saturation_mean": sat_mean,
        "saturation_std": sat_std,
        "hue_std": hue_std,
        "edge_density": edge_density,
        "structure_strength": structure_strength,
        "spatial_coherence": spatial_coherence,
        "palette_harmony": palette_harmony,
        "artistic_bias": artistic_bias,
        "complexity": complexity,
    }
    profile["morphology_class"] = infer_morphology_class(profile)
    return profile


def blend_line_with_global_profile(line_features: dict, global_profile: Optional[dict]) -> dict:
    if not global_profile:
        return dict(line_features)
    merged = dict(line_features)
    merged["artistic_bias"] = 0.6 * global_profile.get("artistic_bias", 0.5) + 0.4 * line_features.get("line_coherence", 0.5)
    merged["spatial_coherence"] = 0.7 * global_profile.get("spatial_coherence", 0.5) + 0.3 * line_features.get("line_coherence", 0.5)
    merged["palette_harmony"] = 0.7 * global_profile.get("palette_harmony", 0.5) + 0.3 * line_features.get("color_harmony", 0.5)
    merged["complexity"] = 0.6 * global_profile.get("complexity", 0.5) + 0.4 * line_features.get("local_complexity", 0.5)
    return merged


def estimate_profile_targets(profile: Optional[dict]) -> dict:
    profile = profile or {}
    art = profile.get("artistic_bias", 0.45)
    comp = profile.get("complexity", 0.5)
    coh = profile.get("spatial_coherence", 0.5)
    # target di proprietà tendenzialmente QED-friendly
    target_mw = 330 + 140 * comp - 60 * art
    target_hetero = 5 + 3.5 * (1.0 - art) + 1.5 * comp
    target_rings = 1.6 + 1.4 * coh + 0.8 * comp
    return {
        "target_mw": float(np.clip(target_mw, 240, 520)),
        "target_hetero": float(np.clip(target_hetero, 3, 10)),
        "target_rings": float(np.clip(target_rings, 1.5, 4.0)),
        "favor_compact": art > 0.58,
    }




def clone_graph(g: MoleculeGraph) -> MoleculeGraph:
    ng = MoleculeGraph()
    for idx, elem in g.elements.items():
        ng.add_atom(idx, elem, g.coords.get(idx, (0.0, 0.0)))
        ng.valence_max[idx] = g.valence_max[idx]
        ng.valence_rem[idx] = g.valence_rem[idx]
    for a, nbrs in g.adj.items():
        for b, order in nbrs.items():
            if a < b:
                ng.adj[a][b] = order
                ng.adj[b][a] = order
    return ng


def graph_component_count(g: MoleculeGraph) -> int:
    if not g.elements:
        return 0
    seen = set()
    count = 0
    for node in g.elements:
        if node in seen:
            continue
        count += 1
        q = deque([node])
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            for v in g.adj[u]:
                if v not in seen:
                    q.append(v)
    return count


def graph_stats(g: MoleculeGraph) -> dict:
    counts = g.atom_counts()
    heavy = sum(counts.values())
    hetero = counts.get('N', 0) + counts.get('O', 0) + counts.get('S', 0)
    sulfur = counts.get('S', 0)
    carbonyls = 0
    tertiary_n = 0
    quaternary_c = 0
    terminal_c = 0
    rotatable_like = 0
    for idx, el in g.elements.items():
        deg = len(g.adj[idx])
        bond_sum = sum(g.adj[idx].values())
        if el == 'N' and deg >= 3:
            tertiary_n += 1
        if el == 'C' and bond_sum >= 4 and deg >= 3:
            quaternary_c += 1
        if el == 'C' and deg <= 1:
            terminal_c += 1
        if el == 'C' and deg == 2 and g.valence_rem[idx] <= 1:
            rotatable_like += 1
    for a, b, order in g.bonds():
        if order == 2:
            ea = g.elements[a]; eb = g.elements[b]
            if {'C', 'O'} == {ea, eb}:
                carbonyls += 1
    return {
        'counts': counts,
        'heavy': heavy,
        'hetero': hetero,
        'sulfur': sulfur,
        'carbonyls': carbonyls,
        'tertiary_n': tertiary_n,
        'quaternary_c': quaternary_c,
        'terminal_c': terminal_c,
        'rotatable_like': rotatable_like,
        'components': graph_component_count(g),
    }


def _target_property_ranges(profile: Optional[dict]) -> dict:
    profile = profile or {}
    art = profile.get('artistic_bias', 0.45)
    comp = profile.get('complexity', 0.5)
    coh = profile.get('spatial_coherence', 0.5)
    # slightly more conservative than the previous heuristic
    target_mw = float(np.clip(290 + 95 * comp - 35 * art, 240, 430))
    target_hetero = float(np.clip(3.8 + 2.4 * (1.0 - art) + 1.2 * comp, 3, 8))
    target_rings = float(np.clip(1.4 + 1.1 * coh + 0.5 * comp, 1.0, 3.5))
    max_sulfur = 0 if art > 0.52 else 1
    return {
        'target_mw': target_mw,
        'target_hetero': target_hetero,
        'target_rings': target_rings,
        'max_sulfur': max_sulfur,
    }


def graph_druglike_score(g: MoleculeGraph, profile: Optional[dict]) -> float:
    profile = profile or {}
    stats = graph_stats(g)
    targets = _target_property_ranges(profile)
    heavy = stats['heavy']
    hetero = stats['hetero']
    sulfur = stats['sulfur']
    carbonyls = stats['carbonyls']
    tertiary_n = stats['tertiary_n']
    quaternary_c = stats['quaternary_c']
    terminal_c = stats['terminal_c']

    score = 0.0
    # graph-level medicinal heuristics
    score -= 0.055 * max(0, heavy - 34)
    score -= 0.020 * max(0, 18 - heavy)
    score -= 0.10 * abs(hetero - targets['target_hetero'])
    score -= 0.55 * max(0, sulfur - targets['max_sulfur'])
    score -= 0.15 * max(0, carbonyls - 2)
    score -= 0.18 * max(0, tertiary_n - 1)
    score -= 0.14 * max(0, quaternary_c - 2)
    if heavy > 0:
        score -= 0.12 * max(0.0, terminal_c / heavy - 0.38)
    if stats['components'] > 1:
        score -= 0.18 * (stats['components'] - 1)

    # RDKit-level refinement where possible
    try:
        mol = graph_to_rwmol(g).GetMol()
        Chem.SanitizeMol(mol)
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        qed = QED.qed(mol)

        score -= 0.014 * abs(mw - targets['target_mw'])
        score -= 0.23 * max(0.0, logp - 4.4)
        score -= 0.16 * max(0.0, 0.9 - logp)
        score -= 0.035 * max(0.0, tpsa - 95)
        score -= 0.045 * max(0.0, 12 - tpsa)
        score -= 0.10 * max(0, hba - 8)
        score -= 0.12 * max(0, hbd - 3)
        score -= 0.16 * abs(rings - targets['target_rings'])
        # prefer moderate sp3, avoid over-saturated bulky output
        score -= 0.35 * max(0.0, sp3 - 0.72)
        score -= 0.20 * max(0.0, 0.18 - sp3)
        score += 1.8 * qed
    except Exception:
        pass
    return float(score)


def property_aware_group_score(group_info: dict, host_idx: int, g: MoleculeGraph, profile: Optional[dict], pool: Counter) -> float:
    base = score_group_candidate(group_info, profile, pool)
    gg = clone_graph(g)
    try:
        frag = group_info['factory']()
        merge_graph_into(gg, frag, attach_to_target=host_idx, attach_from_source_idx=0)
        base += 0.55 * graph_druglike_score(gg, profile)
        host_deg = len(g.neighbors(host_idx))
        # avoid overcrowding already substituted hosts
        base -= 0.12 * max(0, host_deg - 2)
        if group_info['name'] in {'nitro'}:
            base -= 0.30
        if group_info['name'] in {'ester'} and graph_stats(gg)['carbonyls'] > 2:
            base -= 0.18
        if group_info['requirements'].get('S', 0) > 0:
            base -= 0.45
        mol = _safe_mol_from_graph(gg)
        base -= _novelty_penalty_from_mol(mol, profile, proposed_group=group_info['name'])
    except Exception:
        base -= 0.25
        base -= min(0.20, 0.02 * CURRENT_GROUP_USAGE.get(group_info['name'], 0))
    return float(base)


def property_aware_atom_moves(g: MoleculeGraph, pool: Counter, profile: Optional[dict], max_hosts: int = 8):
    host_candidates = [idx for idx, rem in g.valence_rem.items() if rem >= 1]
    host_candidates.sort(key=lambda n: (len(g.neighbors(n)), -g.valence_rem[n]))
    host_candidates = host_candidates[:max_hosts]
    if not host_candidates:
        return []

    counts = pool.copy()
    preferred = []
    targets = _target_property_ranges(profile)
    current = graph_stats(g)
    for elem, cnt in counts.items():
        if cnt <= 0:
            continue
        # conservative atom introduction
        if elem == 'S' and current['sulfur'] >= targets['max_sulfur']:
            continue
        preferred.append(elem)

    moves = []
    for elem in preferred:
        for host in host_candidates:
            gg = clone_graph(g)
            new_idx = gg.new_atom(elem)
            if not gg.add_bond(host, new_idx, order=1):
                continue
            score = graph_druglike_score(gg, profile)
            if elem == 'C' and current['heavy'] >= 32:
                score -= 0.28
            if elem in {'N', 'O'} and current['hetero'] < targets['target_hetero']:
                score += 0.10
            if elem == 'S':
                score -= 0.40
            score -= 0.06 * max(0, len(g.neighbors(host)) - 2)
            try:
                mol = _safe_mol_from_graph(gg)
                score -= _novelty_penalty_from_mol(mol, profile, proposed_group=None)
            except Exception:
                pass
            moves.append((score, elem, host))
    moves.sort(key=lambda x: x[0], reverse=True)
    return moves

def infer_morphology_class(profile: Optional[dict]) -> str:
    profile = profile or {}
    art = profile.get("artistic_bias", 0.45)
    coh = profile.get("spatial_coherence", 0.5)
    comp = profile.get("complexity", 0.5)
    sat = profile.get("saturation_mean", 0.4)
    edge = profile.get("edge_density", 0.4)
    hue = profile.get("hue_std", 0.2)

    if art > 0.68 and coh > 0.72 and comp < 0.52:
        return "serene_compact"
    if comp > 0.62 and edge > 0.34 and sat > 0.34:
        return "crowded_structured"
    if sat > 0.58 and hue > 0.20:
        return "chromatic_expressive"
    if comp < 0.38 and coh > 0.60:
        return "minimal_ordered"
    return "balanced_painterly"


def morphology_family_weights(profile: Optional[dict]) -> dict:
    profile = profile or {}
    morph = infer_morphology_class(profile)
    # Default: broad medicinal-like coverage without any single dominant valley.
    weights = {
        "acyclic": 0.06,
        "simple_carbocycle": 0.17,
        "simple_heterocycle": 0.17,
        "aromatic_single": 0.18,
        "aromatic_fused": 0.05,
        "mixed_polycyclic": 0.16,
        "saturated_polycyclic": 0.10,
        "sulfurized_aliphatic": 0.04,
        "small_ring": 0.03,
        "spiro_bridged": 0.02,
        "unknown": 0.02,
    }
    if morph == "serene_compact":
        weights.update({
            "simple_carbocycle": 0.22,
            "simple_heterocycle": 0.22,
            "aromatic_single": 0.24,
            "mixed_polycyclic": 0.12,
            "saturated_polycyclic": 0.05,
            "aromatic_fused": 0.03,
            "sulfurized_aliphatic": 0.01,
            "small_ring": 0.01,
            "spiro_bridged": 0.00,
        })
    elif morph == "crowded_structured":
        weights.update({
            "simple_carbocycle": 0.15,
            "simple_heterocycle": 0.16,
            "aromatic_single": 0.16,
            "mixed_polycyclic": 0.24,
            "saturated_polycyclic": 0.10,
            "aromatic_fused": 0.06,
            "sulfurized_aliphatic": 0.03,
        })
    elif morph == "chromatic_expressive":
        weights.update({
            "simple_carbocycle": 0.16,
            "simple_heterocycle": 0.16,
            "aromatic_single": 0.18,
            "mixed_polycyclic": 0.18,
            "saturated_polycyclic": 0.10,
            "aromatic_fused": 0.05,
            "sulfurized_aliphatic": 0.06,
        })
    elif morph == "minimal_ordered":
        weights.update({
            "acyclic": 0.10,
            "simple_carbocycle": 0.24,
            "simple_heterocycle": 0.20,
            "aromatic_single": 0.22,
            "mixed_polycyclic": 0.10,
            "saturated_polycyclic": 0.04,
            "aromatic_fused": 0.03,
            "sulfurized_aliphatic": 0.01,
        })

    art = profile.get("artistic_bias", 0.45)
    if art > 0.55:
        weights["aromatic_fused"] *= 0.55
        weights["spiro_bridged"] *= 0.30
        weights["small_ring"] *= 0.45
        weights["sulfurized_aliphatic"] *= 0.55
    for k in list(weights):
        weights[k] = max(0.002, float(weights[k]))
    return weights


def score_group_candidate(group_info: dict, profile: Optional[dict], pool: Counter) -> float:
    profile = profile or {}
    art = profile.get("artistic_bias", 0.45)
    coh = profile.get("spatial_coherence", 0.5)
    comp = profile.get("complexity", 0.5)
    name = group_info["name"]
    need = group_info["requirements"]
    hetero_load = need.get("O", 0) + need.get("N", 0) + need.get("S", 0)

    base = float(group_info.get("weight", 0.3))
    score = base + 0.12 * coh

    if name in {"amide", "amine"}:
        score += 0.10 * art
    if name in {"alcohol", "ether"}:
        score += 0.08 * art
    if name == "ester":
        score += 0.04 * comp
    if name == "nitro":
        score -= 0.28 * art
        score -= 0.18 * coh
    if profile.get("morphology_class") in {"serene_compact", "minimal_ordered"}:
        if name in {"amide", "ester"}:
            score -= 0.04
        if name == "ether":
            score += 0.03

    # penalizza gruppi troppo etero-carichi quando si vuole migliorare QED/SA
    score -= 0.08 * max(0, hetero_load - 1) * (0.6 + art)

    # lieve premio se il gruppo consuma atomi che altrimenti rimarrebbero sovra-rappresentati
    if need.get("O", 0) and pool.get("O", 0) > pool.get("C", 0) * 0.25:
        score += 0.03
    if need.get("N", 0) and pool.get("N", 0) > 4:
        score += 0.03

    return max(0.01, score)


#-----------------------------------------------------------------Seed setup
def image_to_seed(path: str) -> int:
    """
    Computes a deterministic seed from the SHA-256 hash of the image color content.
    In questo modo si preserva anche l'informazione cromatica, utile per distinguere
    immagini artistiche da input puramente rumorosi.
    """
    img = imread_unicode_safe(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    h = hashlib.sha256(img.tobytes()).hexdigest()
    return int(h, 16) % (2**32)

def set_seed(seed: int):
    """
    Sets the seed for all random generators used by the script.
    """
    # Set the seed for Python's standard random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for the custom RNG used by the project
    RNG.seed(seed)

#-----------------------------------------------------------------Leggi image ed estrai righe pixel
def split_pixel_rows(image_path):
    try:
        image = imread_unicode_safe(image_path, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Unable to read file '{image_path}'. Make sure the path is correct.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixel_rows = [row for row in image]
        return pixel_rows

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#-----------------------------------------------------------------Parser da riga pixel a stringa atomica
class PixelLineAnalyzer:
    """
    Analyzes a pixel row to generate a molecular formula based
    sulle sue caratteristiche.
    """
    def __init__(self, pixel_line):
        # Supporta sia righe grayscale sia RGB.
        self.pixel_line = np.array(pixel_line, dtype=float)
        if self.pixel_line.ndim == 2 and self.pixel_line.shape[1] >= 3:
            rgb = self.pixel_line[:, :3]
            self.gray_line = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=float))
            rgb_norm = np.clip(rgb / 255.0, 0.0, 1.0)
            maxc = rgb_norm.max(axis=1)
            minc = rgb_norm.min(axis=1)
            self.sat_line = np.divide(maxc - minc, maxc + 1e-6)
            self.color_line = rgb_norm
        else:
            self.gray_line = self.pixel_line.reshape(-1)
            self.sat_line = np.zeros_like(self.gray_line, dtype=float)
            self.color_line = None
        self.length = len(self.gray_line)
        if self.length < 3:
            raise ValueError("La riga di pixel deve contenere almeno 3 valori.")

    def analyze_pixel_pattern(self):
        """Analizza il pattern dei pixel per estrarre caratteristiche numeriche."""
        mean_val = np.mean(self.gray_line)
        std_val = np.std(self.gray_line)
        gradient = np.diff(self.gray_line)
        
        peaks = self._find_peaks()
        valleys = self._find_valleys()
        
        return {
            'mean': mean_val,
            'std': std_val,
            'gradient_mean': np.mean(gradient) if len(gradient) > 0 else 0,
            'num_peaks': len(peaks),
            'num_valleys': len(valleys),
            'contrast': np.max(self.gray_line) - np.min(self.gray_line),
            'saturation_mean': float(np.mean(self.sat_line)),
            'saturation_std': float(np.std(self.sat_line)),
            'line_coherence': _clip01(1.0 - np.mean(np.abs(gradient)) / 64.0),
            'local_complexity': _clip01((std_val / 64.0) * 0.6 + (len(peaks) / max(8.0, self.length * 0.08)) * 0.4),
            'color_harmony': _clip01(1.0 - abs(float(np.mean(self.sat_line)) - 0.45) / 0.45)
        }
    
    def _find_peaks(self, threshold=0.3):
        """Trova i picchi nella riga di pixel."""
        peaks = []
        # Evita errori se la linea è piatta (max_val == min_val)
        if np.ptp(self.gray_line) == 0:
            return []
        
        max_val = np.max(self.gray_line)
        min_val = np.min(self.gray_line)
        threshold_val = min_val + threshold * (max_val - min_val)
        
        for i in range(1, self.length - 1):
            if (self.gray_line[i] > self.gray_line[i-1] and 
                self.gray_line[i] > self.gray_line[i+1] and 
                self.gray_line[i] > threshold_val):
                peaks.append(i)
                
        return peaks
    
    def _find_valleys(self, threshold=0.3):
        """Trova le valli nella riga di pixel."""
        valleys = []
        # Evita errori se la linea è piatta
        if np.ptp(self.gray_line) == 0:
            return []

        max_val = np.max(self.gray_line)
        min_val = np.min(self.gray_line)
        threshold_val = max_val - threshold * (max_val - min_val)
        
        for i in range(1, self.length - 1):
            if (self.gray_line[i] < self.gray_line[i-1] and 
                self.gray_line[i] < self.gray_line[i+1] and 
                self.gray_line[i] < threshold_val):
                valleys.append(i)
                
        return valleys
    
    def generate_molecular_formula_string(self, rng, allow_halogens=False, global_profile=None):
        """
        Generates a molecular formula using local and global features.
        This version also uses a global morphological class derived from the image
        per differenziare meglio immagini serene/compatte da scene affollate.
        """
        features = blend_line_with_global_profile(self.analyze_pixel_pattern(), global_profile)
        artistic_bias = features.get('artistic_bias', 0.45)
        coherence = features.get('spatial_coherence', 0.5)
        complexity = features.get('complexity', 0.5)
        morph = infer_morphology_class(features)

        base_atoms = 20 + 18 * complexity + 6 * features['saturation_mean'] - 8 * artistic_bias
        if morph == 'serene_compact':
            base_atoms -= 3.5
        elif morph == 'crowded_structured':
            base_atoms += 2.0
        elif morph == 'minimal_ordered':
            base_atoms -= 2.0
        total_atoms_approx = int(np.clip(base_atoms - 1.5 * artistic_bias, 18, 42))

        carbon_fraction = 0.58 + 0.16 * artistic_bias + 0.06 * coherence - 0.05 * features['saturation_mean']
        if morph in {'serene_compact', 'minimal_ordered'}:
            carbon_fraction += 0.03
        carbon_fraction = float(np.clip(carbon_fraction, 0.56, 0.82))
        c_count = int(np.clip(round(total_atoms_approx * carbon_fraction), 14, 34))
        atoms = {'C': c_count}

        hetero_budget = int(np.clip(total_atoms_approx - c_count, 2, 12))
        o_target = int(np.clip(round(1 + 1.1 * (1.0 - artistic_bias) + 0.5 * complexity + 0.7 * features['saturation_mean']), 1, 4))
        n_target = int(np.clip(round(1 + 0.9 * (1.0 - artistic_bias) + 0.4 * complexity + 0.4 * features['std'] / 90.0), 1, 4))

        s_target = 0
        if morph == 'crowded_structured' and complexity > 0.76 and coherence > 0.52 and features['num_peaks'] > max(12, self.length // 18):
            s_target = 1
        elif morph == 'chromatic_expressive' and complexity > 0.80 and artistic_bias < 0.46:
            s_target = 1

        if morph == 'serene_compact':
            o_target = min(o_target, 3)
            n_target = min(n_target, 3)
            s_target = 0
        if morph == 'minimal_ordered':
            o_target = min(o_target, 2)
            n_target = min(n_target, 2)
            s_target = 0

        targets = {'O': o_target, 'N': n_target, 'S': s_target}
        total_hetero = sum(targets.values())
        if total_hetero > hetero_budget:
            scale = hetero_budget / max(1, total_hetero)
            for k in ['O', 'N']:
                targets[k] = max(1, int(round(targets[k] * scale)))
            targets['S'] = 1 if (targets['S'] and hetero_budget >= 5 and morph in {'crowded_structured', 'chromatic_expressive'}) else 0

        for atom in ['O', 'N', 'S']:
            if targets.get(atom, 0) > 0:
                atoms[atom] = int(targets[atom])

        if allow_halogens and morph in {'serene_compact', 'balanced_painterly'} and artistic_bias > 0.62 and coherence > 0.58:
            atoms['F'] = atoms.get('F', 0) + 1

        current_total = sum(atoms.values())
        if current_total < 18:
            atoms['C'] += 18 - current_total
        elif current_total > 46:
            excess = current_total - 46
            for atom in ['O', 'N', 'S', 'C']:
                if excess <= 0:
                    break
                floor = 14 if atom == 'C' else 0
                if atom in atoms and atoms[atom] > floor:
                    removable = min(excess, atoms[atom] - floor)
                    atoms[atom] -= removable
                    excess -= removable

        formula_parts = []
        for atom, count in atoms.items():
            formula_parts.extend([atom] * count)
        rng.shuffle(formula_parts)
        extended_formula = ''.join(formula_parts)
        return extended_formula
    
    def _map_value(self, value, in_min, in_max, out_min, out_max):
        """Mappa un valore da un intervallo a un altro in modo lineare."""
        if in_max == in_min:
            return out_min
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# -------------------------
# Merge di grafi: copia 'source' dentro 'target'
# -------------------------
def merge_graph_into(target: MoleculeGraph, source: MoleculeGraph, attach_to_target=None, attach_from_source_idx=None):
    mapping = {}
    for old_idx in sorted(source.elements.keys()):
        el = source.elements[old_idx]
        coords = source.coords.get(old_idx, (0.0,0.0))
        new_idx = target.new_atom(el, coords=coords)
        mapping[old_idx] = new_idx
    # legami interni: non filtrati (sono già validi per definizione del sintone)
    for a,b,order in source.bonds():
        target.adj[mapping[a]][mapping[b]] = order
        target.adj[mapping[b]][mapping[a]] = order
        target.valence_rem[mapping[a]] -= order
        target.valence_rem[mapping[b]] -= order
    # legame di attacco: passa da add_bond con i controlli
    if attach_to_target is not None and attach_from_source_idx is not None:
        target.add_bond(attach_to_target, mapping[attach_from_source_idx], order=1)
    return mapping

# -------------------------
# Utility pool / parsing
# -------------------------
def parse_atoms(input_str: str):
    tokens = re.findall(r'Cl|Br|C|N|O|S|I|F', input_str)
    if not tokens:
        raise ValueError("Input string non valida o nessun atomo riconosciuto")
    for t in tokens:
        if t not in ALLOWED_ATOMS:
            raise ValueError("Atomo non ammesso: " + t)
    return Counter(tokens)

def count_graph_atoms(g: MoleculeGraph):
    return g.atom_counts()

def consume_from_pool(pool: Counter, needed: Counter):
    for el, cnt in needed.items():
        if pool[el] < cnt:
            raise ValueError("Consumazione impossibile: pool insufficiente")
        pool[el] -= cnt

# demote helper
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

# -------------------------
# Scelte "intelligenti" (core dinamico + scelta host + scelta sinton)
# -------------------------
def _family_weight_map(profile: Optional[dict]) -> dict:
    profile = profile or {}
    weights = morphology_family_weights(profile)
    # small local modulation from line-level features
    coh = profile.get("line_coherence", profile.get("spatial_coherence", 0.5))
    comp = profile.get("local_complexity", profile.get("complexity", 0.5))
    if comp > 0.62 and coh < 0.55:
        weights["mixed_polycyclic"] += 0.04
        weights["simple_heterocycle"] += 0.03
    if coh > 0.72:
        weights["simple_carbocycle"] += 0.03
        weights["aromatic_single"] += 0.03
        weights["saturated_polycyclic"] *= 0.75
    for k in list(weights):
        weights[k] = max(0.002, float(weights[k]))
    return weights


def _family_usage_penalty(family: str, profile: Optional[dict]) -> float:
    total = sum(CURRENT_FAMILY_USAGE.values())
    if total < 10:
        return 0.0
    weights = _family_weight_map(profile)
    target_share = weights.get(family, 0.05) / max(1e-6, sum(weights.values()))
    observed = CURRENT_FAMILY_USAGE.get(family, 0) / max(1, total)
    excess = max(0.0, observed - 1.15 * target_share)
    return min(0.92, 2.6 * excess)


def choose_core_by_pool(pool: Counter, profile: Optional[dict] = None):
    """Select morphology-informed scaffold family first, then a core inside it."""
    profile = profile or {}
    morph = infer_morphology_class(profile)
    weights_by_family = _family_weight_map(profile)

    available = []
    available_families = Counter()
    for core_info, meta in zip(CORE_LIBRARY, CORE_METADATA):
        if not all(pool.get(atom, 0) >= count for atom, count in core_info["requirements"].items()):
            continue
        fam = meta.get("family", "unknown")
        available.append((core_info, meta))
        available_families[fam] += 1
    if not available:
        return None, None

    family_scores = {}
    for fam, count in available_families.items():
        base = weights_by_family.get(fam, 0.01)
        rarity_bonus = 1.0 / math.sqrt(max(1, count))
        anti_collapse = 1.0 - _family_usage_penalty(fam, profile)
        family_scores[fam] = max(1e-8, base * rarity_bonus * anti_collapse)

    fam_order = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
    fam_short = [fam for fam, _ in fam_order[:max(3, min(6, len(fam_order)))]]
    sig = f"{morph}|{profile.get('artistic_bias',0.45):.3f}|{profile.get('spatial_coherence',0.5):.3f}|{profile.get('complexity',0.5):.3f}|{profile.get('line_coherence',0.5):.3f}|{profile.get('local_complexity',0.5):.3f}|{len(fam_short)}"
    fam_pick = int(_deterministic_unit(sig) * len(fam_short))
    fam_pick = min(max(fam_pick, 0), len(fam_short) - 1)
    chosen_family = fam_short[fam_pick]

    candidates = []
    for core_info, meta in available:
        family = meta.get("family", "unknown")
        if family != chosen_family:
            continue
        req = core_info["requirements"]
        carbon = req.get("C", 0)
        hetero = req.get("N", 0) + req.get("O", 0) + req.get("S", 0)
        sulfur = req.get("S", 0)
        num_rings = meta.get("num_rings", 0)
        aromatic_fraction = meta.get("aromatic_fraction", 0.0)
        score = 0.15

        if morph == "serene_compact":
            if family in {"aromatic_single", "simple_heterocycle", "simple_carbocycle"}:
                score += 0.28
            if family in {"saturated_polycyclic", "sulfurized_aliphatic", "spiro_bridged"}:
                score -= 0.35
        elif morph == "crowded_structured":
            if family in {"mixed_polycyclic", "simple_heterocycle", "aromatic_single"}:
                score += 0.22
            if family in {"saturated_polycyclic", "spiro_bridged"}:
                score -= 0.18
        elif morph == "minimal_ordered":
            if family in {"simple_carbocycle", "simple_heterocycle", "acyclic"}:
                score += 0.22
            if num_rings > 2:
                score -= 0.18
        else:
            if family in {"aromatic_single", "simple_heterocycle", "mixed_polycyclic", "simple_carbocycle"}:
                score += 0.18

        # compact medicinal-like cores preferred
        if 5 <= meta.get("num_atoms", 0) <= 11:
            score += 0.07
        if 1 <= num_rings <= 2:
            score += 0.06
        if hetero <= 2:
            score += 0.04
        if sulfur > 0:
            score -= 0.18 if morph in {"serene_compact", "minimal_ordered"} else 0.08
        if aromatic_fraction > 0.62 and family == "aromatic_fused":
            score -= 0.25
        if family == "saturated_polycyclic" and carbon >= 9:
            score -= 0.14
        if family == "mixed_polycyclic" and num_rings > 3:
            score -= 0.10
        if meta.get("spiro_atoms", 0) > 0 or any(size <= 3 for size in meta.get("ring_sizes", [])):
            score -= 1.30

        # dynamic anti-overuse for individual core
        core_use = CURRENT_CORE_USAGE.get(core_info["name"], 0)
        score -= min(0.55, 0.08 * core_use)
        score += 0.03 * _deterministic_unit(f"core|{core_info['name']}|{sig}")
        candidates.append((score, core_info, meta))

    if not candidates:
        # graceful fallback: relax morphology one step, never recurse indefinitely.
        fallback_profile = dict(profile)
        fallback_profile["artistic_bias"] = max(0.25, profile.get("artistic_bias", 0.45) - 0.08)
        fallback_profile["complexity"] = min(0.75, profile.get("complexity", 0.5) + 0.04)
        if profile is not fallback_profile:
            alt_weights = morphology_family_weights(fallback_profile)
            for fam in fam_short:
                if alt_weights.get(fam,0) > 0:
                    chosen_family = fam
                    break
            for core_info, meta in available:
                if meta.get("family","unknown") == chosen_family:
                    candidates.append((0.0, core_info, meta))
        if not candidates:
            return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    shortlist = candidates[:max(1, min(10, len(candidates)))]
    pick_index = int(_deterministic_unit(f"pick|{sig}|{chosen_family}|{len(shortlist)}") * len(shortlist))
    pick_index = min(max(pick_index, 0), len(shortlist)-1)
    _, chosen_core, chosen_meta = shortlist[pick_index]
    CURRENT_CORE_USAGE[chosen_core["name"]] += 1
    CURRENT_FAMILY_USAGE[chosen_meta.get("family", "unknown")] += 1
    return chosen_core['name'], chosen_core['factory']


def choose_host(g: MoleculeGraph, max_candidates=6):
    cand = [n for n, v in g.valence_rem.items() if v >= 1]
    if not cand:
        return None
    def _host_key(n):
        el = g.elements[n]
        deg = len(g.neighbors(n))
        crowd_pen = 2 if deg >= 3 else 0
        hetero_pen = 1 if el in {'N', 'O', 'S'} and deg >= 2 else 0
        return (crowd_pen + hetero_pen, deg, -g.valence_rem[n], 0 if el == 'C' else 1)
    cand.sort(key=_host_key)
    top = cand[:max(1, min(max_candidates, max(1, len(cand)//2)))]
    return RNG.choice(top)

def choose_sinton_or_atom(pool: Counter):
    # decide se aggiungere un sinton (gruppo) o un singolo atomo
    # probabilità guidate dalle quantità nel pool
    total = sum(pool.values())
    if total == 0:
        return None
    # prefer groups if there are O/N combos
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
        # fallback: scegli un atomo tra i più numerosi
        most = pool.most_common()
        return most[0][0]
    # default fallback
    most = pool.most_common()
    return most[0][0]

def choose_next_elem_for_chain(pool: Counter):
    # prefer elementi con valenza più alta per mantenere capacità di ramificazione
    candidates = [(VALENCES[el], el) for el, cnt in pool.items() if cnt > 0]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    chosen = candidates[0][1]
    pool[chosen] -= 1
    return chosen

# -------------------------
# Orchestratore principale
# -------------------------
def assemble_from_input_string(input_str: str, image_profile: Optional[dict] = None, line_profile: Optional[dict] = None, retry_salt: int = 0):
    """
    Assembles a molecule from an atom string using
    librerie dinamiche per la selezione di core e gruppi funzionali.
    """
    pool = parse_atoms(input_str)
    g = MoleculeGraph()
    combined_profile = blend_line_with_global_profile(line_profile or {}, image_profile)
    targets = estimate_profile_targets(combined_profile)

    # 1) SELEZIONE E INSERIMENTO DEL CORE INIZIALE
    # La logica ora utilizza la funzione dinamica `choose_core_by_pool`.
    n_cores = 1
    art = combined_profile.get("artistic_bias", 0.45)
    coh = combined_profile.get("spatial_coherence", 0.5)
    comp = combined_profile.get("complexity", 0.5)
    morph = infer_morphology_class(combined_profile)
    if sum(pool.values()) >= 24 and morph == "crowded_structured" and comp > 0.68 and art < 0.48:
        n_cores = 2 if coh < 0.48 else 1

    atoms_in_first_core = set()

    for i in range(n_cores):
        core_name, core_factory = choose_core_by_pool(pool, combined_profile)
        if core_factory is None:
            continue
        
        # Recupera le informazioni complete del core dalla libreria
        try:
            core_info = next(item for item in CORE_LIBRARY if item["name"] == core_name)
            needed_atoms = core_info["requirements"]
        except StopIteration:
            # Fallback di sicurezza se il nome non viene trovato, anche se non dovrebbe accadere
            print(f"Warning: core '{core_name}' was not found in the library.", file=sys.stderr)
            continue
            
        core_graph = core_factory()
        host_for_second_core = None

        # Logica per attaccare il SECONDO anello
        if g.elements and i > 0: # Ensure this is the second or subsequent core
            possible_hosts = [idx for idx in g.elements if idx not in atoms_in_first_core and g.valence_rem[idx] >= 1]
            
            if possible_hosts:
                host_for_second_core = RNG.choice(possible_hosts)
            else:
                # Se non ci sono host, prova a creare un linker di Carbonio
                if pool.get('C', 0) > 0:
                    needed_atoms['C'] = needed_atoms.get('C', 0) + 1 # Aggiungi il costo del linker
                else:
                    continue # Unable to add the second ring

        # Controlla se il pool può soddisfare il fabbisogno (già fatto in choose_core, ma è una doppia sicurezza)
        if not all(pool.get(el, 0) >= count for el, count in needed_atoms.items()):
            continue

        # Esegui le modifiche: consuma atomi e aggiungi il core
        consume_from_pool(pool, needed_atoms)

        if not g.elements:
            # Primo core
            mapping = merge_graph_into(g, core_graph)
            atoms_in_first_core = set(mapping.values())
        else:
            # Secondo core (o successivi)
            attach_from_source_idx = RNG.choice(list(core_graph.elements.keys()))
            if host_for_second_core:
                merge_graph_into(g, core_graph, attach_to_target=host_for_second_core, attach_from_source_idx=attach_from_source_idx)
            else:
                # Crea il linker che abbiamo "pagato"
                linker_attach_point = RNG.choice(list(atoms_in_first_core)) if atoms_in_first_core else RNG.choice(list(g.elements.keys()))

                if g.valence_rem.get(linker_attach_point, 0) >= 1:
                    linker_atom_idx = g.new_atom("C")
                    g.add_bond(linker_attach_point, linker_atom_idx, order=1)
                    merge_graph_into(g, core_graph, attach_to_target=linker_atom_idx, attach_from_source_idx=attach_from_source_idx)

    # Fallback se nessun core è stato inserito
    if not g.elements:
        if not pool:
            raise ValueError("Input string non valida, nessun atomo per iniziare.")
        # Scegli l'elemento più abbondante per iniziare
        most_common_atom = pool.most_common(1)[0][0]
        pool[most_common_atom] -= 1
        g.new_atom(most_common_atom)

    # 2) CICLO PRINCIPALE DI ASSEMBLAGGIO
    # Add functional groups or individual atoms until the pool is empty.
    while sum(pool.values()) > 0:
        # --- NEW FUNCTIONAL GROUP SELECTION LOGIC ---
        possible_group_moves = []
        host_candidates = [n for n, v in g.valence_rem.items() if v >= 1]
        host_candidates.sort(key=lambda n: (len(g.neighbors(n)), -g.valence_rem[n]))
        host_candidates = host_candidates[:max(2, min(8, len(host_candidates)))]

        for group_info in GROUP_LIBRARY:
            if not all(pool.get(el, 0) >= cnt for el, cnt in group_info["requirements"].items()):
                continue
            for host in host_candidates:
                move_score = property_aware_group_score(group_info, host, g, combined_profile, pool)
                possible_group_moves.append((move_score, group_info, host))

        group_added = False
        if possible_group_moves:
            possible_group_moves.sort(key=lambda x: x[0], reverse=True)
            shortlist = possible_group_moves[:max(1, min(6, len(possible_group_moves)))]
            chosen_score, chosen_group, chosen_host = shortlist[0]
            if len(shortlist) > 1:
                pick = int(_deterministic_unit(f"group|{retry_salt}|{chosen_group['name']}|{chosen_host}|{sum(pool.values())}|{combined_profile.get('artistic_bias',0.45):.3f}") * min(3, len(shortlist)))
                chosen_score, chosen_group, chosen_host = shortlist[min(pick, len(shortlist)-1)]
            if chosen_score > -2.25:
                factory = chosen_group['factory']
                need = chosen_group['requirements']
                consume_from_pool(pool, need)
                merge_graph_into(g, factory(), attach_to_target=chosen_host, attach_from_source_idx=0)
                group_added = True

        if group_added:
            continue

        # --- LOGICA DI FALLBACK: AGGIUNTA DI SINGOLI ATOMI PROPERTY-AWARE ---
        moves = property_aware_atom_moves(g, pool, combined_profile)
        if not moves:
            # small rescue: free capacity and retry once
            demote_some_double_bonds_until_capacity(g, 2)
            moves = property_aware_atom_moves(g, pool, combined_profile)
        if not moves:
            break

        top_moves = moves[:max(1, min(8, len(moves)))]
        mv_pick = int(_deterministic_unit(f"atom|{retry_salt}|{sum(pool.values())}|{combined_profile.get('complexity',0.5):.3f}|{len(top_moves)}") * min(3, len(top_moves)))
        _, elem_to_add, host = top_moves[min(mv_pick, len(top_moves)-1)]
        pool[elem_to_add] -= 1
        new_idx = g.new_atom(elem_to_add)
        connected = g.add_bond(host, new_idx, order=1)

        if not connected:
            g.elements.pop(new_idx, None)
            g.coords.pop(new_idx, None)
            g.valence_max.pop(new_idx, None)
            g.valence_rem.pop(new_idx, None)
            g.adj.pop(new_idx, None)
            pool[elem_to_add] += 1
            break

    # 3) CONNESSIONE FINALE DEI FRAMMENTI
    # La logica per connettere componenti disgiunte rimane invariata.
    def components(graph: MoleculeGraph):
        # ... (codice della funzione components invariato) ...
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
            print("Warning: unable to connect all molecular fragments.", file=sys.stderr)
            break
            
        if g.add_bond(node_a, node_b, order=1):
            comps = components(g)
        
        attempts += 1


    # 4) CONVERSIONE IN RDKIT E SANITIZZAZIONE
    # La logica di conversione e sanitizzazione finale rimane invariata.
    rwmol = graph_to_rwmol(g)
    mol = rwmol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Fallback: demota tutti i doppi legami e riprova
        for a, b, order in list(g.bonds()):
            while g.adj[a][b] >= 2:
                g.demote_bond(a, b)
        rwmol = graph_to_rwmol(g)
        mol = rwmol.GetMol()
        Chem.SanitizeMol(mol) # Se fallisce di nuovo, solleverà un'eccezione

    AllChem.Compute2DCoords(mol)
    return mol, g

# -------------------------
# Conversione grafo -> RDKit RWMol
# -------------------------
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
        # protezione se mappa mancante
        if a in idx_map and b in idx_map:
            rwmol.AddBond(idx_map[a], idx_map[b], bt)
    return rwmol

CORE_METADATA = _build_core_metadata()
CORE_METADATA_BY_NAME = {m["name"]: m for m in CORE_METADATA}


#-----------------------------------------------------------------Refine SMILES
def refine_smiles(smiles: str) -> str:
    """
    Identifica i frammenti in una stringa SMILES separati da punti ('.').
    Returns the fragment with the largest number of characters (tokens).

    Args:
        smiles: La stringa SMILES di input, che può contenere uno o più frammenti.

    Returns:
        La sottostringa del frammento più lungo. Se non ci sono punti,
        restituisce la stringa originale.
    """
    # 1. Divide la stringa SMILES in una lista di frammenti usando il punto come separatore.
    frammenti = smiles.split('.')

    # 2. Se c'è un solo frammento (nessun punto), restituisce la stringa così com'è.
    if len(frammenti) == 1:
        return smiles

    # 3. Trova il frammento con la lunghezza massima.
    #    La funzione max() con una chiave `key=len` itera sulla lista e
    #    determina quale elemento ha il valore più alto quando gli viene applicata
    #    la funzione len().
    frammento_piu_grande = max(frammenti, key=len)

    return frammento_piu_grande

#-----------------------------------------------------------------
# NEW FUNCTION: Calcolo Drug Likeness
#-----------------------------------------------------------------
def calculate_qed_score(smiles: str) -> float:
    """
    Computes the Quantitative Estimate of Drug-likeness (QED) for a molecule.

    The original score (0-1) is scaled to 0-100 for compatibility
    with the graphical interface.

    Args:
        smiles: The molecule SMILES string.

    Returns:
        Il punteggio QED scalato come percentuale (da 0.0 a 100.0).
        Returns 0.0 if the SMILES is invalid or an error occurs.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 0.0  # SMILES non valido, punteggio 0

    try:
        # Compute QED, which returns a value between 0 and 1
        qed_value = QED.qed(mol)
        # Scala il valore a 100 per coerenza con la progress bar
        return qed_value * 100.0
    except Exception:
        # In caso di errori nel calcolo, restituisce 0
        return 0.0
    
#-----------------------------------------------------------------
# NEW FUNCTION: Ricerca del CID su PubChem (troppo tempo, abbandonata)
#-----------------------------------------------------------------
def get_pubchem_cid(smiles: str) -> int | None:
    """
    Verifica se un composto esiste in PubChem tramite una ricerca di identità
    basata sulla stringa SMILES e restituisce il suo CID (Compound ID).

    Args:
        smiles: The molecule SMILES string to search.

    Returns:
        The CID integer if the molecule is found; otherwise None.
    """
    if pcp is None:
        return None
    try:
        # Run an identity search. It is faster and more precise for this purpose.
        results = pcp.get_compounds(smiles, 'smiles')
        
        # Se la ricerca produce risultati, restituisce il CID del primo composto.
        if results:
            return results[0].cid
        
        # Nessun risultato trovato
        return None
        
    except pcp.PubChemHTTPError as e:
        # Handle common errors such as "PUGREST.NotFound" for missing molecules
        # o problemi di connessione.
        if "PUGREST.NotFound" in str(e):
            return None # The molecule was not found, which is the expected outcome.
        else:
            print(f"A PubChem connection error occurred: {e}")
            return None # Tratta altri errori di rete come "non trovato" per semplicità.
    except Exception as e:
        print(f"Unexpected error during PubChem search for SMILES '{smiles}': {e}")
        return None
    
# Processing entry point retained for GUI integration
# e rimuovi il blocco 'if __name__ == "__main__":' originale.

def run_imagichem_processing(image_path, progress_callback):
    """
    Runs the complete image analysis and molecule generation process.

    Args:
        image_path (str): The input image file path.
        progress_callback (function): Una funzione (o un segnale PyQt) da chiamare
                                      per riportare il progresso (da 0 a 100).

    Returns:
        list: Una lista di tuple, dove ogni tupla contiene (smiles, score).
              Returns an empty list in case of input error.
    """
    try:
        image_seed = image_to_seed(image_path)
        set_seed(image_seed)
        global_profile = compute_artistic_profile(image_path)
        CURRENT_IMAGE_PROFILE.clear()
        CURRENT_IMAGE_PROFILE.update(global_profile)
        CURRENT_CORE_USAGE.clear()
        CURRENT_FAMILY_USAGE.clear()
        CURRENT_GROUP_USAGE.clear()
        CURRENT_SCAFFOLD_USAGE.clear()
        CURRENT_SIGNATURE_USAGE.clear()
        CURRENT_MOTIF_USAGE.clear()
        CURRENT_FINAL_SMILES.clear()
        pixel_rows = split_pixel_rows(image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Critical error while reading the image: {e}")
        return []

    if not pixel_rows:
        return []

    lista_formule = []
    for i, pixel_row in enumerate(pixel_rows):
        try:
            analyzer = PixelLineAnalyzer(pixel_row)
            line_features = analyzer.analyze_pixel_pattern()
            # Adds halogens only rarely and only for harmonious profiles.
            can_add_halogens = (i % 37 == 0 and global_profile.get("artistic_bias", 0.0) > 0.60)
            formula_string = analyzer.generate_molecular_formula_string(RNG, allow_halogens=can_add_halogens, global_profile=global_profile)
            lista_formule.append((formula_string, line_features))
        except ValueError:
            lista_formule.append(None)
    
    lista_smiles = []
    total_formulas = len(lista_formule)
    for i, formula_item in enumerate(lista_formule):
        if formula_item:
            try:
                formula, line_features = formula_item
                accepted_smiles = None
                accepted_mol = None
                duplicate_fallback = None
                duplicate_fallback_mol = None
                for retry_idx in range(3):
                    try:
                        mol, graph = assemble_from_input_string(formula, image_profile=global_profile, line_profile=line_features, retry_salt=retry_idx)
                        smiles = Chem.MolToSmiles(mol, canonical=True)
                        refined = refine_smiles(smiles)
                        refined_mol = Chem.MolFromSmiles(refined)
                        if refined_mol is None:
                            continue
                        if CURRENT_FINAL_SMILES.get(refined, 0) > 0:
                            if duplicate_fallback is None:
                                duplicate_fallback = refined
                                duplicate_fallback_mol = refined_mol
                            continue
                        accepted_smiles = refined
                        accepted_mol = refined_mol
                        break
                    except Exception:
                        continue
                if accepted_smiles is None and duplicate_fallback is not None:
                    accepted_smiles = duplicate_fallback
                    accepted_mol = duplicate_fallback_mol
                if accepted_smiles is not None:
                    _register_molecule_signature(accepted_mol, chosen_group=None)
                    _register_final_smiles(accepted_smiles)
                    lista_smiles.append(accepted_smiles)
            except Exception:
                # Ignore assembly errors for a specific formula
                pass
        
        # Update progress
        progress = int((i + 1) / total_formulas * 100)
        progress_callback(progress)

    # Clean SMILES and compute the drug-likeness score
    results_with_scores = []
    for smiles in lista_smiles:
        if smiles:
            frammento_principale = smiles
            # --- UPDATED SECTION ---
            # Chiama la nuova funzione per il calcolo del QED
            score = calculate_qed_score(frammento_principale)
            results_with_scores.append((frammento_principale, score))

    # Sort results from highest to lowest score
    results_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    return results_with_scores

# -------------------------
# Integrated generation router: library / fromscratch / hybrid
# -------------------------
try:
    from fromscratch_backend import generate_from_image_profile as _generate_fromscratch_image_profile
except Exception:
    _generate_fromscratch_image_profile = None

_run_imagichem_processing_library = run_imagichem_processing


def _merge_unique_results(*result_lists):
    merged = []
    seen = set()
    for results in result_lists:
        for smiles, score in results:
            if smiles in seen:
                continue
            seen.add(smiles)
            merged.append((smiles, score))
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


def _prepare_image_profile_and_target(image_path):
    image_seed = image_to_seed(image_path)
    set_seed(image_seed)
    profile = compute_artistic_profile(image_path)
    pixel_rows = split_pixel_rows(image_path)
    target_n = max(60, min(len(pixel_rows), 800)) if pixel_rows else 120
    return image_seed, profile, pixel_rows, target_n


def _run_fromscratch_processing(image_path, progress_callback):
    if _generate_fromscratch_image_profile is None:
        raise RuntimeError('The from-scratch backend is not available')
    try:
        image_seed, profile, pixel_rows, target_n = _prepare_image_profile_and_target(image_path)
    except (FileNotFoundError, ValueError):
        return []
    # In from-scratch mode, the target count must truly depend on the image,
    # come avviene nel backend library-based. Evitiamo quindi il vecchio cap fisso a 500.
    n = target_n
    return _generate_fromscratch_image_profile(
        profile,
        image_seed,
        n,
        progress_callback=progress_callback,
        preset='balanced',
        max_attempts=max(12000, n * 180),
    )


def run_imagichem_processing(image_path, progress_callback, generation_mode='library'):
    mode = (generation_mode or 'library').lower()
    if mode == 'library':
        return _run_imagichem_processing_library(image_path, progress_callback)
    if mode == 'fromscratch':
        return _run_fromscratch_processing(image_path, progress_callback)
    if mode == 'hybrid':
        # Phase 1: library backend retained as-is.
        lib_results = _run_imagichem_processing_library(image_path, lambda p: progress_callback(min(55, int(p * 0.55))))
        # Phase 2: image-conditioned from-scratch backend guided by the same global image profile.
        if _generate_fromscratch_image_profile is None:
            progress_callback(100)
            return lib_results
        try:
            image_seed, profile, pixel_rows, target_n = _prepare_image_profile_and_target(image_path)
        except (FileNotFoundError, ValueError):
            return lib_results
        # In hybrid manteniamo un batch from-scratch più leggero del library-based,
        # while still depending on image size.
        fs_n = max(120, min(target_n // 2, 1200))
        fs_results = _generate_fromscratch_image_profile(
            profile,
            image_seed ^ 0x5A5A1357,
            fs_n,
            progress_callback=lambda p: progress_callback(55 + int(p * 0.45)),
            preset='balanced',
            max_attempts=max(12000, fs_n * 180),
        )
        merged = _merge_unique_results(fs_results, lib_results)
        progress_callback(100)
        return merged
    return _run_imagichem_processing_library(image_path, progress_callback)
