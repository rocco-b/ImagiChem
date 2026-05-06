#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based from-scratch molecular generator v2.2.

Main upgrades vs v2.1:
- more natural medicinal-chemistry-like connections between ring systems
- less direct ring-on-ring "builder" attachment bias
- more ring-linker-ring motifs (aryl-CH2-heterocycle, aryl-O-heterocycle, aryl-CH2-aryl)
- reduced reliance on heavily saturated fused-like constructions
- still keeps low amide/carbamate dependence and useful topological complexity
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem.rdchem import BondType


@dataclass
class Range:
    min: float
    max: float

    def contains(self, value: float) -> bool:
        return self.min <= value <= self.max

    def score(self, value: float, optimum: Optional[Tuple[float, float]] = None) -> float:
        if not self.contains(value):
            if value < self.min:
                span = max(1e-6, self.min)
                return max(0.0, 1.0 - (self.min - value) / span)
            span = max(1e-6, self.max)
            return max(0.0, 1.0 - (value - self.max) / span)
        if optimum is None:
            return 1.0
        lo, hi = optimum
        if lo <= value <= hi:
            return 1.0
        if value < lo:
            return max(0.0, 1.0 - (lo - value) / max(1e-6, lo - self.min + 1e-6))
        return max(0.0, 1.0 - (value - hi) / max(1e-6, self.max - hi + 1e-6))


@dataclass
class GeneratorConstraints:
    molecular_weight: Range
    clogp: Range
    clogs: Range
    tpsa: Range
    tsa: Range
    hbd: Range
    hba: Range
    carbon: Range
    oxygen: Range
    nitrogen: Range
    aromatic_nitrogen: Range
    sulfur: Range
    fluorine: Range
    chlorine: Range
    bromine: Range
    iodine: Range
    bonds: Range
    single_bonds: Range
    double_bonds: Range
    triple_bonds: Range
    aromatic_bonds: Range
    total_atoms: Range
    rings: Range
    aromatic_rings: Range
    amides: Range
    amines: Range
    optimum_mw: Tuple[float, float] = (250.0, 400.0)
    optimum_total_atoms: Tuple[float, float] = (20.0, 35.0)
    optimum_rings: Tuple[float, float] = (2.0, 3.0)
    optimum_aromatic_rings: Tuple[float, float] = (1.0, 2.0)


BALANCED_CONSTRAINTS = GeneratorConstraints(
    molecular_weight=Range(150, 600),
    clogp=Range(-5, 5),
    clogs=Range(-9, 5),
    tpsa=Range(0, 335),
    tsa=Range(80, 550),
    hbd=Range(0, 5),
    hba=Range(0, 10),
    carbon=Range(8, 40),
    oxygen=Range(1, 10),
    nitrogen=Range(1, 5),
    aromatic_nitrogen=Range(0, 4),
    sulfur=Range(0, 2),
    fluorine=Range(0, 2),
    chlorine=Range(0, 2),
    bromine=Range(0, 2),
    iodine=Range(0, 1),
    bonds=Range(4, 30),
    single_bonds=Range(0, 29),
    double_bonds=Range(0, 6),
    triple_bonds=Range(0, 2),
    aromatic_bonds=Range(0, 14),
    total_atoms=Range(10, 40),
    rings=Range(1, 4),
    aromatic_rings=Range(0, 3),
    amides=Range(0, 3),
    amines=Range(0, 5),
)

STRICT_USER_CONSTRAINTS = GeneratorConstraints(
    molecular_weight=Range(150, 600),
    clogp=Range(-5, 5),
    clogs=Range(-9, 5),
    tpsa=Range(0, 335),
    tsa=Range(80, 550),
    hbd=Range(0, 5),
    hba=Range(0, 10),
    carbon=Range(1, 40),
    oxygen=Range(1, 10),
    nitrogen=Range(1, 4),
    aromatic_nitrogen=Range(0, 4),
    sulfur=Range(1, 2),
    fluorine=Range(1, 2),
    chlorine=Range(1, 2),
    bromine=Range(1, 2),
    iodine=Range(1, 2),
    bonds=Range(4, 30),
    single_bonds=Range(0, 29),
    double_bonds=Range(0, 5),
    triple_bonds=Range(0, 2),
    aromatic_bonds=Range(0, 10),
    total_atoms=Range(5, 30),
    rings=Range(0, 3),
    aromatic_rings=Range(0, 3),
    amides=Range(0, 4),
    amines=Range(0, 4),
)

PRESETS = {"balanced": BALANCED_CONSTRAINTS, "strict_user": STRICT_USER_CONSTRAINTS}

ATOM_NUM = {"C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17, "Br": 35, "I": 53}
HALOGENS = ["F", "Cl", "Br", "I"]

AMIDE_SMARTS = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")
CARBAMATE_SMARTS = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[OX2][#6]")
AMINE_SMARTS = Chem.MolFromSmarts("[NX3;!$(N-C=O);!a]")
ETHER_SMARTS = Chem.MolFromSmarts("[OD2]([#6])[#6]")


class MolBuilder:
    def __init__(self) -> None:
        self.mol = Chem.RWMol()
        self.attachment_sites: List[int] = []
        self.ring_anchor_sites: List[int] = []
        self.rings_meta: List[Dict[str, object]] = []

    def add_atom(self, symbol: str, aromatic: bool = False, formal_charge: int = 0) -> int:
        atom = Chem.Atom(ATOM_NUM[symbol])
        atom.SetFormalCharge(formal_charge)
        atom.SetIsAromatic(aromatic)
        return self.mol.AddAtom(atom)

    def add_bond(self, a: int, b: int, bond_type: BondType) -> None:
        if self.mol.GetBondBetweenAtoms(a, b) is None:
            self.mol.AddBond(a, b, bond_type)
            bond = self.mol.GetBondBetweenAtoms(a, b)
            if bond_type == BondType.AROMATIC:
                bond.SetIsAromatic(True)

    def _register_sites(self, ring_atoms: List[int], aromatic: bool, ring_name: str) -> None:
        ring_carbons = []
        for idx in ring_atoms:
            atom = self.mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() in {"C", "N"}:
                self.attachment_sites.append(idx)
                self.ring_anchor_sites.append(idx)
                if atom.GetSymbol() == "C":
                    ring_carbons.append(idx)
        self.rings_meta.append({"name": ring_name, "atoms": list(ring_atoms), "aromatic": aromatic, "carbons": ring_carbons})

    def add_ring(self, ring_type: str, attach_to: Optional[int] = None) -> List[int]:
        ring_atoms: List[int] = []
        aromatic = False
        if ring_type == "benzene":
            aromatic = True
            for _ in range(6):
                ring_atoms.append(self.add_atom("C", aromatic=True))
            for i in range(6):
                self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 6], BondType.AROMATIC)
        elif ring_type == "pyridine":
            aromatic = True
            for i in range(6):
                ring_atoms.append(self.add_atom("N" if i == 0 else "C", aromatic=True))
            for i in range(6):
                self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 6], BondType.AROMATIC)
        elif ring_type == "pyrimidine":
            aromatic = True
            for i in range(6):
                ring_atoms.append(self.add_atom("N" if i in (0, 2) else "C", aromatic=True))
            for i in range(6):
                self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 6], BondType.AROMATIC)
        elif ring_type == "pyrazine":
            aromatic = True
            for i in range(6):
                ring_atoms.append(self.add_atom("N" if i in (0, 3) else "C", aromatic=True))
            for i in range(6):
                self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 6], BondType.AROMATIC)
        elif ring_type == "imidazole":
            aromatic = True
            syms = ["N", "C", "N", "C", "C"]
            for s in syms:
                ring_atoms.append(self.add_atom(s, aromatic=True))
            for i in range(5):
                self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 5], BondType.AROMATIC)
        elif ring_type == "oxazole":
            aromatic = True
            syms = ["O", "C", "N", "C", "C"]
            for s in syms:
                ring_atoms.append(self.add_atom(s, aromatic=True))
            for i in range(5):
                self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 5], BondType.AROMATIC)
        elif ring_type == "thiophene":
            aromatic = True
            for i in range(5):
                ring_atoms.append(self.add_atom("S" if i == 0 else "C", aromatic=True))
            for i in range(5):
                self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 5], BondType.AROMATIC)
        elif ring_type == "cyclohexane":
            for _ in range(6): ring_atoms.append(self.add_atom("C"))
            for i in range(6): self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 6], BondType.SINGLE)
        elif ring_type == "piperidine":
            for i in range(6): ring_atoms.append(self.add_atom("N" if i == 0 else "C"))
            for i in range(6): self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 6], BondType.SINGLE)
        elif ring_type == "morpholine":
            for i in range(6):
                sym = "N" if i == 0 else ("O" if i == 3 else "C")
                ring_atoms.append(self.add_atom(sym))
            for i in range(6): self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 6], BondType.SINGLE)
        elif ring_type == "piperazine":
            for i in range(6):
                sym = "N" if i in (0, 3) else "C"
                ring_atoms.append(self.add_atom(sym))
            for i in range(6): self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 6], BondType.SINGLE)
        elif ring_type == "cyclopentane":
            for _ in range(5): ring_atoms.append(self.add_atom("C"))
            for i in range(5): self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 5], BondType.SINGLE)
        elif ring_type == "pyrrolidine":
            for i in range(5): ring_atoms.append(self.add_atom("N" if i == 0 else "C"))
            for i in range(5): self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 5], BondType.SINGLE)
        elif ring_type == "tetrahydrofuran":
            for i in range(5): ring_atoms.append(self.add_atom("O" if i == 0 else "C"))
            for i in range(5): self.add_bond(ring_atoms[i], ring_atoms[(i + 1) % 5], BondType.SINGLE)
        else:
            raise ValueError(ring_type)

        if attach_to is not None:
            target = random.choice(ring_atoms)
            self.add_bond(attach_to, target, BondType.SINGLE)
        self._register_sites(ring_atoms, aromatic=aromatic, ring_name=ring_type)
        return ring_atoms

    def pop_site(self, rng: random.Random, from_ring_only: bool = False, carbon_only: bool = False, aromatic_only: bool = False) -> Optional[int]:
        sites = list(self.ring_anchor_sites if from_ring_only and self.ring_anchor_sites else self.attachment_sites)
        if carbon_only:
            sites = [i for i in sites if self.mol.GetAtomWithIdx(i).GetSymbol() == "C"]
        if aromatic_only:
            sites = [i for i in sites if self.mol.GetAtomWithIdx(i).GetIsAromatic()]
        if not sites:
            return None
        site = rng.choice(sites)
        if site in self.attachment_sites:
            self.attachment_sites.remove(site)
        if site in self.ring_anchor_sites:
            self.ring_anchor_sites.remove(site)
        return site

    def add_chain(self, attach_to: int, atoms: Sequence[str]) -> int:
        prev = attach_to
        first_new = None
        for symbol in atoms:
            idx = self.add_atom(symbol)
            if first_new is None:
                first_new = idx
            self.add_bond(prev, idx, BondType.SINGLE)
            if symbol in {"C", "N", "O", "S"}:
                self.attachment_sites.append(idx)
            prev = idx
        return prev if first_new is not None else attach_to

    def attach_halogen(self, attach_to: int, symbol: str) -> None:
        idx = self.add_atom(symbol)
        self.add_bond(attach_to, idx, BondType.SINGLE)

    def attach_ether(self, attach_to: int) -> None:
        o_idx = self.add_atom("O")
        c1 = self.add_atom("C")
        self.add_bond(attach_to, o_idx, BondType.SINGLE)
        self.add_bond(o_idx, c1, BondType.SINGLE)
        self.attachment_sites.append(c1)

    def attach_methoxy(self, attach_to: int) -> None:
        o_idx = self.add_atom("O")
        c1 = self.add_atom("C")
        self.add_bond(attach_to, o_idx, BondType.SINGLE)
        self.add_bond(o_idx, c1, BondType.SINGLE)

    def attach_hydroxyethyl(self, attach_to: int) -> None:
        c1 = self.add_atom("C")
        c2 = self.add_atom("C")
        o = self.add_atom("O")
        self.add_bond(attach_to, c1, BondType.SINGLE)
        self.add_bond(c1, c2, BondType.SINGLE)
        self.add_bond(c2, o, BondType.SINGLE)
        self.attachment_sites.append(c1)
        self.attachment_sites.append(c2)

    def attach_alcohol(self, attach_to: int) -> None:
        o_idx = self.add_atom("O")
        self.add_bond(attach_to, o_idx, BondType.SINGLE)

    def attach_nitrile(self, attach_to: int) -> None:
        c_idx = self.add_atom("C")
        n_idx = self.add_atom("N")
        self.add_bond(attach_to, c_idx, BondType.SINGLE)
        self.add_bond(c_idx, n_idx, BondType.TRIPLE)

    def attach_amine(self, attach_to: int, secondary_bias: bool = True) -> None:
        n_idx = self.add_atom("N")
        self.add_bond(attach_to, n_idx, BondType.SINGLE)
        self.attachment_sites.append(n_idx)
        if secondary_bias:
            c1 = self.add_atom("C")
            self.add_bond(n_idx, c1, BondType.SINGLE)
            self.attachment_sites.append(c1)

    def attach_dimethylamine_like(self, attach_to: int) -> None:
        n = self.add_atom("N")
        c1 = self.add_atom("C")
        c2 = self.add_atom("C")
        self.add_bond(attach_to, n, BondType.SINGLE)
        self.add_bond(n, c1, BondType.SINGLE)
        self.add_bond(n, c2, BondType.SINGLE)

    def attach_amide(self, attach_to: int, n_substituted: bool = False) -> None:
        carbonyl_c = self.add_atom("C")
        carbonyl_o = self.add_atom("O")
        amide_n = self.add_atom("N")
        self.add_bond(attach_to, carbonyl_c, BondType.SINGLE)
        self.add_bond(carbonyl_c, carbonyl_o, BondType.DOUBLE)
        self.add_bond(carbonyl_c, amide_n, BondType.SINGLE)
        if n_substituted:
            sub = self.add_atom("C")
            self.add_bond(amide_n, sub, BondType.SINGLE)
            self.attachment_sites.append(sub)

    def attach_ester(self, attach_to: int) -> None:
        carbonyl_c = self.add_atom("C")
        carbonyl_o = self.add_atom("O")
        single_o = self.add_atom("O")
        alkyl = self.add_atom("C")
        self.add_bond(attach_to, carbonyl_c, BondType.SINGLE)
        self.add_bond(carbonyl_c, carbonyl_o, BondType.DOUBLE)
        self.add_bond(carbonyl_c, single_o, BondType.SINGLE)
        self.add_bond(single_o, alkyl, BondType.SINGLE)
        self.attachment_sites.append(alkyl)

    def attach_ketone(self, attach_to: int) -> None:
        c = self.add_atom("C")
        o = self.add_atom("O")
        alk = self.add_atom("C")
        self.add_bond(attach_to, c, BondType.SINGLE)
        self.add_bond(c, o, BondType.DOUBLE)
        self.add_bond(c, alk, BondType.SINGLE)
        self.attachment_sites.append(alk)

    def attach_thioether(self, attach_to: int) -> None:
        s_idx = self.add_atom("S")
        c_idx = self.add_atom("C")
        self.add_bond(attach_to, s_idx, BondType.SINGLE)
        self.add_bond(s_idx, c_idx, BondType.SINGLE)
        self.attachment_sites.append(c_idx)

    def annulate_on_ring(self, meta_idx: int, mode: str = "carbocycle") -> bool:
        meta = self.rings_meta[meta_idx]
        carbons = list(meta["carbons"])
        if len(carbons) < 2:
            return False
        a, b = carbons[0], carbons[1]
        if self.mol.GetBondBetweenAtoms(a, b) is None:
            return False
        if mode == "carbocycle":
            x1 = self.add_atom("C"); x2 = self.add_atom("C"); x3 = self.add_atom("C")
            self.add_bond(a, x1, BondType.SINGLE); self.add_bond(x1, x2, BondType.SINGLE); self.add_bond(x2, x3, BondType.SINGLE); self.add_bond(x3, b, BondType.SINGLE)
            self._register_sites([x1, x2, x3], aromatic=False, ring_name="annulated_carbocycle")
        elif mode == "heterocycle":
            x1 = self.add_atom("N"); x2 = self.add_atom("C"); x3 = self.add_atom("O")
            self.add_bond(a, x1, BondType.SINGLE); self.add_bond(x1, x2, BondType.SINGLE); self.add_bond(x2, x3, BondType.SINGLE); self.add_bond(x3, b, BondType.SINGLE)
            self._register_sites([x1, x2, x3], aromatic=False, ring_name="annulated_heterocycle")
        else:
            return False
        return True

    def add_spiro_carbocycle(self, center_idx: int) -> bool:
        center = self.mol.GetAtomWithIdx(center_idx)
        if center.GetSymbol() != "C":
            return False
        x1 = self.add_atom("C"); x2 = self.add_atom("C"); x3 = self.add_atom("C")
        self.add_bond(center_idx, x1, BondType.SINGLE); self.add_bond(x1, x2, BondType.SINGLE); self.add_bond(x2, x3, BondType.SINGLE); self.add_bond(x3, center_idx, BondType.SINGLE)
        self._register_sites([x1, x2, x3], aromatic=False, ring_name="spiro_cyclobutane")
        return True

    def attach_ring_via_linker(self, attach_to: int, ring_type: str, linker_mode: str) -> None:
        if linker_mode == "direct":
            self.add_ring(ring_type, attach_to=attach_to)
            return
        if linker_mode == "methylene":
            pivot = self.add_atom("C")
            self.add_bond(attach_to, pivot, BondType.SINGLE)
        elif linker_mode == "ether":
            pivot = self.add_atom("O")
            self.add_bond(attach_to, pivot, BondType.SINGLE)
        elif linker_mode == "amine":
            pivot = self.add_atom("N")
            self.add_bond(attach_to, pivot, BondType.SINGLE)
        elif linker_mode == "ethyl":
            c1 = self.add_atom("C")
            c2 = self.add_atom("C")
            self.add_bond(attach_to, c1, BondType.SINGLE)
            self.add_bond(c1, c2, BondType.SINGLE)
            pivot = c2
        elif linker_mode == "oxyethyl":
            o = self.add_atom("O")
            c = self.add_atom("C")
            self.add_bond(attach_to, o, BondType.SINGLE)
            self.add_bond(o, c, BondType.SINGLE)
            pivot = c
        else:
            pivot = attach_to
        self.add_ring(ring_type, attach_to=pivot)

    def get_mol(self) -> Chem.Mol:
        mol = self.mol.GetMol()
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        Chem.SanitizeMol(mol)
        return mol


def estimate_esol_logS(mol: Chem.Mol) -> float:
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rotors = Lipinski.NumRotatableBonds(mol)
    heavy = max(1, mol.GetNumHeavyAtoms())
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    aromatic_proportion = aromatic_atoms / heavy
    return 0.16 - 1.5 * logp - 0.0062 * mw + 0.066 * rotors + 0.066 * aromatic_proportion


def count_amides(mol: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(AMIDE_SMARTS, uniquify=True))


def count_carbamates(mol: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(CARBAMATE_SMARTS, uniquify=True))


def count_amines(mol: Chem.Mol) -> int:
    atoms = {m[0] for m in mol.GetSubstructMatches(AMINE_SMARTS, uniquify=True)}
    return len(atoms)


def count_ethers(mol: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(ETHER_SMARTS, uniquify=True))


def count_esters(mol: Chem.Mol) -> int:
    patt = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][#6]")
    return len(mol.GetSubstructMatches(patt, uniquify=True))


def aromatic_ring_count(mol: Chem.Mol) -> int:
    ring_info = mol.GetRingInfo()
    count = 0
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            count += 1
    return count


def atom_counts(mol: Chem.Mol) -> Dict[str, int]:
    counts = {k: 0 for k in ATOM_NUM}
    aromatic_n = 0
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
        if sym == "N" and atom.GetIsAromatic():
            aromatic_n += 1
    counts["aromatic_N"] = aromatic_n
    return counts


def bond_counts(mol: Chem.Mol) -> Dict[str, int]:
    out = {"total": 0, "single": 0, "double": 0, "triple": 0, "aromatic": 0}
    for bond in mol.GetBonds():
        out["total"] += 1
        bt = bond.GetBondType()
        if bt == BondType.SINGLE:
            out["single"] += 1
        elif bt == BondType.DOUBLE:
            out["double"] += 1
        elif bt == BondType.TRIPLE:
            out["triple"] += 1
        elif bt == BondType.AROMATIC:
            out["aromatic"] += 1
    return out


def descriptors(mol: Chem.Mol) -> Dict[str, float]:
    counts = atom_counts(mol)
    bonds = bond_counts(mol)
    from rdkit.Chem import QED
    return {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "mw": Descriptors.MolWt(mol),
        "clogp": Crippen.MolLogP(mol),
        "clogs": estimate_esol_logS(mol),
        "tpsa": rdmd.CalcTPSA(mol),
        "tsa": rdmd.CalcLabuteASA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "carbon": counts["C"],
        "oxygen": counts["O"],
        "nitrogen": counts["N"],
        "aromatic_nitrogen": counts["aromatic_N"],
        "sulfur": counts["S"],
        "fluorine": counts["F"],
        "chlorine": counts["Cl"],
        "bromine": counts["Br"],
        "iodine": counts["I"],
        "total_atoms": mol.GetNumAtoms(),
        "rings": rdmd.CalcNumRings(mol),
        "aromatic_rings": aromatic_ring_count(mol),
        "amides": count_amides(mol),
        "carbamates": count_carbamates(mol),
        "amines": count_amines(mol),
        "ethers": count_ethers(mol),
        "esters": count_esters(mol),
        "rotors": Lipinski.NumRotatableBonds(mol),
        "fsp3": rdmd.CalcFractionCSP3(mol),
        "spiro_atoms": rdmd.CalcNumSpiroAtoms(mol),
        "qed": float(QED.qed(mol)),
        **bonds,
    }


def validate(mol: Chem.Mol, constraints: GeneratorConstraints) -> Tuple[bool, Dict[str, float], float, List[str]]:
    desc = descriptors(mol)
    issues: List[str] = []
    checks = {
        "mw": constraints.molecular_weight,
        "clogp": constraints.clogp,
        "clogs": constraints.clogs,
        "tpsa": constraints.tpsa,
        "tsa": constraints.tsa,
        "hbd": constraints.hbd,
        "hba": constraints.hba,
        "carbon": constraints.carbon,
        "oxygen": constraints.oxygen,
        "nitrogen": constraints.nitrogen,
        "aromatic_nitrogen": constraints.aromatic_nitrogen,
        "sulfur": constraints.sulfur,
        "fluorine": constraints.fluorine,
        "chlorine": constraints.chlorine,
        "bromine": constraints.bromine,
        "iodine": constraints.iodine,
        "total": constraints.bonds,
        "single": constraints.single_bonds,
        "double": constraints.double_bonds,
        "triple": constraints.triple_bonds,
        "aromatic": constraints.aromatic_bonds,
        "total_atoms": constraints.total_atoms,
        "rings": constraints.rings,
        "aromatic_rings": constraints.aromatic_rings,
        "amides": constraints.amides,
        "amines": constraints.amines,
    }
    ok = True
    for key, rng in checks.items():
        if not rng.contains(desc[key]):
            ok = False
            issues.append(f"{key}={desc[key]}")
    if desc["carbamates"] > 0:
        ok = False
        issues.append(f"carbamates={desc['carbamates']}")
    if desc["rotors"] > 8:
        ok = False
        issues.append(f"rotors={desc['rotors']}")
    if desc["fsp3"] < 0.18 or desc["fsp3"] > 0.82:
        ok = False
        issues.append(f"fsp3={desc['fsp3']:.2f}")
    if desc["spiro_atoms"] > 1:
        ok = False
        issues.append(f"spiro_atoms={desc['spiro_atoms']}")

    score = 0.0
    score += constraints.molecular_weight.score(desc["mw"], constraints.optimum_mw)
    score += constraints.total_atoms.score(desc["total_atoms"], constraints.optimum_total_atoms)
    score += 1.25 * constraints.rings.score(desc["rings"], (2.2, 3.2))
    score += 1.10 * constraints.aromatic_rings.score(desc["aromatic_rings"], (1.0, 2.2))
    score += constraints.clogp.score(desc["clogp"], (-0.5, 4.0))
    score += constraints.clogs.score(desc["clogs"], (-6.0, 1.0))
    score += constraints.hba.score(desc["hba"], (1.0, 7.0))
    score += constraints.hbd.score(desc["hbd"], (0.0, 2.5))
    score += constraints.tpsa.score(desc["tpsa"], (30.0, 115.0))
    score += min(1.0, max(0.0, desc["qed"]))
    score += 1.0 - min(1.0, desc["amides"] / 1.5)
    score += 1.0 - min(1.0, desc["carbamates"] / 1.0)
    score += 1.0 - min(1.0, max(0.0, desc.get("esters", 0) - 1) / 2.0)
    score += 1.0 - min(1.0, max(0.0, desc["rotors"] - 5) / 5.0)
    score += 1.0 - min(1.0, abs(desc["fsp3"] - 0.50) / 0.50)
    score += 1.0 - min(1.0, desc["spiro_atoms"] / 1.0)
    score += 1.0 - min(1.0, max(0.0, 1.0 - desc["ethers"])) * 0.35
    return ok, desc, score, issues


RING_WEIGHTS = {
    "balanced": {
        "benzene": 0.18,
        "pyridine": 0.14,
        "pyrimidine": 0.08,
        "pyrazine": 0.05,
        "imidazole": 0.05,
        "oxazole": 0.04,
        "thiophene": 0.04,
        "cyclohexane": 0.06,
        "piperidine": 0.10,
        "morpholine": 0.08,
        "piperazine": 0.07,
        "cyclopentane": 0.03,
        "pyrrolidine": 0.05,
        "tetrahydrofuran": 0.03,
    },
    "strict_user": {
        "benzene": 0.16,
        "pyridine": 0.13,
        "pyrimidine": 0.08,
        "pyrazine": 0.04,
        "imidazole": 0.04,
        "oxazole": 0.03,
        "thiophene": 0.10,
        "cyclohexane": 0.07,
        "piperidine": 0.09,
        "morpholine": 0.07,
        "piperazine": 0.05,
        "cyclopentane": 0.04,
        "pyrrolidine": 0.05,
        "tetrahydrofuran": 0.05,
    },
}

AROMATIC_RINGS = {"benzene", "pyridine", "pyrimidine", "pyrazine", "imidazole", "oxazole", "thiophene"}
ALIPHATIC_RINGS = {"cyclohexane", "piperidine", "morpholine", "piperazine", "cyclopentane", "pyrrolidine", "tetrahydrofuran"}
LINKER_MODES = ["direct", "methylene", "ether", "amine", "ethyl", "oxyethyl"]


class RuleBasedFromScratchGeneratorV22:
    def __init__(self, constraints: GeneratorConstraints, preset_name: str = "balanced", seed: int = 12345):
        self.constraints = constraints
        self.preset_name = preset_name
        self.rng = random.Random(seed)

    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        items = list(weights.items())
        total = sum(w for _, w in items)
        x = self.rng.random() * total
        acc = 0.0
        for key, weight in items:
            acc += weight
            if x <= acc:
                return key
        return items[-1][0]

    def _build_blueprint(self) -> Dict[str, int]:
        if self.preset_name == "strict_user":
            ring_count = self.rng.choice([2, 3])
            aromatic_rings = self.rng.choice([1, 2])
        else:
            ring_count = self.rng.choices([2, 3, 4], weights=[0.15, 0.55, 0.30])[0]
            aromatic_rings = self.rng.choices([1, 2, 3], weights=[0.15, 0.55, 0.30])[0]
        return {
            "ring_count": ring_count,
            "aromatic_rings": min(aromatic_rings, ring_count),
            "annulations": self.rng.choices([0, 1, 2], weights=[0.18, 0.55, 0.27])[0],
            "spiro": self.rng.choices([0, 1], weights=[0.985, 0.015])[0],
            "linked_ring_extensions": self.rng.choices([1, 2, 3], weights=[0.18, 0.58, 0.24])[0],
            "side_chains": self.rng.randint(2, 4),
            "amide_targets": 0 if self.rng.random() < 0.92 else 1,
            "amine_targets": self.rng.choices([0, 1, 2], weights=[0.22, 0.55, 0.23])[0],
            "ester_targets": self.rng.choices([0, 1], weights=[0.78, 0.22])[0],
            "ketone_targets": self.rng.choices([0, 1], weights=[0.76, 0.24])[0],
            "ether_targets": self.rng.choices([1, 2, 3], weights=[0.32, 0.48, 0.20])[0],
            "methoxy_targets": self.rng.choices([0, 1, 2], weights=[0.48, 0.40, 0.12])[0],
            "hydroxyethyl_targets": self.rng.choices([0, 1], weights=[0.72, 0.28])[0],
            "alcohol_targets": self.rng.choices([0, 1, 2], weights=[0.56, 0.32, 0.12])[0],
            "nitrile_targets": self.rng.choices([0, 1], weights=[0.80, 0.20])[0],
            "thioether_targets": 1 if self.preset_name == "strict_user" else self.rng.choices([0, 1], weights=[0.84, 0.16])[0],
            "halogens_target": 4 if self.preset_name == "strict_user" else self.rng.choices([0, 1, 2], weights=[0.52, 0.34, 0.14])[0],
            "tertiary_amine_like": self.rng.choices([0, 1], weights=[0.88, 0.12])[0],
        }

    def _pick_ring(self, aromatic: bool) -> str:
        weights = RING_WEIGHTS[self.preset_name]
        pool = {k: v for k, v in weights.items() if (k in AROMATIC_RINGS) == aromatic}
        return self._weighted_choice(pool)

    def _add_core(self, builder: MolBuilder, blueprint: Dict[str, int]) -> None:
        # First ring system: one aromatic or heteroaromatic anchor is strongly preferred
        first_ring = self._pick_ring(aromatic=True if blueprint["aromatic_rings"] >= 1 else False)
        builder.add_ring(first_ring)
        current_arom = 1 if first_ring in AROMATIC_RINGS else 0
        current_rings = 1

        # Add remaining rings with a mix of direct and short-linker connections
        while current_rings < blueprint["ring_count"]:
            want_aromatic = current_arom < blueprint["aromatic_rings"]
            ring = self._pick_ring(aromatic=want_aromatic if self.rng.random() < 0.78 else not want_aromatic)
            site = builder.pop_site(self.rng, from_ring_only=True, carbon_only=True) or builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            linker_weights = {
                "direct": 0.28,
                "methylene": 0.24,
                "ether": 0.16,
                "amine": 0.08,
                "ethyl": 0.14,
                "oxyethyl": 0.10,
            }
            if ring in AROMATIC_RINGS:
                linker_weights["direct"] += 0.05
                linker_weights["methylene"] += 0.03
            else:
                linker_weights["ether"] += 0.03
                linker_weights["ethyl"] += 0.03
            linker = self._weighted_choice(linker_weights)
            builder.attach_ring_via_linker(site, ring, linker)
            current_rings += 1
            if ring in AROMATIC_RINGS:
                current_arom += 1

        # Annulate after the base ring count is in place; this preserves complexity while avoiding crude over-fusion
        aromatic_indices = [i for i, m in enumerate(builder.rings_meta) if bool(m["aromatic"])]
        for _ in range(blueprint["annulations"]):
            if aromatic_indices and self.rng.random() < 0.62 and current_rings < 4:
                if builder.annulate_on_ring(self.rng.choice(aromatic_indices), mode="heterocycle" if self.rng.random() < 0.70 else "carbocycle"):
                    current_rings += 1

        for _ in range(blueprint["spiro"]):
            site = builder.pop_site(self.rng, from_ring_only=True, carbon_only=True)
            if site is not None:
                builder.add_spiro_carbocycle(site)

    def _decorate(self, builder: MolBuilder, blueprint: Dict[str, int]) -> None:
        chain_recipes = [
            ["C"], ["C", "C"], ["C", "O"], ["C", "N"], ["C", "C", "O"],
            ["C", "O", "C"], ["C", "C", "N"], ["C", "N", "C"], ["C", "C", "C"],
        ]
        for _ in range(blueprint["side_chains"]):
            site = builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.add_chain(site, self.rng.choice(chain_recipes))

        # Prefer subtle oxygen/nitrogen functionality over repeated amides
        for _ in range(blueprint["ether_targets"]):
            site = builder.pop_site(self.rng, carbon_only=True) or builder.pop_site(self.rng)
            if site is None:
                break
            builder.attach_ether(site)

        for _ in range(blueprint["methoxy_targets"]):
            site = builder.pop_site(self.rng, aromatic_only=True, carbon_only=True) or builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.attach_methoxy(site)

        for _ in range(blueprint["hydroxyethyl_targets"]):
            site = builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.attach_hydroxyethyl(site)

        for _ in range(blueprint["alcohol_targets"]):
            site = builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.attach_alcohol(site)

        for _ in range(blueprint["amine_targets"]):
            site = builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.attach_amine(site, secondary_bias=True)

        for _ in range(blueprint["tertiary_amine_like"]):
            site = builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.attach_dimethylamine_like(site)

        for _ in range(blueprint["ketone_targets"]):
            site = builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.attach_ketone(site)

        for _ in range(blueprint["amide_targets"]):
            site = builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.attach_amide(site, n_substituted=False)

        for _ in range(blueprint["ester_targets"]):
            site = builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.attach_ester(site)

        for _ in range(blueprint["nitrile_targets"]):
            site = builder.pop_site(self.rng, aromatic_only=True, carbon_only=True) or builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.attach_nitrile(site)

        for _ in range(blueprint["thioether_targets"]):
            site = builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.attach_thioether(site)

        if self.preset_name == "strict_user":
            halos = ["F", "Cl", "Br", "I"]
        else:
            halos = self.rng.sample(HALOGENS, k=min(blueprint["halogens_target"], len(HALOGENS)))
        for halo in halos:
            site = builder.pop_site(self.rng, aromatic_only=True, carbon_only=True) or builder.pop_site(self.rng, from_ring_only=True, carbon_only=True) or builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            builder.attach_halogen(site, halo)

    def build_candidate(self) -> Optional[Chem.Mol]:
        builder = MolBuilder()
        blueprint = self._build_blueprint()
        self._add_core(builder, blueprint)
        self._decorate(builder, blueprint)
        try:
            mol = builder.get_mol()
        except Exception:
            return None
        return mol if mol.GetNumAtoms() > 0 else None

    def generate(self, n: int, max_attempts: int = 25000) -> List[Dict[str, float]]:
        accepted: List[Tuple[float, Dict[str, float]]] = []
        seen = set()
        attempts = 0
        while len(accepted) < n and attempts < max_attempts:
            attempts += 1
            mol = self.build_candidate()
            if mol is None:
                continue
            ok, desc, score, _ = validate(mol, self.constraints)
            smi = desc["smiles"]
            if not ok or smi in seen:
                continue
            seen.add(smi)
            desc["score"] = round(score, 4)
            accepted.append((score, desc))
        accepted.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in accepted[:n]]


def write_outputs(records: List[Dict[str, float]], outdir: Path, basename: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    smi_path = outdir / f"{basename}.smi"
    csv_path = outdir / f"{basename}_descriptors.csv"
    json_path = outdir / f"{basename}_descriptors.json"
    with smi_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec["smiles"] + "\n")
    if records:
        fieldnames = list(records[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(records)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rule-based from-scratch molecule generator v2.2")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="balanced")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-attempts", type=int, default=30000)
    p.add_argument("--outdir", type=Path, default=Path("outputs_fromscratch_v2_2"))
    p.add_argument("--basename", type=str, default="fromscratch_v2_2_batch")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    generator = RuleBasedFromScratchGeneratorV22(PRESETS[args.preset], preset_name=args.preset, seed=args.seed)
    records = generator.generate(args.n, max_attempts=args.max_attempts)
    write_outputs(records, args.outdir, args.basename)
    print(f"Generated {len(records)} molecules using preset={args.preset} [v2.2]")
    if args.preset == "strict_user":
        print("Note: strict_user remains restrictive because it forces S/F/Cl/Br/I together.")


if __name__ == "__main__":
    main()


# -------------------------
# Image-conditioned integration helpers for ImagiChem
# -------------------------

def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

class ImageConditionedFromScratchGeneratorV22(RuleBasedFromScratchGeneratorV22):
    """Image-conditioned wrapper around the v2.2 rule-based generator.

    The image profile steers topological complexity, aromatic/aliphatic balance,
    heterocycle usage, linker preferences, and decoration intensity.
    """
    def __init__(self, constraints: GeneratorConstraints, image_profile: Optional[Dict[str, float]] = None,
                 preset_name: str = 'balanced', seed: int = 12345):
        super().__init__(constraints=constraints, preset_name=preset_name, seed=seed)
        self.image_profile = dict(image_profile or {})
        self.morphology = self.image_profile.get('morphology_class', 'balanced_painterly')
        self.art = float(self.image_profile.get('artistic_bias', 0.45))
        self.coh = float(self.image_profile.get('spatial_coherence', 0.5))
        self.comp = float(self.image_profile.get('complexity', 0.5))
        self.sat = float(self.image_profile.get('saturation_mean', 0.4))
        self.palette = float(self.image_profile.get('palette_harmony', 0.5))
        self.edge = float(self.image_profile.get('edge_density', 0.35))

    def _profile_ring_weights(self) -> Dict[str, float]:
        weights = dict(RING_WEIGHTS[self.preset_name])
        aromatic_gain = 1.0 + 0.35 * self.art + 0.20 * self.palette
        hetero_gain = 1.0 + 0.20 * self.comp + 0.25 * (1.0 - self.palette)
        saturated_gain = 1.0 + 0.15 * max(0.0, self.comp - 0.35) + 0.15 * (1.0 - self.art)

        for ring in ('benzene', 'pyridine', 'pyrimidine', 'pyrazine', 'imidazole', 'oxazole', 'thiophene'):
            weights[ring] *= aromatic_gain
        for ring in ('pyridine', 'pyrimidine', 'pyrazine', 'imidazole', 'oxazole'):
            weights[ring] *= hetero_gain
        for ring in ('piperidine', 'morpholine', 'piperazine', 'pyrrolidine', 'tetrahydrofuran', 'cyclohexane', 'cyclopentane'):
            weights[ring] *= saturated_gain

        if self.morphology == 'serene_compact':
            for ring in ('benzene', 'pyridine', 'morpholine', 'piperidine'):
                weights[ring] *= 1.35
            for ring in ('thiophene', 'cyclopentane'):
                weights[ring] *= 0.80
        elif self.morphology == 'crowded_structured':
            for ring in ('pyridine', 'pyrimidine', 'imidazole', 'piperazine', 'tetrahydrofuran'):
                weights[ring] *= 1.30
            for ring in ('benzene', 'cyclohexane'):
                weights[ring] *= 1.10
        elif self.morphology == 'chromatic_expressive':
            for ring in ('oxazole', 'thiophene', 'morpholine', 'tetrahydrofuran'):
                weights[ring] *= 1.35
        elif self.morphology == 'minimal_ordered':
            for ring in ('benzene', 'pyridine', 'pyrimidine'):
                weights[ring] *= 1.35
            for ring in ('piperazine', 'cyclopentane', 'thiophene'):
                weights[ring] *= 0.75

        for k in list(weights):
            weights[k] = max(0.01, float(weights[k]))
        return weights

    def _pick_ring(self, aromatic: bool) -> str:
        weights = self._profile_ring_weights()
        pool = {k: v for k, v in weights.items() if (k in AROMATIC_RINGS) == aromatic}
        return self._weighted_choice(pool)

    def _build_blueprint(self) -> Dict[str, int]:
        # Image-conditioned blueprint: higher coherence and complexity -> richer but more ordered ring systems
        if self.morphology == 'serene_compact':
            ring_count_choices = [2, 3]
            ring_count_w = [0.62, 0.38]
            aromatic_choices = [1, 2]
            aromatic_w = [0.45, 0.55]
        elif self.morphology == 'crowded_structured':
            ring_count_choices = [3, 4]
            ring_count_w = [0.58, 0.42]
            aromatic_choices = [1, 2, 3]
            aromatic_w = [0.18, 0.54, 0.28]
        elif self.morphology == 'chromatic_expressive':
            ring_count_choices = [2, 3, 4]
            ring_count_w = [0.22, 0.56, 0.22]
            aromatic_choices = [1, 2, 3]
            aromatic_w = [0.22, 0.50, 0.28]
        elif self.morphology == 'minimal_ordered':
            ring_count_choices = [2, 3]
            ring_count_w = [0.68, 0.32]
            aromatic_choices = [1, 2]
            aromatic_w = [0.40, 0.60]
        else:
            ring_count_choices = [2, 3, 4]
            ring_count_w = [0.26, 0.54, 0.20]
            aromatic_choices = [1, 2, 3]
            aromatic_w = [0.20, 0.56, 0.24]

        ring_count = self.rng.choices(ring_count_choices, weights=ring_count_w)[0]
        aromatic_rings = self.rng.choices(aromatic_choices, weights=aromatic_w)[0]
        aromatic_rings = min(aromatic_rings, ring_count)

        annulations = self.rng.choices([0, 1, 2], weights=[0.30, 0.50, 0.20])[0]
        if self.comp > 0.62 and self.coh > 0.48:
            annulations = self.rng.choices([0, 1, 2], weights=[0.20, 0.50, 0.30])[0]
        if self.morphology == 'minimal_ordered':
            annulations = self.rng.choices([0, 1], weights=[0.55, 0.45])[0]

        linked_ring_extensions = self.rng.choices([1, 2, 3], weights=[0.20, 0.56, 0.24])[0]
        side_chains = self.rng.choices([1, 2, 3, 4], weights=[0.12, 0.40, 0.34, 0.14])[0]

        ether_targets = self.rng.choices([1, 2, 3], weights=[0.26, 0.52, 0.22])[0]
        methoxy_targets = self.rng.choices([0, 1, 2], weights=[0.50, 0.38, 0.12])[0]
        alcohol_targets = self.rng.choices([0, 1, 2], weights=[0.58, 0.30, 0.12])[0]
        hydroxyethyl_targets = self.rng.choices([0, 1], weights=[0.70, 0.30])[0]
        amine_targets = self.rng.choices([0, 1, 2], weights=[0.24, 0.54, 0.22])[0]
        ester_targets = self.rng.choices([0, 1], weights=[0.90, 0.10])[0]
        ketone_targets = self.rng.choices([0, 1], weights=[0.78, 0.22])[0]
        amide_targets = 0 if self.rng.random() < 0.96 else 1
        nitrile_targets = self.rng.choices([0, 1], weights=[0.82, 0.18])[0]
        thioether_targets = self.rng.choices([0, 1], weights=[0.90, 0.10])[0]
        halogens_target = self.rng.choices([0, 1, 2], weights=[0.60, 0.30, 0.10])[0]
        tertiary_amine_like = self.rng.choices([0, 1], weights=[0.90, 0.10])[0]

        # small image-driven nudges
        if self.morphology == 'serene_compact':
            amine_targets = min(1, amine_targets)
            side_chains = max(1, side_chains - 1)
        elif self.morphology == 'crowded_structured':
            side_chains = min(4, side_chains + 1)
            linked_ring_extensions = min(3, linked_ring_extensions + 1)
        elif self.morphology == 'chromatic_expressive':
            ether_targets = min(3, ether_targets + 1)
            alcohol_targets = min(2, alcohol_targets + 1)
            methoxy_targets = min(2, methoxy_targets + 1)
        elif self.morphology == 'minimal_ordered':
            halogens_target = self.rng.choices([0, 1], weights=[0.72, 0.28])[0]
            tertiary_amine_like = 0

        # stronger bias against spiro and overbuilt architectures
        spiro = 0

        return {
            'ring_count': ring_count,
            'aromatic_rings': aromatic_rings,
            'annulations': annulations,
            'spiro': spiro,
            'linked_ring_extensions': linked_ring_extensions,
            'side_chains': side_chains,
            'amide_targets': amide_targets,
            'amine_targets': amine_targets,
            'ester_targets': ester_targets,
            'ketone_targets': ketone_targets,
            'ether_targets': ether_targets,
            'methoxy_targets': methoxy_targets,
            'hydroxyethyl_targets': hydroxyethyl_targets,
            'alcohol_targets': alcohol_targets,
            'nitrile_targets': nitrile_targets,
            'thioether_targets': thioether_targets,
            'halogens_target': halogens_target,
            'tertiary_amine_like': tertiary_amine_like,
        }

    def _add_core(self, builder: MolBuilder, blueprint: Dict[str, int]) -> None:
        # Override to make ring linking more fluid and less direct than vanilla v2.2
        first_ring = self._pick_ring(aromatic=True if blueprint['aromatic_rings'] >= 1 else False)
        builder.add_ring(first_ring)
        current_arom = 1 if first_ring in AROMATIC_RINGS else 0
        current_rings = 1

        while current_rings < blueprint['ring_count']:
            want_aromatic = current_arom < blueprint['aromatic_rings']
            ring = self._pick_ring(aromatic=want_aromatic if self.rng.random() < 0.78 else not want_aromatic)
            site = builder.pop_site(self.rng, from_ring_only=True, carbon_only=True) or builder.pop_site(self.rng, carbon_only=True)
            if site is None:
                break
            linker_weights = {
                'direct': 0.16,
                'methylene': 0.28,
                'ether': 0.18,
                'amine': 0.07,
                'ethyl': 0.18,
                'oxyethyl': 0.13,
            }
            if ring in AROMATIC_RINGS:
                linker_weights['methylene'] += 0.04
                linker_weights['ethyl'] += 0.03
            else:
                linker_weights['ether'] += 0.03
                linker_weights['oxyethyl'] += 0.03
            if self.morphology == 'minimal_ordered':
                linker_weights['direct'] += 0.06
                linker_weights['methylene'] += 0.04
            linker = self._weighted_choice(linker_weights)
            builder.attach_ring_via_linker(site, ring, linker)
            current_rings += 1
            if ring in AROMATIC_RINGS:
                current_arom += 1

        aromatic_indices = [i for i, m in enumerate(builder.rings_meta) if bool(m['aromatic'])]
        for _ in range(blueprint['annulations']):
            if aromatic_indices and self.rng.random() < 0.55 and current_rings < 4:
                mode = 'heterocycle' if (self.comp > 0.48 or self.morphology in {'crowded_structured', 'chromatic_expressive'}) else 'carbocycle'
                if builder.annulate_on_ring(self.rng.choice(aromatic_indices), mode=mode):
                    current_rings += 1


def generate_from_image_profile(image_profile: Optional[Dict[str, float]], seed: int, n: int,
                                progress_callback=None, preset: str = 'balanced',
                                max_attempts: Optional[int] = None) -> List[Tuple[str, float]]:
    """Generate a batch of from-scratch molecules conditioned by an image profile.

    Returns a list of (smiles, qed_percent) sorted by score.
    """
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')
    constraints = PRESETS.get(preset, BALANCED_CONSTRAINTS)
    generator = ImageConditionedFromScratchGeneratorV22(constraints=constraints, image_profile=image_profile, preset_name=preset, seed=seed)
    if max_attempts is None:
        max_attempts = max(6000, int(n * 140))

    accepted: List[Tuple[float, Dict[str, float]]] = []
    seen = set()
    attempts = 0
    last_progress = -1

    while len(accepted) < n and attempts < max_attempts:
        attempts += 1
        mol = generator.build_candidate()
        if mol is None:
            prog = int(100 * attempts / max_attempts)
            if progress_callback and prog != last_progress:
                progress_callback(prog)
                last_progress = prog
            continue
        ok, desc, score, _ = validate(mol, constraints)
        smi = desc['smiles']
        if ok and smi not in seen:
            seen.add(smi)
            desc['score'] = round(score, 4)
            accepted.append((score, desc))
        prog = int(100 * attempts / max_attempts)
        if progress_callback and prog != last_progress:
            progress_callback(prog)
            last_progress = prog

    accepted.sort(key=lambda x: x[0], reverse=True)
    records = [d for _, d in accepted[:n]]
    results = [(rec['smiles'], float(rec.get('qed', 0.0)) * 100.0) for rec in records]
    if progress_callback:
        progress_callback(100)
    return results
