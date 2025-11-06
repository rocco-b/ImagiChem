import math
from graph_utils import MoleculeGraph

def ester_graph(start_id=0):
    g = MoleculeGraph()
    g.add_atom(start_id, "C")
    g.add_atom(start_id + 1, "O")
    g.add_atom(start_id + 2, "O")
    g.add_bond(start_id, start_id + 1, order=2)
    g.add_bond(start_id, start_id + 2, order=1)
    return g

def amide_graph(start_id=0):
    g = MoleculeGraph()
    g.add_atom(start_id, "C")
    g.add_atom(start_id + 1, "O")
    g.add_atom(start_id + 2, "N")
    g.add_bond(start_id, start_id + 1, order=2)
    g.add_bond(start_id, start_id + 2, order=1)
    return g

def alcohol_graph(start_id=0):
    g = MoleculeGraph()
    g.add_atom(start_id, "O")
    return g

def amine_graph(start_id=0):
    g = MoleculeGraph()
    g.add_atom(start_id, "N")
    return g

def nitro_graph(start_id=0):
    g = MoleculeGraph()
    g.add_atom(start_id, "N")
    g.add_atom(start_id + 1, "O")
    g.add_atom(start_id + 2, "O")
    g.add_bond(start_id, start_id + 1, order=2)
    g.add_bond(start_id, start_id + 2, order=1)
    return g

def ether_graph(start_id=0):
    g = MoleculeGraph()
    g.add_atom(start_id, "O")
    g.add_atom(start_id + 1, "C")
    g.add_bond(start_id, start_id + 1, order=1)
    return g

GROUP_LIBRARY = [
    {
        "name": "ester",
        "factory": ester_graph,
        "requirements": {"C": 1, "O": 2},
        "weight": 0.45
    },
    {
        "name": "amide",
        "factory": amide_graph,
        "requirements": {"C": 1, "O": 1, "N": 1},
        "weight": 0.45
    },
    {
        "name": "amine",
        "factory": amine_graph,
        "requirements": {"N": 1},
        "weight": 0.25
    },
    {
        "name": "alcohol",
        "factory": alcohol_graph,
        "requirements": {"O": 1},
        "weight": 0.20
    },
    {
        "name": "nitro",
        "factory": nitro_graph,
        "requirements": {"N": 1, "O": 2},
        "weight": 0.15
    },
    {
        "name": "ether",
        "factory": ether_graph,
        "requirements": {"N": 1, "O": 2},
        "weight": 0.15
    },
]