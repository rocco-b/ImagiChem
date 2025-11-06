from collections import defaultdict, deque, Counter
import random

VALENCES = {"C": 4, "N": 3, "O": 2, "S": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}
ALLOWED_ATOMS = set(VALENCES.keys())
RNG = random.Random(42)
HETERO = {"O","N","S"}
HALOGENS = {"Cl","Br","I","F"}

class MoleculeGraph:
    def __init__(self):
        self.adj = defaultdict(dict)
        self.elements = {}
        self.coords = {}
        self.valence_max = {}
        self.valence_rem = {}

    def get_next_idx(self):
        return max(self.elements.keys()) + 1 if self.elements else 0

    def add_atom(self, idx, element, coords=(0.0, 0.0)):
        if element not in VALENCES:
            raise ValueError(f"Not supported element: {element}")
        self.elements[idx] = element
        self.coords[idx] = coords
        self.valence_max[idx] = VALENCES[element]
        self.valence_rem[idx] = VALENCES[element]

    def new_atom(self, element, coords=(0.0, 0.0)):
        idx = self.get_next_idx()
        self.add_atom(idx, element, coords)
        return idx

    def add_bond(self, a, b, order=1):
        if a == b:
            return False
        if a not in self.elements or b not in self.elements:
            return False
        if self.valence_rem[a] < order or self.valence_rem[b] < order:
            return False
        if not is_bond_allowed(self, a, b):
            return False
        if b in self.adj[a]:
            self.adj[a][b] = self.adj[b][a] = max(self.adj[a][b], order)
            return True
        self.adj[a][b] = order
        self.adj[b][a] = order
        self.valence_rem[a] -= order
        self.valence_rem[b] -= order
        return True

    def remove_bond(self, a, b):
        if b in self.adj[a]:
            order = self.adj[a].pop(b)
            self.adj[b].pop(a, None)
            self.valence_rem[a] += order
            self.valence_rem[b] += order
            return True
        return False

    def demote_bond(self, a, b):
        if b in self.adj[a] and self.adj[a][b] >= 2:
            self.adj[a][b] -= 1
            self.adj[b][a] -= 1
            self.valence_rem[a] += 1
            self.valence_rem[b] += 1
            return True
        return False

    def neighbors(self, a):
        return list(self.adj[a].keys())

    def bonds(self):
        seen = set()
        for a, nbrs in self.adj.items():
            for b, order in nbrs.items():
                if (b, a) in seen:
                    continue
                seen.add((a, b))
                yield a, b, order

    def is_connected(self):
        if not self.elements:
            return True
        start = next(iter(self.elements.keys()))
        visited = set()
        q = deque([start])
        while q:
            u = q.popleft()
            if u in visited:
                continue
            visited.add(u)
            for v in self.adj[u]:
                if v not in visited:
                    q.append(v)
        return len(visited) == len(self.elements)

    def total_capacity(self):
        return sum(self.valence_rem.values())

    def atom_counts(self):
        c = Counter()
        for el in self.elements.values():
            c[el] += 1
        return c

def is_bond_allowed(g: MoleculeGraph, a: int, b: int) -> bool:
    ea = g.elements[a]
    eb = g.elements[b]

    if ea in HETERO and eb in HETERO:
        return False

    if (ea in HETERO and eb in HALOGENS) or (eb in HETERO and ea in HALOGENS):
        return False

    if ea in HALOGENS and eb != "C":
        return False
    if eb in HALOGENS and ea != "C":
        return False

    return True