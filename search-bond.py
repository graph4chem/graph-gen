import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree as KDTree
from tqdm.notebook import tqdm

VALENCE_MAX = {'C': 4,
               'H': 1,
               'N': 4,
               'O': 2,
               'F': 1}
VALENCE_STD = {'C': 4,
               'H': 1,
               'N': 3,
               'O': 2,
               'F': 1}

# expected distances in [A] for covalence 1 bond
BOND_DIST_C1 = {'C': 0.77,
                'H': 0.38,
                'N': 0.75,
                'O': 0.73,
                'F': 0.71}

# order used for finding bonds by atom type
BOND_ORDER = {'H': 0,
              'F': 0,
              'O': 1,
              'N': 2,
              'C': 3}


def add_bond(n_avail, nbond, a0, a1, d1=None):
    key = tuple(sorted((a0, a1)))
    if key in nbond:
        nbond[key][0] += 1.0
    elif d1 is not None:
        nbond[key] = [1.0, d1]
    else:
        raise Exception(f"{a0},{a1} added after phase 1")
    n_avail[a0] -= 1
    n_avail[a1] -= 1


def get_bonded_atoms(atoms, nbond, i):
    """returns: [sorted atoms list], [sorted atom index] )"""
    bonded = []
    for (a0, a1), (n, _) in nbond.items():
        if a0 == i:
            bonded.append((a1, atoms[a1]))
        elif a1 == i:
            bonded.append((a0, atoms[a0]))
    bonded = sorted(bonded, key=lambda b: b[1])
    return "".join([b[1] for b in bonded]), [b[0] for b in bonded]


def search_bonds(kdt, n_avail, nbond, connected, isleaf, coords, atoms,
                 atoms_idx, a0, connect_once=True, VALENCE=VALENCE_STD):
    atom0 = atoms[a0]
    if n_avail[a0] == 0:
        return

    # select closest atoms ORDERED BY DISTANCE: closest first
    # note: the first answer is the atom itself and must be removed
    next_dist, next_i = kdt.query(coords[a0], min(1 + VALENCE[atom0], len(atoms)))
    next_dist = next_dist[1:]  # remove a0 from list
    next_i = next_i[1:]

    # for each #VALENCE closest atoms
    found = False
    for d1, a1 in zip(next_dist, next_i):
        if connect_once and (a1 in connected[a0]):
            continue  # enforce 1-bond only in STEP 1
        atom1 = atoms[a1]
        predicted_bond = BOND_DIST_C1[atom0] + BOND_DIST_C1[atom1]
        if abs(d1 / predicted_bond) < 1.2:  # keep only atoms in the 20% expected distance or closer
            if n_avail[a1] > 0:
                add_bond(n_avail, nbond, a0, a1, d1)
                connected[a0][a1] = 1
                connected[a1][a0] = 1
                if (n_avail[a0] == 0) or (n_avail[a1] == 0):
                    isleaf[a0] = 1
                    isleaf[a1] = 1
                found = True
                # print("leaF/Trunk & avail: "+ ", ".join([f"{i}:{atoms[i]}={leaflabel[isleaf[i]]}{n_avail[i]}"
                #                 for i in ordered_atoms_index]))

        else:
            #print(f"-- match failure in molecule_name={name} {a0}:{atom0}-{a1}:{atoms[a1]}={d1} predicted={predicted_bond}")
            pass
    return found
