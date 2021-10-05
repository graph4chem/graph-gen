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


def compute_bonds(structures, molecules):
    out_name = []
    out_a0 = []
    out_a1 = []
    out_n = []
    out_dist = []
    out_error = []
    out_type = []
    cycle_name = []
    cycle_index = []
    cycle_seq = []
    cycle_atom_index = []
    charge_name = []
    charge_atom_index = []
    charge_value = []

    for imol, name in tqdm(list(enumerate(molecules))):
        molecule = structures.loc[name]
        error = 0
        atoms = molecule.atom.values
        atoms_idx = molecule.atom_index.values

        n_avail = np.asarray([VALENCE_STD[a] for a in atoms])
        n_charge = np.zeros(len(atoms), dtype=np.float16)
        isleaf = np.zeros(len(atoms), dtype=bool)  # is the atom in the leafs of connection tree?
        coords = molecule[['x', 'y', 'z']].values
        kdt = KDTree(coords)  # use an optimized structure for closest match query
        nbond = {}
        connected = {i: {} for i in atoms_idx}

        # select Hydrogen first to avoid butadyne-like ordering failures (molecule_name=dsgdb9nsd_000023)
        ordered_atoms_index = list(atoms_idx)
        ordered_atoms_index.sort(key=lambda i: BOND_ORDER[atoms[i]])
        ordered_atoms_index = np.asarray(ordered_atoms_index)

        # STEP 1: 1-bond connect each atom with closest match
        #         only one bond for each atom pair is done in step 1
        for a0 in ordered_atoms_index:
            search_bonds(kdt, n_avail, nbond, connected, isleaf, coords, atoms, atoms_idx,
                         a0, connect_once=True, VALENCE=VALENCE_STD)

        # STEP 2: greedy connect n-bonds, progressing from leafs of connection tree
        while (((n_avail > 0).sum() > 0) and isleaf).sum() > 0:
            progress = False
            for a0 in ordered_atoms_index:
                # print("leaF/Trunk & avail: " + ", ".join([f"{i}:{atoms[i]}={leaflabel[leaf[i]]}{n_avail[i]}"
                #                                          for i in ordered_atoms_index]))
                if (n_avail[a0] > 0) and isleaf[a0]:
                    for a1 in connected[a0]:
                        if (n_avail[a0] > 0) and (n_avail[a1] > 0):
                            add_bond(n_avail, nbond, a0, a1)
                            progress = True
                            if (n_avail[a0] == 0) or (n_avail[a1] == 0):
                                isleaf[a0] = 1
                                isleaf[a1] = 1
            if not progress:
                break

        # gather remaining multiple bonds
        if n_avail.sum() > 0:
            for key in nbond.keys():
                a0, a1 = key
                while (n_avail[a0] > 0) and (n_avail[a1] > 0):
                    add_bond(n_avail, nbond, a0, a1)

        # STEP 3: Detect cycles : algorithm complexity in O(m^2 * n)
        # paper : https://link.springer.com/article/10.1007/s00453-007-9064-z
        # nx doc:
        # https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.cycles.minimum_cycle_basis.html
        graph = nx.Graph([bond for bond in nbond.keys()])
        unordered_cycles = nx.minimum_cycle_basis(graph)

        # index atoms by their sequential order in the cycle: i.e follow bonds
        # Note: this code can be written in a much cleaner way!
        if len(unordered_cycles) > 0:
            for icycle, c in enumerate(unordered_cycles):
                available = {i: 1 for i in c}
                a0 = c[0]
                cycle = [a0]
                del(available[a0])
                for index in range(1, len(c)):
                    # get atoms bonded to a0
                    bonded = [b for b in nbond.keys() if a0 in b]
                    bonded = list(map(lambda b: b[0] if b[1] == a0 else b[1], bonded))

                    # get next atom and remove it from cycle
                    assert(len(bonded) > 0)
                    found = False
                    for a1 in bonded:
                        if (a1 in bonded) and (a1 in available):
                            cycle.append(a1)
                            del(available[a1])
                            a0 = a1
                            found = True
                            break
                    assert(found)

                # and add cycles found to the cycle dataframe lists
                cycle_name.extend([name] * len(cycle))
                cycle_index.extend([icycle] * len(cycle))
                cycle_seq.extend(np.arange(len(cycle)))
                cycle_atom_index.extend(cycle)

        # display info on failed molecules
        if n_avail.sum() > 0:
            error = 1
            print(f"   Remaining bondings={n_avail.sum()} for molecule_name={name}, atoms: " +
                  ", ".join([f"{i}:{atoms[i]}" for i in atoms_idx if n_avail[i] > 0]))

        # inputs for DataFrame bonds
        for (a0, a1), (n, dist) in nbond.items():
            # append to python lists which is 7x faster than toa pd.DataFrame
            out_name.append(name)
            out_a0.append(a0)
            out_a1.append(a1)
            out_n.append(n)
            out_dist.append(dist)
            out_error.append(error)
            out_type.append(f"{n:0.1f}" + "".join(sorted(f"{atoms[a0]}{atoms[a1]}")))

        # inputs for DataFrame charges
        charge_name.extend([name] * len(atoms))
        charge_atom_index.extend(molecule.atom_index.values)
        charge_value.extend(n_charge)

    bonds = pd.DataFrame({'molecule_name': out_name, 'atom_index_0': out_a0, 'atom_index_1': out_a1, 'nbond': out_n,
                          'L2dist': out_dist, 'error': out_error, 'bond_type': out_type})
    cycles = pd.DataFrame({'molecule_name': cycle_name, 'cycle_index': cycle_index,
                           'cycle_seq': cycle_seq, 'atom_index': cycle_atom_index})

    return bonds, cycles
