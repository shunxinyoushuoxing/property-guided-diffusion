"""Dataset-free symmetry input and structure conversion."""
from types import SimpleNamespace
import numpy as np
import torch
from ase import Atoms


def make_batch(spec, device='cpu'):
    """Build one graph from explicit symmetry operators, or P1 identity operators."""
    n = int(spec['num_atoms'])
    group = int(spec.get('spacegroup', 1))
    if n < 1 or not 1 <= group <= 230:
        raise ValueError('Invalid number of sites or space group')
    if group != 1 and not all(k in spec for k in ['anchor_index', 'operations']):
        raise ValueError('Non-P1 symmetry requires explicit anchors and operations')
    anchors = torch.tensor(spec.get('anchor_index', list(range(n))), dtype=torch.long, device=device)
    ops = torch.tensor(spec.get('operations', np.tile(np.eye(4), (n, 1, 1)).tolist()),
                       dtype=torch.float32, device=device)
    if anchors.shape != (n,) or ops.shape != (n, 4, 4):
        raise ValueError('Expected N anchors and N homogeneous 4 x 4 operations')
    if (anchors < 0).any() or (anchors >= n).any():
        raise ValueError('Anchor indices are out of range')
    if not torch.equal(anchors[anchors], anchors):
        raise ValueError('Each anchor must point to itself')
    return SimpleNamespace(num_graphs=1, num_nodes=n,
                           num_atoms=torch.tensor([n], device=device),
                           batch=torch.zeros(n, dtype=torch.long, device=device),
                           anchor_index=anchors, ops=ops,
                           ops_inv=torch.linalg.inv(ops[:, :3, :3]),
                           spacegroup=torch.tensor([group], device=device))


def to_atoms(output):
    numbers = output['atom_types'].detach().cpu().numpy().astype(int)
    coords = output['frac_coords'].detach().cpu().numpy()
    lattice = output['lattices'][0].detach().cpu().numpy()
    if (numbers < 1).any() or (numbers > 118).any():
        raise ValueError('Generated structure contains unresolved category tokens')
    if not np.isfinite(coords).all() or not np.isfinite(lattice).all():
        raise ValueError('Generated structure contains non-finite geometry')
    return Atoms(numbers=numbers, scaled_positions=coords, cell=lattice, pbc=True)


def valid_structure(atoms, min_distance=0.5, min_volume=0.1):
    if len(atoms) == 0 or not np.isfinite(atoms.positions).all() or atoms.get_volume() <= min_volume:
        return False
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    return bool(np.all(distances >= min_distance))


def map_to_template(atoms, template, groups, max_distance):
    """Assign generated sites to user-defined template groups, allowing vacancies.

    Each group defines generated atomic numbers and compatible template indices.
    Template sites outside groups stay fixed. Unassigned sites inside groups are
    removed only when allow_vacancies is explicitly true.
    """
    from scipy.optimize import linear_sum_assignment
    from ase.geometry import find_mic
    result = template.copy()
    used_source, used_target, remove = set(), set(), []
    for group in groups:
        targets = list(group['template_indices'])
        source = [i for i, z in enumerate(atoms.numbers) if int(z) in group['atomic_numbers']]
        if set(source) & used_source or set(targets) & used_target:
            raise ValueError('Template mapping groups must not overlap')
        if any(i < 0 or i >= len(template) for i in targets):
            raise ValueError('Template index is out of range')
        used_source.update(source)
        used_target.update(targets)
        if len(source) > len(targets):
            raise ValueError('More generated sites than compatible template sites')
        if len(source) != len(targets) and not group.get('allow_vacancies', False):
            raise ValueError('Site count differs and vacancies are disabled')
        assigned = set()
        if source:
            delta = atoms.positions[source, None, :] - template.positions[None, targets, :]
            _, distance = find_mic(delta.reshape(-1, 3), template.cell, template.pbc)
            cost = distance.reshape(len(source), len(targets))
            rows, cols = linear_sum_assignment(cost)
            if np.any(cost[rows, cols] > max_distance):
                raise ValueError('Template assignment exceeds maximum distance')
            for row, col in zip(rows, cols):
                result.numbers[targets[col]] = atoms.numbers[source[row]]
                assigned.add(targets[col])
        remove.extend(set(targets) - assigned)
    if used_source != set(range(len(atoms))):
        raise ValueError('Every generated site must belong to a mapping group')
    if remove:
        del result[sorted(remove)]
    return result
