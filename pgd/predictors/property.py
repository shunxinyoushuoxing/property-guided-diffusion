"""Convert predicted energies into generic site-resolved objectives."""
import numpy as np
from ase import Atoms


def aggregate(values, mode):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite property values')
    reducers = {'mean': np.mean, 'min': np.min, 'max': np.max,
                'min_abs': lambda a: np.min(np.abs(a))}
    if mode not in reducers:
        raise ValueError('Unknown aggregation mode')
    return float(reducers[mode](x))


class PropertyEvaluator:
    """Evaluate energy or a user-specified set of derived state energies.

    A recipe supplies sites, appended atom positions/numbers, and either a
    reaction path barrier or a matrix of linear energy differences. No chemical
    system, reference energy, or adsorption geometry is built into this class.
    """
    def __init__(self, predictor, recipe=None):
        self.predictor = predictor
        self.recipe = recipe or {'mode': 'energy'}

    def __call__(self, atoms):
        recipe = self.recipe
        if recipe['mode'] == 'energy':
            value = float(self.predictor.energies([atoms])[0])
            return value / len(atoms) if recipe.get('per_atom', False) else value
        sites = recipe.get('site_indices', list(range(len(atoms))))
        if not sites:
            raise ValueError('No evaluation sites')
        scores = []
        for site in sites:
            if not 0 <= site < len(atoms):
                raise ValueError('Evaluation site is out of range')
            states = []
            for specification in recipe['states']:
                state = atoms.copy()
                numbers = specification.get('numbers', [])
                offsets = np.asarray(specification.get('offsets', []), dtype=float).reshape(-1, 3)
                if len(numbers) != len(offsets):
                    raise ValueError('State numbers and offsets must match')
                if len(numbers):
                    if specification.get('coordinates', 'cartesian') == 'fractional':
                        offsets = offsets @ np.asarray(atoms.cell)
                    elif specification.get('coordinates', 'cartesian') != 'cartesian':
                        raise ValueError('Unknown coordinate system')
                    state += Atoms(numbers=numbers, positions=atoms.positions[site] + offsets)
                states.append(state)
            energies = np.asarray(self.predictor.energies(states), dtype=float)
            energies += np.asarray(recipe.get('energy_offsets', np.zeros(len(states))), dtype=float)
            if recipe['mode'] == 'path_barrier':
                value = float(np.max(energies) - energies[int(recipe.get('reference_state', 0))])
            elif recipe['mode'] == 'linear_steps':
                coefficients = np.asarray(recipe['coefficients'], dtype=float)
                if coefficients.ndim != 2 or coefficients.shape[1] != len(energies):
                    raise ValueError('Coefficient matrix must have one column per state')
                steps = coefficients @ energies + np.asarray(recipe.get('step_offsets', 0.0))
                value = aggregate(steps, recipe.get('step_reduction', 'max'))
                value -= float(recipe.get('reference_value', 0.0))
            else:
                raise ValueError('Unknown property recipe mode')
            scores.append(value)
        return aggregate(scores, recipe.get('site_reduction', 'mean'))
