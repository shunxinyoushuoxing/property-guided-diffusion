"""Frozen-model population search with local diffusion and fitness sharing."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
from .geometry import to_atoms, valid_structure


@dataclass
class Member:
    coords: torch.Tensor
    lattice: torch.Tensor
    numbers: torch.Tensor
    atoms: object
    value: float
    reward: float


def noise_state(model, batch, member, step):
    if not 1 <= step < model.beta_scheduler.timesteps:
        raise ValueError('Noise step must be in [1, timesteps - 1]')
    model._sync_d3pm_device()
    coords = member.coords.to(model.device)
    lattice = member.lattice.to(model.device)
    noisy_coords = (coords + model.sigma_scheduler.sigmas[step] * torch.randn_like(coords)) % 1
    alpha = model.beta_scheduler.alphas_cumprod[step]
    noise = model.crystal_family.proj_k_to_spacegroup(torch.randn_like(lattice), batch.spacegroup)
    clean = model.crystal_family.proj_k_to_spacegroup(lattice, batch.spacegroup)
    noisy_lattice = model.crystal_family.proj_k_to_spacegroup(
        alpha.sqrt() * clean + (1 - alpha).sqrt() * noise, batch.spacegroup)
    indices = model.vocabulary.encode(member.numbers.to(model.device))
    discrete_step = min(int(step * model.d3pm.T / model.beta_scheduler.timesteps) + 1, model.d3pm.T)
    steps = torch.full_like(indices, discrete_step)
    noisy_types = model.d3pm.q_sample(indices, steps)[batch.anchor_index]
    return dict(init_coords=noisy_coords, init_lattice=noisy_lattice, init_types=noisy_types)


def composition(member):
    counts = np.bincount(member.numbers.numpy().astype(int), minlength=119).astype(float)
    return counts / counts.sum()


def selection_probabilities(population, beta, radius=None):
    if not population or beta < 0 or (radius is not None and radius <= 0):
        raise ValueError('Invalid selection parameters')
    rewards = np.array([m.reward for m in population])
    if not np.isfinite(rewards).all():
        raise ValueError('Population scores must be finite')
    log_weights = beta * rewards
    if radius is not None:
        vectors = np.array([composition(m) for m in population])
        distances = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1)
        density = np.maximum(0.0, 1.0 - distances / radius).sum(axis=1)
        log_weights -= np.log(density)
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


def resample(population, rng, mode='diverse', beta=2.0, elites=1, radius=0.15):
    if mode == 'none':
        return population
    if mode not in ['standard', 'diverse'] or not 0 <= elites <= len(population):
        raise ValueError('Invalid resampling mode or elite count')
    best = np.argsort([-m.reward for m in population], kind='stable')[:elites]
    probs = selection_probabilities(population, beta, radius if mode == 'diverse' else None)
    sampled = rng.choice(len(population), size=len(population)-elites, p=probs)
    return [deepcopy(population[int(i)]) for i in np.concatenate((best, sampled))]


class Search:
    def __init__(self, model, batch, evaluator, options, postprocess=None, initial_model=None):
        self.model, self.batch, self.evaluator = model, batch, evaluator
        self.options = dict(options)
        self.postprocess = postprocess or (lambda atoms: atoms)
        self.initial_model = initial_model or model
        self.failures = []
        self.evaluations = 0
        self.initial_model.eval().requires_grad_(False)
        self.model.eval().requires_grad_(False)
        if self.model.vocabulary.numbers != self.initial_model.vocabulary.numbers:
            raise ValueError('Initialization and iterative models must use identical vocabularies')

    def evaluate(self, output):
        try:
            raw = to_atoms(output)
            atoms = self.postprocess(raw)
            if not valid_structure(atoms, self.options.get('min_distance', 0.5)):
                raise ValueError('Structure failed geometry validation')
            self.evaluations += 1
            value = float(self.evaluator(atoms))
            if not np.isfinite(value):
                raise ValueError('Non-finite property prediction')
            objective = self.options.get('objective', 'minimize')
            if objective == 'minimize':
                reward = -value
            elif objective == 'maximize':
                reward = value
            elif objective == 'target':
                reward = -abs(value - float(self.options['target']))
            else:
                raise ValueError('Unknown objective')
            matrix = self.model.crystal_family.de_so3(output['lattices'].to(self.model.device))
            latent = self.model.crystal_family.m2v(matrix).detach().cpu()
            return Member(output['frac_coords'].detach().cpu(), latent,
                          output['atom_types'].detach().cpu(), atoms, value, reward)
        except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
            self.failures.append(str(exc))
            return None

    @torch.no_grad()
    def run(self, callback=None):
        opt = self.options
        size, children, iterations = (int(opt[k]) for k in ['population', 'candidates', 'iterations'])
        if min(size, children, iterations) < 1:
            raise ValueError('Population, candidates and iterations must be positive')
        start, end = int(opt['noise_start']), int(opt.get('noise_end', opt['noise_start']))
        if not 1 <= min(start, end) <= max(start, end) < self.model.beta_scheduler.timesteps:
            raise ValueError('Noise schedule is outside the model time range')
        seed = int(opt.get('seed', 0))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        step_lr = float(opt.get('step_lr', 1e-5))
        population = []
        for _ in range(int(opt.get('max_initial_attempts', size * 20))):
            output, _ = self.initial_model.sample(self.batch, step_lr=step_lr)
            member = self.evaluate(output)
            if member is not None:
                population.append(member)
            if len(population) == size:
                break
        if len(population) != size:
            raise RuntimeError('Could not initialize a valid population; inspect inputs and failures')
        best = deepcopy(max(population, key=lambda m: m.reward))
        history = []
        for iteration in range(iterations):
            step = round(start + iteration * (end-start) / max(1, iterations-1))
            updated = []
            for parent in population:
                winner = parent
                anchor_mode = opt.get('anchor_mode', 'independent')
                if anchor_mode not in ['shared', 'independent']:
                    raise ValueError('anchor_mode must be shared or independent')
                shared = noise_state(self.model, self.batch, parent, step) if anchor_mode == 'shared' else None
                for _ in range(children):
                    anchor = shared if shared is not None else noise_state(self.model, self.batch, parent, step)
                    output, _ = self.model.sample(self.batch, time_start=step, step_lr=step_lr, **anchor)
                    candidate = self.evaluate(output)
                    if candidate is not None and candidate.reward > winner.reward:
                        winner = candidate
                updated.append(winner)
            current_best = max(updated, key=lambda m: m.reward)
            if current_best.reward > best.reward:
                best = deepcopy(current_best)
            population = resample(updated, rng, mode=opt.get('resampling', 'diverse'),
                                  beta=float(opt.get('beta', 2.0)), elites=int(opt.get('elites', 1)),
                                  radius=float(opt.get('sharing_radius', 0.15)))
            row = dict(iteration=iteration, noise_step=step, evaluations=self.evaluations,
                       best_value=best.value, mean_value=float(np.mean([m.value for m in population])),
                       unique_compositions=len({tuple(composition(m).round(3)) for m in population}),
                       failed_candidates=len(self.failures))
            history.append(row)
            if callback:
                callback(row, population, best)
        return population, best, history
