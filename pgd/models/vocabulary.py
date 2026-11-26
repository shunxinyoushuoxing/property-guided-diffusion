"""Explicit category order and optional spatial constraints for inference."""
import torch


class Vocabulary:
    def __init__(self, atomic_numbers, position_rules=()):
        if len(atomic_numbers) < 2 or atomic_numbers[0] != 0:
            raise ValueError('Vocabulary must begin with the mask token 0')
        if len(set(atomic_numbers)) != len(atomic_numbers):
            raise ValueError('Vocabulary entries must be unique')
        if any(type(z) is not int or not 0 <= z <= 118 for z in atomic_numbers):
            raise ValueError('Vocabulary entries must be integer atomic numbers')
        self.numbers = tuple(atomic_numbers)
        self.z_to_index = {z: i for i, z in enumerate(atomic_numbers) if i}
        self.rules = tuple(position_rules)
        for rule in self.rules:
            if rule['axis'] not in [0, 1, 2] or not rule['allowed_indices']:
                raise ValueError('Invalid position rule')
            if any(i < 0 or i >= len(atomic_numbers) for i in rule['allowed_indices']):
                raise ValueError('Position rule category is out of range')

    def encode(self, numbers):
        return torch.tensor([self.z_to_index[int(z)] for z in numbers],
                            device=numbers.device, dtype=torch.long)

    def decode(self, indices):
        return torch.tensor(self.numbers, device=indices.device)[indices]

    def position_mask(self, coords):
        mask = torch.ones((len(coords), len(self.numbers)), dtype=torch.bool, device=coords.device)
        for rule in self.rules:
            region = (coords[:, rule['axis']] >= rule.get('lower', 0.0)) & (
                coords[:, rule['axis']] < rule.get('upper', 1.0))
            allowed = torch.zeros(len(self.numbers), dtype=torch.bool, device=coords.device)
            allowed[rule['allowed_indices']] = True
            mask[region] &= allowed
        if not mask.any(dim=1).all():
            raise ValueError('Overlapping position rules forbid all categories at a site')
        return mask
