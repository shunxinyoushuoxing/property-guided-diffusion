"""Frozen MACE and Equiformer V2 energy inference using bundled model code."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch


def use_bundled_models():
    vendor = str(Path(__file__).resolve().parents[2] / 'vendor')
    for name in ('mace', 'nets', 'ocpmodels'):
        module = sys.modules.get(name)
        if module is not None and getattr(module, '__file__', None):
            if not Path(module.__file__).resolve().is_relative_to(Path(vendor)):
                raise RuntimeError('Start a fresh process to avoid mixing bundled and installed model code')
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


class MACEPredictor:
    def __init__(self, checkpoint, device='cpu', head=None, dtype='float64', **unused):
        use_bundled_models()
        from mace.tools import AtomicNumberTable
        self.device = device
        self.dtype = getattr(torch, dtype)
        # Only load trusted serialized models; model files are supplied separately.
        # e3nn embeds scripted submodules whose device is otherwise not remapped
        # by the outer torch.load call. Keep CPU inference possible for GPU saves.
        original_jit_load = torch.jit.load
        def load_script(*args, **kwargs):
            kwargs.setdefault('map_location', device)
            return original_jit_load(*args, **kwargs)
        with patch('torch.jit.load', load_script):
            self.model = torch.load(checkpoint, map_location=device, weights_only=False)
        if not isinstance(self.model, torch.nn.Module):
            raise ValueError('MACE expects a serialized inference model')
        self.model = self.model.to(device=device, dtype=self.dtype).eval().requires_grad_(False)
        self.elements = AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        self.heads = getattr(self.model, 'heads', ['Default'])
        if head is None and len(self.heads) != 1:
            raise ValueError('Select an explicit head for a multi-head model')
        self.head = head or self.heads[0]
        if self.head not in self.heads:
            raise ValueError('Unknown predictor head')

    def energies(self, structures):
        from mace.data import AtomicData, config_from_atoms
        from mace.tools.torch_geometric import Batch
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(self.dtype)
            graphs = [AtomicData.from_config(config_from_atoms(atoms, head_name=self.head),
                       z_table=self.elements, cutoff=float(self.model.r_max), heads=self.heads)
                      for atoms in structures]
            batch = Batch.from_data_list(graphs).to(self.device)
            with torch.no_grad():
                result = self.model(batch.to_dict(), training=False, compute_force=False)
            return result['energy'].detach().cpu().numpy().reshape(-1)
        finally:
            torch.set_default_dtype(previous_dtype)


class EquiformerV2Predictor:
    def __init__(self, checkpoint, device='cpu', batch_size=4, radius=6.0, max_neighbors=200, **unused):
        use_bundled_models()
        from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
        from ocpmodels.preprocessing import AtomsToGraphs
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = dict(payload['config']['model_attributes'])
        for key in ['num_atoms', 'bond_feat_dim', 'num_targets']:
            config.pop(key, None)
        self.model = EquiformerV2_OC20(num_atoms=-1, bond_feat_dim=-1, num_targets=-1, **config)
        state = {}
        for key, value in payload['state_dict'].items():
            while key.startswith('module.'):
                key = key[7:]
            state[key] = value
        self.model.load_state_dict(state, strict=True)
        self.model.regress_forces = False
        self.model.to(device).eval().requires_grad_(False)
        self.device = device
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive')
        normalizer = payload.get('normalizers', {}).get('target')
        if normalizer is None:
            raise ValueError('Energy normalization metadata is required')
        self.mean = torch.as_tensor(normalizer['mean']).detach().cpu().numpy()
        self.std = torch.as_tensor(normalizer['std']).detach().cpu().numpy()
        if not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError('Invalid energy normalization')
        self.converter = AtomsToGraphs(max_neigh=max_neighbors, radius=radius)

    def energies(self, structures):
        from torch_geometric.data import Batch
        values = []
        for start in range(0, len(structures), self.batch_size):
            graphs = [self.converter.convert(a) for a in structures[start:start+self.batch_size]]
            batch = Batch.from_data_list(graphs).to(self.device)
            with torch.no_grad():
                prediction = self.model(batch)
            if isinstance(prediction, dict):
                prediction = prediction['energy']
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            values.append(prediction.detach().cpu().numpy().reshape(-1))
        return np.concatenate(values) * self.std + self.mean


def make_predictor(config):
    options = dict(config)
    backend = options.pop('backend')
    if backend == 'mace':
        return MACEPredictor(**options)
    if backend == 'equiformer_v2':
        return EquiformerV2Predictor(**options)
    raise ValueError('backend must be mace or equiformer_v2')
