###########################################################################################
# Data parsing utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging

import os

from dataclasses import dataclass, field

from typing import Any, Dict, List, Optional, Sequence, Tuple

import ase.data

import ase.io


import numpy as np

from mace.tools import AtomicNumberTable, DefaultKeys

Positions = np.ndarray  # [..., 3]

Cell = np.ndarray  # [3,3]

Pbc = tuple  # (3,)

DEFAULT_CONFIG_TYPE = "Default"

DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}

@dataclass
class KeySpecification:
    info_keys: Dict[str, str] = field(default_factory=dict)
    arrays_keys: Dict[str, str] = field(default_factory=dict)

    def update(
        self,
        info_keys: Optional[Dict[str, str]] = None,
        arrays_keys: Optional[Dict[str, str]] = None,
    ):
        if info_keys is not None:
            self.info_keys.update(info_keys)
        if arrays_keys is not None:
            self.arrays_keys.update(arrays_keys)
        return self

    @classmethod
    def from_defaults(cls):
        instance = cls()
        return update_keyspec_from_kwargs(instance, DefaultKeys.keydict())

def update_keyspec_from_kwargs(
    keyspec: KeySpecification, keydict: Dict[str, str]
) -> KeySpecification:
    # convert command line style property_key arguments into a keyspec
    infos = [
        "energy_key",
        "stress_key",
        "virials_key",
        "dipole_key",
        "head_key",
        "elec_temp_key",
        "total_charge_key",
        "polarizability_key",
        "total_spin_key",
    ]
    arrays = ["forces_key", "charges_key"]
    info_keys = {}
    arrays_keys = {}
    for key in infos:
        if key in keydict:
            info_keys[key[:-4]] = keydict[key]
    for key in arrays:
        if key in keydict:
            arrays_keys[key[:-4]] = keydict[key]

    # automagically add properties for embeddings
    if keydict.get("embedding_specs") is not None:
        for embed_name, embed_spec in keydict["embedding_specs"].items():
            key = embed_spec.get("key", embed_name)
            if embed_spec["per"] == "atom":
                arrays_keys[embed_name] = key
            elif embed_spec["per"] == "graph":
                info_keys[embed_name] = key
            else:
                raise ValueError(
                    f"Unsupported embedding_specs per {embed_spec['per']} for {embed_name}"
                )

    keyspec.update(info_keys=info_keys, arrays_keys=arrays_keys)
    return keyspec

@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    properties: Dict[str, Any]
    property_weights: Dict[str, float]
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

    weight: float = 1.0  # weight of config in loss
    config_type: str = DEFAULT_CONFIG_TYPE  # config_type of config
    head: str = "Default"  # head used to compute the config

Configurations = List[Configuration]

def config_from_atoms(
    atoms: ase.Atoms,
    key_specification: KeySpecification = KeySpecification(),
    config_type_weights: Optional[Dict[str, float]] = None,
    head_name: str = "Default",
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc().tolist())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", "Default")
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type, 1.0
    )

    properties = {}
    property_weights = {}
    for name in list(key_specification.arrays_keys) + list(key_specification.info_keys):
        property_weights[name] = atoms.info.get(f"config_{name}_weight", 1.0)

    for name, atoms_key in key_specification.info_keys.items():
        properties[name] = atoms.info.get(atoms_key, None)
        if not atoms_key in atoms.info:
            property_weights[name] = 0.0

    for name, atoms_key in key_specification.arrays_keys.items():
        properties[name] = atoms.arrays.get(atoms_key, None)
        if not atoms_key in atoms.arrays:
            property_weights[name] = 0.0

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        properties=properties,
        weight=weight,
        property_weights=property_weights,
        head=head_name,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )
