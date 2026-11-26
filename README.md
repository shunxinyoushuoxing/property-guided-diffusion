# Property Guided Diffusion

Property Guided Diffusion (PGD) uses a frozen structural diffusion model as a local search operator and a separate frozen property predictor to guide selection. Each iteration partially noises parent structures, generates candidate structures by reverse diffusion, accepts improving candidates, and resamples the population. Composition-based fitness sharing reduces concentration around similar candidates.

The implementation includes joint coordinate/lattice/discrete-species diffusion, population search, template mapping, energy-to-property evaluation, and MACE and Equiformer V2 predictors.

## Installation

Use Python 3.10 and a PyTorch 2.4 environment. Install PyTorch and matching `torch-scatter` and `torch-geometric` packages for your CPU/CUDA platform, then install the relevant requirements:

```bash
pip install -r requirements/generator.txt
```

Predictor dependencies are listed in `requirements/mace.txt` and `requirements/equiformer.txt`, with model implementations under `vendor/`. To run a predictor in a separate environment, set `predictor_python` to that environment's Python executable.

## Inference

Configure the model, symmetry, and predictor paths in `examples/inference.json`. See [inference inputs](docs/inference.md) for parameter definitions.

```bash
python -m pgd sample --config examples/inference.json
python -m pgd search --config examples/inference.json
```

`sample` generates structures; `search` runs property-guided population optimization. Search saves the best structure, final population, property values, and iteration history.

The search supports independent or shared forward-noise anchors, fixed or linearly annealed noise, and standard or diversity-aware population resampling. Set `resampling` to `none` to evolve each population member independently.

## Checks

```bash
python -m unittest discover -s tests -v
```

The tests cover diffusion operators, property evaluation, geometry mapping, and population selection.

## Third-party code

The implementation builds on [DiffCSP++](https://github.com/jiaor17/DiffCSP-PP), [MACE](https://github.com/ACEsuit/mace), [Equiformer V2](https://github.com/atomicarchitects/equiformer_v2), and Open Catalyst graph utilities. See [licenses](licenses/) and [third-party notes](THIRD_PARTY.md).
