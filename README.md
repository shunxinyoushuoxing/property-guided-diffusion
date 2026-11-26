# Property Guided Diffusion

Property Guided Diffusion (PGD) uses a frozen structural diffusion model as a local search operator and a separate frozen property predictor to guide selection. Each iteration partially noises parent structures, generates candidate structures by reverse diffusion, accepts improving candidates, and resamples the population. Composition-based fitness sharing reduces concentration around similar candidates.

This repository contains inference code: joint coordinate/lattice/discrete-species diffusion, population search, optional template assignment, generic energy-to-property evaluation, and MACE and Equiformer V2 predictor implementations. It contains no datasets, pretrained weights, training scripts, or training configurations. Users supply compatible weights and inference metadata separately.

## Installation

Use Python 3.10 and a PyTorch 2.4 environment. Install PyTorch and matching `torch-scatter` and `torch-geometric` packages for your CPU/CUDA platform, then install the relevant requirements:

```bash
pip install -r requirements/generator.txt
```

Predictors may run in separate environments: install `requirements/mace.txt` or `requirements/equiformer.txt` there. Model sources are bundled under `vendor/`; installing another copy of those model packages is unnecessary. Set `predictor_python` in the inference input to the predictor environment's Python executable when using separate environments. See [inference inputs](docs/inference.md) for dependencies and file contracts.

## Inference

Supply a generator checkpoint, model architecture and category order, a symmetry specification, and a predictor configuration. The example contains only inference options and placeholder paths.

```bash
python -m pgd sample --config examples/inference.json
python -m pgd search --config examples/inference.json
```

Edit the paths and provide the private inputs before running these commands. Search writes the best structure, final population, property values, and iteration history to a new output directory. Both models remain frozen. The search does not produce an updated model checkpoint.

The default `independent` anchor mode samples a forward perturbation for each candidate, matching the original population loop. Set `anchor_mode` to `shared` to draw one forward anchor per parent and branch several reverse trajectories from it. No resampling, standard resampling, diversity-aware resampling, fixed noise and linear noise annealing are supported.

## Checks

```bash
python -m unittest discover -s tests -v
```

These checks use synthetic inputs and verify the method's numerical operators and search rules. They do not reproduce application-specific results or validate a user's predictor calibration.

## Third-party code

The diffusion implementation derives from [DiffCSP++](https://github.com/jiaor17/DiffCSP-PP). Predictor runtimes derive from [MACE](https://github.com/ACEsuit/mace), [Equiformer V2](https://github.com/atomicarchitects/equiformer_v2), and the Open Catalyst graph utilities. Public upstream copyright notices and licenses are retained in [licenses](licenses/). See [third-party notes](THIRD_PARTY.md) for the inference subset and modifications.
