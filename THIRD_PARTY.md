# Third-party implementation notes

This distribution retains the public copyright notices required by upstream code. They identify upstream software authors, not project participants or private deployment information.

| Component | Source | Scope in this repository |
| --- | --- | --- |
| Diffusion backbone and lattice representation | DiffCSP++ | Message passing, schedules, lattice transforms and reverse sampling; trainer and dataset integrations removed |
| Discrete diffusion | Project inference implementation using the D3PM formulation | Forward and reverse categorical kernels; loss functions removed |
| MACE | ACEsuit/mace, source revision `667eee4e58d23a38ff5a75122109ec2025809649` | Energy model classes, graph conversion and mathematical utilities; no trainer, optimizer, fine-tuning entry point or dataset |
| Equiformer V2 | atomicarchitects/equiformer_v2, source revision `cf45ee209abdceaa606c9a8dce50c9c753d470b2` | Network and energy-inference dependencies; model registration removed |
| Open Catalyst | Utilities accompanying the Equiformer V2 source | Periodic graph construction, base model and radial functions |

License texts are in `licenses/`. Original notices remain in the corresponding source files.
