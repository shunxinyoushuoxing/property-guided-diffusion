# Inference input contract

Run commands from the repository root. Relative input paths are resolved from the working directory. Keep user inputs under `private_inputs/`, which is ignored by Git. The inference configuration references these inputs; it does not embed training settings.

## Generator

`checkpoint` is a trusted PyTorch file containing a `state_dict` or the state dictionary itself. Parameter names match the joint discrete-type diffusion model. Loading is strict: missing parameters, unexpected parameters, and shape differences are errors. The loader does not read a training dataset, instantiate a trainer, or use training hyperparameters embedded in a checkpoint.

`model_metadata` points to JSON with:

- `architecture.time_dim`: dimension of the sinusoidal timestep embedding.
- `architecture.steps`: number of continuous diffusion steps.
- `architecture.discrete_steps`: number of discrete diffusion steps.
- `architecture.transition`: `uniform` or `absorbing`.
- `architecture.decoder`: inference architecture arguments for `CSPNet`, including `hidden_dim`, `num_layers`, `num_freqs`, `ln`, and any nondefault decoder options. Do not include `latent_dim`, `pred_type`, `smooth`, `discrete_type`, or `num_type_classes`; the loader sets these consistently.
- `architecture.beta`: scheduler arguments, including `scheduler_mode` and optional `beta_start`/`beta_end`.
- `architecture.sigma`: `sigma_begin` and `sigma_end`.
- `vocabulary`: ordered integer atomic numbers, with reserved mask token `0` at index zero. This order must match the checkpoint's embedding and output classes. It cannot be inferred from tensor shapes alone.
- Optional `position_rules`: objects with `axis` (0, 1 or 2), `lower`, `upper`, and `allowed_indices`. Bounds are fractional coordinates; the upper bound is exclusive. Rules restrict predicted categories in those regions. An unresolved mask category in a final structure makes that candidate invalid.

Scheduler buffers, including coordinate score normalization, are restored from the checkpoint. Architecture metadata must correspond to that checkpoint; inventing an element order or schedule will not reproduce its behavior.

`symmetry` points to JSON containing `num_atoms` and optionally `spacegroup` (default 1). P1 uses identity symmetry without any reference data. Other space groups require `anchor_index` of length N and `operations` of shape N x 4 x 4. Each operation is a homogeneous fractional-coordinate transform from its anchor to a site, and each anchor must point to itself. These constraints must match the model's intended sampling domain.

Optional `initial_checkpoint` and `initial_model_metadata` use another frozen generator for initialization. It must have the same vocabulary and support the same symmetry/site specification as the iterative model.

## Predictors

`predictor_config` points to a JSON object with `backend` (`mace` or `equiformer_v2`), `checkpoint`, and `device` (`cpu` or a CUDA device). Files are not downloaded automatically.

MACE expects a serialized `MACE`/`ScaleShiftMACE` model compatible with the bundled runtime. Optional `head` selects an output head; it is mandatory if the model has multiple heads. Optional `dtype` defaults to `float64`. The adapter builds periodic neighbor graphs, runs energy inference, and preserves the caller's default tensor dtype.

Equiformer V2 expects a checkpoint containing `config.model_attributes`, `state_dict`, and `normalizers.target.mean`/`std`. It strips distributed `module.` prefixes, loads the architecture strictly, disables force regression, and converts normalized predictions back to energies. Optional `batch_size`, `radius`, and `max_neighbors` default to 4, 6.0 and 200. These graph settings must match those used by the supplied predictor.

Only deserialize checkpoints from a trusted source. No trained checkpoint is included here.

The inspected runtime environments use Python 3.10, PyTorch 2.4, and different e3nn versions: MACE uses e3nn 0.4.4 and Equiformer V2 uses e3nn 0.5.1. Run them in separate environments when necessary. Add `predictor_python` to the main input; its value is an absolute Python executable path. A persistent worker exchanges structures and energies over JSON lines without invoking a shell. This also avoids reloading a predictor for every candidate.

## Property objectives

Without `property_recipe`, total predicted energy is minimized by default. This is a generic runnable objective, not a substitute for a task-specific property definition. For varying compositions, define an appropriate referenced property rather than comparing unrelated absolute energies.

To evaluate a derived property, provide `property_recipe` as a JSON path. All reference values, geometry offsets, species and sites are supplied by the user:

- `mode: energy`: optional `per_atom` divides energy by the site count.
- `mode: path_barrier`: build `states` at each evaluation site, predict energies, add optional `energy_offsets`, then take the maximum energy minus the energy at `reference_state` (default 0).
- `mode: linear_steps`: compute `coefficients @ energies + step_offsets`, reduce with `step_reduction`, and subtract `reference_value`.

For derived properties, `site_indices` specifies evaluation sites (default all). Each entry of `states` specifies appended atom `numbers` and 3D `offsets` relative to that site. `coordinates` is `cartesian` by default, or `fractional`. An empty object represents the unchanged structure. `coefficients` has one column per state. `site_reduction` combines site-level results and defaults to `mean`. Reducers are `mean`, `min`, `max` and `min_abs`; choose the one that defines the intended property. Geometry generation and reference values are not universal and must be specified for the intended inference problem.

## Geometry postprocessing

Optional `template_mapping` points to JSON with `template` (an ASE-readable structure), `max_distance`, and `groups`. Each group has `atomic_numbers`, `template_indices`, and optional `allow_vacancies`. The mapper assigns generated atoms to compatible template sites by minimum-cost periodic distance. Unassigned grouped sites become vacancies only when enabled; sites outside groups remain fixed. All generated atoms must be covered exactly once. The template and generated structure must already use a consistent cell, origin and orientation. No system-specific geometry or template is distributed.

## Search controls and outputs

`search` includes `population`, `candidates`, `iterations`, `noise_start`, optional `noise_end`, `step_lr`, `seed`, `min_distance`, and `max_initial_attempts`. Noise steps must be between 1 and `architecture.steps - 1`. Initial candidates are sampled unconditionally until the valid population is full or the attempt limit is reached.

The `objective` is `minimize`, `maximize`, or `target` with a numeric `target`. Greedy family selection replaces a parent only when one of its offspring has a higher reward. `resampling` is `none`, `standard`, or `diverse`. `beta` is inverse temperature/selection pressure, `elites` is the number of retained top members, and `sharing_radius` controls composition-space crowding. With rewards r and composition vectors c, selection probabilities are proportional to `exp(beta * r) / density`, where `density_i = sum_j max(0, 1 - ||c_i-c_j|| / sharing_radius)`.

The original population loop draws a new forward perturbation for each child (`anchor_mode: independent`). `shared` draws one forward anchor for each parent and reuses it for all reverse trajectories. The release exposes this distinction explicitly. Random-number consumption and results depend on this setting.

The raw generated state is retained for the next diffusion step; postprocessed structures are used for property scoring and output. `best.vasp` tracks the best candidate ever evaluated, even if resampling loses that candidate. Final population structures and values are saved separately. The output directory must not already exist, avoiding accidental replacement of a previous run.
