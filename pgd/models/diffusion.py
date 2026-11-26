"""Frozen joint diffusion model. No dataset loader or training machinery."""
import math
import torch
from torch import nn
from torch_scatter import scatter
from .cspnet import CSPNet
from .discrete import D3PM
from .schedules import BetaScheduler, SigmaScheduler
from .lattice.crystal_family import CrystalFamily
from .vocabulary import Vocabulary

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        frequencies = torch.exp(torch.arange(half, device=time.device) * (-math.log(10000) / (half - 1)))
        phases = time[:, None] * frequencies[None, :]
        return torch.cat((phases.sin(), phases.cos()), dim=-1)

class CrystalDiffusion(nn.Module):
    def __init__(self, architecture, vocabulary, position_rules=()):
        super().__init__()
        self.vocabulary = Vocabulary(vocabulary, position_rules)
        self.num_classes = len(vocabulary)
        self.z_to_idx = self.vocabulary.z_to_index
        self.idx_to_z = dict(enumerate(vocabulary))
        cfg = dict(architecture)
        self.time_dim = int(cfg['time_dim'])
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.d3pm = D3PM(self.num_classes, cfg['discrete_steps'], transition_type=cfg['transition'])
        self.register_buffer('d3pm_q_mats', self.d3pm.q_mats)
        self.register_buffer('d3pm_q_one_step_transposed', self.d3pm.q_one_step_transposed)
        self.decoder = CSPNet(**dict(cfg['decoder'], latent_dim=self.time_dim,
                              pred_type=True, smooth=False, discrete_type=True,
                              num_type_classes=self.num_classes))
        self.beta_scheduler = BetaScheduler(cfg['steps'], **cfg['beta'])
        self.sigma_scheduler = SigmaScheduler(cfg['steps'], **cfg['sigma'], initialize_norm=False)
        self.crystal_family = CrystalFamily()

    @property
    def device(self):
        return self.d3pm_q_mats.device

    def _sync_d3pm_device(self):
        self.d3pm.q_mats = self.d3pm_q_mats
        self.d3pm.q_one_step_transposed = self.d3pm_q_one_step_transposed

    @classmethod
    def from_checkpoint(cls, checkpoint, metadata, device='cpu'):
        # Full serialized checkpoints must come from a trusted source.
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = payload.get('state_dict', payload)
        model = cls(metadata['architecture'], metadata['vocabulary'], metadata.get('position_rules', []))
        model.load_state_dict(state, strict=True)
        model.to(device).eval().requires_grad_(False)
        model._sync_d3pm_device()
        return model

    @torch.no_grad()
    def sample(self, batch, step_lr=1e-5,
               init_coords=None, init_lattice=None, init_types=None,
               deterministic=False, d3pm_argmax_last_k=0,
               time_start=None, record_trajectory=False):
        self._sync_d3pm_device()
        batch_size = batch.num_graphs

        # === Initialization (supports external initial noise) ===
        if init_coords is not None:
            x_T = init_coords.to(self.device)
        else:
            x_T = torch.rand([batch.num_nodes, 3]).to(self.device)

        # Save pre-expansion coordinate noise for return
        x_T_original = x_T.clone().cpu()

        if init_lattice is not None:
            crys_fam_T_raw = init_lattice.clone().cpu()
            crys_fam_T = init_lattice.to(self.device)
            crys_fam_T = self.crystal_family.proj_k_to_spacegroup(crys_fam_T, batch.spacegroup)
        else:
            crys_fam_T = torch.randn([batch_size, 6]).to(self.device)
            crys_fam_T_raw = crys_fam_T.clone().cpu()
            crys_fam_T = self.crystal_family.proj_k_to_spacegroup(crys_fam_T, batch.spacegroup)

        # D3PM initialization
        if init_types is not None:
            a_T = init_types.to(self.device)
        elif self.d3pm.transition_type == 'uniform':
            a_T = torch.randint(1, self.d3pm.K, (batch.num_nodes,), device=self.device)
        else:
            a_T = torch.zeros(batch.num_nodes, dtype=torch.long, device=self.device)
        a_T_raw = a_T.clone().cpu()

        # Coordinate symmetry expansion (same as original)
        l_T = self.crystal_family.v2m(crys_fam_T)

        x_T_all = torch.cat([
            x_T[batch.anchor_index],
            torch.ones(batch.ops.size(0), 1).to(self.device)
        ], dim=-1).unsqueeze(-1)
        x_T = (batch.ops @ x_T_all).squeeze(-1)[:, :3] % 1.

        if time_start is None:
            time_start = self.beta_scheduler.timesteps - 1

        if not 1 <= time_start < self.beta_scheduler.timesteps:
            raise ValueError("time_start must be in [1, timesteps - 1]")

        a_t = a_T
        x_t = x_T
        crys_fam_t = crys_fam_T
        l_t = l_T

        traj = {time_start: {
            'num_atoms': batch.num_atoms,
            'atom_types': a_t,
            'frac_coords': x_t % 1.,
            'lattices': l_t,
            'crys_fam': crys_fam_t,
        }}

        for t in range(time_start, 0, -1):
            times = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            # === Corrector (coords only, same as original) ===
            if deterministic:
                rand_x = torch.zeros_like(x_T)
            else:
                rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            pred_crys_fam, pred_x, _ = self.decoder(
                time_emb, a_t, x_t, crys_fam_t, batch.num_atoms, batch.batch
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)

            pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce='mean')[batch.anchor_index]
            pred_x_corr = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

            x_t_minus_05 = x_t - step_size * pred_x_corr + std_x * rand_x

            crys_fam_t_minus_05 = crys_fam_t

            frac_coords_all = torch.cat([
                x_t_minus_05[batch.anchor_index],
                torch.ones(batch.ops.size(0), 1).to(x_t_minus_05.device)
            ], dim=-1).unsqueeze(-1)
            x_t_minus_05 = (batch.ops @ frac_coords_all).squeeze(-1)[:, :3] % 1.

            a_t_minus_05 = a_t

            # === Predictor (all three) ===
            if deterministic:
                rand_crys_fam = torch.zeros_like(crys_fam_T)
                rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)
            else:
                rand_crys_fam = torch.randn_like(crys_fam_T)
                rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)
            ori_crys_fam = crys_fam_t_minus_05
            if deterministic:
                rand_x = torch.zeros_like(x_T)
            else:
                rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size_pred = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x_pred = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            pred_crys_fam, pred_x, type_logits = self.decoder(
                time_emb, a_t_minus_05, x_t_minus_05,
                crys_fam_t_minus_05, batch.num_atoms, batch.batch
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)

            # Lattice update
            crys_fam_t = c0 * (ori_crys_fam - c1 * pred_crys_fam) + sigmas * rand_crys_fam
            crys_fam_t = self.crystal_family.proj_k_to_spacegroup(crys_fam_t, batch.spacegroup)

            # Coordinate update
            pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce='mean')[batch.anchor_index]
            pred_x_final = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

            x_t = x_t_minus_05 - step_size_pred * pred_x_final + std_x_pred * rand_x

            frac_coords_all = torch.cat([
                x_t[batch.anchor_index],
                torch.ones(batch.ops.size(0), 1).to(x_t.device)
            ], dim=-1).unsqueeze(-1)
            x_t = (batch.ops @ frac_coords_all).squeeze(-1)[:, :3] % 1.

            # Type update (D3PM p_sample or argmax for last k steps)
            type_logits = scatter(type_logits, batch.anchor_index, dim=0, reduce='mean')[batch.anchor_index]

            d3pm_t_step = int(t * self.d3pm.T / self.beta_scheduler.timesteps) + 1
            d3pm_t_step = min(d3pm_t_step, self.d3pm.T)
            d3pm_t_per_node = torch.full(
                (batch.num_nodes,), d3pm_t_step, dtype=torch.long, device=self.device
            )

            position_mask = self.vocabulary.position_mask(x_t_minus_05)

            use_argmax = (d3pm_argmax_last_k > 0 and t <= d3pm_argmax_last_k)
            if use_argmax:
                if position_mask is not None:
                    type_logits = type_logits.masked_fill(~position_mask, -1e9)
                a_t = torch.argmax(type_logits, dim=-1)
            else:
                a_t = self.d3pm.p_sample(
                    type_logits, a_t_minus_05, d3pm_t_per_node, mask=position_mask
                )
            a_t = a_t[batch.anchor_index]

            l_t = self.crystal_family.v2m(crys_fam_t)

            if record_trajectory:
                traj[t - 1] = {
                'num_atoms': batch.num_atoms,
                'atom_types': a_t,
                'frac_coords': x_t % 1.,
                'lattices': l_t,
                'crys_fam': crys_fam_t,
            }

        # === Final result ===
        final_atom_types = self.vocabulary.decode(a_t)

        traj_stack = {}
        if record_trajectory:
            traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': final_atom_types,
            'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices': torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)]),
        }

        res = {
            'num_atoms': batch.num_atoms,
            'atom_types': final_atom_types,
            'frac_coords': x_t % 1.,
            'lattices': l_t,
            'init_coords': x_T_original,
            'init_lattice': crys_fam_T_raw,
            'init_types': a_T_raw,
        }

        return res, traj_stack
