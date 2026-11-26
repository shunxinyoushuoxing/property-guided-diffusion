"""D3PM: Discrete Denoising Diffusion with Absorbing / Uniform Transition.

Supports two transition kernels:
  - 'absorbing': all states flow to mask (index 0), stationary = all-mask
  - 'uniform':   all states flow to uniform distribution, stationary = Cat(1/K)

References:
    - Austin et al., 2021 (D3PM)
    - Chemeleon-DNG: chemeleon_dng/diffusion/models/d3pm.py
    - MatterGen: mattergen/diffusion/d3pm/d3pm.py
"""
import torch
import torch.nn.functional as F


def reciprocal_beta_schedule(timesteps):
    """Reciprocal schedule: beta_t = 1/(T-t+1), t=1..T"""
    t = torch.arange(1, timesteps + 1)
    return 1.0 / (timesteps - t + 1)


class D3PM:
    """D3PM for discrete atom types with configurable transition kernel.

    State space: {0, 1, ..., K-1}
      - 0 = mask/absorbing state (only used by 'absorbing' mode)
      - 1..K-1 = valid atom type indices

    Args:
        num_classes: K (including mask at index 0)
        timesteps: number of diffusion steps T
        transition_type: 'absorbing' or 'uniform'
    """

    def __init__(self, num_classes: int, timesteps: int = 256,
                 transition_type: str = 'absorbing'):
        self.K = num_classes
        self.T = timesteps
        self.transition_type = transition_type
        self.eps = 1e-6

        betas = reciprocal_beta_schedule(timesteps)
        self.betas = betas
        assert (betas > 0).all() and (betas <= 1).all()

        if transition_type == 'absorbing':
            get_mat = self._get_absorbing_transition_mat
        elif transition_type == 'uniform':
            get_mat = self._get_uniform_transition_mat
        else:
            raise ValueError(f"Unknown transition_type: {transition_type}")

        self.q_one_step_mats = torch.stack(
            [get_mat(t) for t in range(timesteps)],
            dim=0,
        )

        q_mat_t = self.q_one_step_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, timesteps):
            q_mat_t = q_mat_t @ self.q_one_step_mats[t]
            q_mats.append(q_mat_t)
        self.q_mats = torch.stack(q_mats, dim=0)  # [T, K, K]

        self.q_one_step_transposed = self.q_one_step_mats.transpose(1, 2)

    def _get_absorbing_transition_mat(self, t: int):
        """Absorbing-state transition matrix Q_t.

        Q_t[i,j] = (1 - beta_t) if i == j
                  = beta_t      if j == 0 (mask)
                  = 0           otherwise
        """
        beta_t = self.betas[t].float()
        mat = (1 - beta_t) * torch.eye(self.K)
        mat[:, 0] += beta_t
        return mat

    def _get_uniform_transition_mat(self, t: int):
        """Uniform transition matrix Q_t.

        Q_t[i,j] = (1 - beta_t) + beta_t/K   if i == j
                  = beta_t/K                   if i != j

        Stationary distribution is uniform Cat(1/K) over all K classes.
        """
        beta_t = self.betas[t].float()
        mat = (1 - beta_t) * torch.eye(self.K) + (beta_t / self.K) * torch.ones(self.K, self.K)
        return mat

    def _at(self, a: torch.Tensor, t_per_node: torch.Tensor, x: torch.Tensor):
        """Index into matrix batch a by (t, x) to get row vectors.

        Args:
            a: [T, K, K]
            t_per_node: [N] timesteps (1-indexed)
            x: [N] category indices

        Returns:
            [N, K]
        """
        a = a.to(x.device)
        t_idx = torch.clamp(t_per_node - 1, min=0)
        return a[t_idx, x, :]

    def q_sample(self, x_start: torch.Tensor, t_per_node: torch.Tensor,
                 noise: torch.Tensor = None):
        """Forward diffusion: sample x_t from x_0.

        q(x_t | x_0) = Cat(x_t; p = one_hot(x_0) @ Q_bar_t)
        Uses Gumbel-max trick for sampling.

        Args:
            x_start: [N] integer category indices (1..K-1 for real data)
            t_per_node: [N] timesteps (1..T)
            noise: [N, K] optional uniform noise

        Returns:
            x_t: [N] noisy discrete categories
        """
        logits = torch.log(
            self._at(self.q_mats, t_per_node, x_start) + self.eps
        )

        if noise is None:
            noise = torch.rand(x_start.shape[0], self.K, device=x_start.device)
        noise = torch.clamp(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))

        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(self, x_start, x_t: torch.Tensor,
                           t_per_node: torch.Tensor,
                           is_x_start_logits: bool = False):
        """Compute posterior q(x_{t-1} | x_t, x_0) unnormalized log-probs.

        q(x_{t-1}|x_t, x_0) ~ q(x_t|x_{t-1}) * q(x_{t-1}|x_0)

        fact1 = Q_t^T[x_t, :] = q(x_t | x_{t-1}=j) for all j
        fact2 = softmax(x_start_logits) @ Q_bar_{t-1} = q(x_{t-1}=j | x_0)

        Args:
            x_start: [N] true category (training) or [N, K] logits (inference)
            x_t: [N] current state
            t_per_node: [N] timesteps (1..T)
            is_x_start_logits: whether x_start is logits

        Returns:
            posterior_logits: [N, K]
        """
        if is_x_start_logits:
            x_start_logits = x_start
        else:
            x_start_logits = torch.log(
                F.one_hot(x_start, self.K).float() + self.eps
            )

        fact1 = self._at(self.q_one_step_transposed, t_per_node, x_t)  # [N, K]

        softmaxed = torch.softmax(x_start_logits, dim=-1)  # [N, K]
        q_mats_device = self.q_mats.to(x_start_logits.device)

        t_idx = torch.clamp(t_per_node - 2, min=0)  # [N]
        qmats_t = q_mats_device[t_idx]  # [N, K, K]
        fact2 = torch.einsum("bc, bcd -> bd", softmaxed, qmats_t)  # [N, K]

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t_per_node.unsqueeze(-1)  # [N, 1]
        return torch.where(t_broadcast == 1, x_start_logits, out)



    @torch.no_grad()
    def p_sample(self, pred_x_start_logits: torch.Tensor,
                 x_t: torch.Tensor, t_per_node: torch.Tensor,
                 mask: torch.Tensor = None):
        """Reverse sampling step: x_t -> x_{t-1} via Gumbel-max.

        Args:
            pred_x_start_logits: [N, K] network output logits
            x_t: [N] current state
            t_per_node: [N] current timestep
            mask: [N, K] position mask (True=allowed, inference only)

        Returns:
            x_{t-1}: [N] sampled result
        """
        if mask is not None:
            pred_x_start_logits = pred_x_start_logits.masked_fill(~mask, -1e9)

        pred_posterior_logits = self.q_posterior_logits(
            pred_x_start_logits, x_t, t_per_node, is_x_start_logits=True
        )

        noise = torch.rand_like(pred_posterior_logits)
        noise = torch.clamp(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))

        nonzero_mask = (t_per_node != 1).float().unsqueeze(-1)  # [N, 1]

        sample = torch.argmax(
            pred_posterior_logits + gumbel_noise * nonzero_mask, dim=-1
        )
        return sample
