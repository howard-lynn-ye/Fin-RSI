#!/usr/bin/env python3
"""MUTABLE CANDIDATE OPERATOR SLOT (fin_rsi_multimodal_alpha_campaign).

RSI agents evolve this module across Generations Gen-1 -> Gen-2 -> Gen-3 while keeping
`frozen_harness.py` strictly read-only and cryptographically locked by `HARNESS_LOCK.json`
(Rule 21.1). Zero access to ground-truth forward return labels or arm identifiers.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict

import torch
import torch.nn as nn

FIN_SKILLS_ROOT = pathlib.Path("/usr/local/google/home/shwaihe/fin-skills")
if str(FIN_SKILLS_ROOT) not in sys.path:
    sys.path.insert(0, str(FIN_SKILLS_ROOT))

from fin_skills.fin_rsi import (
    StreamingWoodburyFisherOperator,
    SubspacePrecisionSteinOperator,
    ValueSpaceBoundedESSOperator,
)


class CandidateRSIOperator(nn.Module):
    """Unified wrapper over the 3-Generation Financial RSI Operator Evolution trajectory.

    Generations:
    - `Gen-1` (`Row 5: RSI_Gen1_ValueSpace_BoundedESS`):
        Primitives 1 & 4 — Post-Encoder Value-Space Salience Pooling with temperature-bounded
        activation w_t = exp(tau * tanh(z_t / tau)) (tau = 2.5) and Log-Sum-Exp
        partition-conserving intraday burst pooling.
    - `Gen-2` (`Row 6: RSI_Gen2_Subspace_Precision_Stein`):
        Primitives 3 & 9 — Adds 16D Per-Channel Subspace Precision Gate
        alpha_d(n_eff, |q_soc - q_ann|) + Positive-Part James-Stein MMSE Empirical-Bayes
        shrinkage toward the fundamental announcement prior.
    - `Gen-3` (`Row 7: RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC`):
        Primitives 7 & 9 — Adds Online Streaming Rank-1 Sherman-Morrison-Woodbury
        inverse-covariance adaptation Sigma_t^{-1} + Bernoulli Fisher Information Precision
        Gate g_Fisher(p_t) = 4 * p_t * (1 - p_t) coupled with 64-KC associative plasticity
        and 80th-percentile volatility gating.
    """

    def __init__(self, generation: str = "gen3", embed_dim: int = 16, tau: float = 2.5) -> None:
        super().__init__()
        self.generation = generation.lower()
        self.embed_dim = embed_dim
        self.pooler = ValueSpaceBoundedESSOperator(embed_dim=embed_dim, tau=tau)
        self.subspace_stein = SubspacePrecisionSteinOperator(
            embed_dim=embed_dim, tau=tau, c_stein=0.28
        )
        self.woodbury_fisher = StreamingWoodburyFisherOperator(
            embed_dim=embed_dim, kc_dim=64, top_k_kc=8, tau=tau, c_stein=0.28
        )

    def forward(
        self,
        seq_social_emb: torch.Tensor,
        salience_logits: torch.Tensor,
        burst_block_sizes: torch.Tensor,
        announcement_prior_emb: torch.Tensor,
        eff_sample_count: torch.Tensor,
        jev_calibrated_prob: torch.Tensor,
        volatility_gate_mask: torch.Tensor,
        noise_var: float = 0.35,
    ) -> Dict[str, torch.Tensor]:
        """Run the specified RSI generation operator without access to ground-truth returns."""
        pooled_soc, pool_diag = self.pooler(
            seq_embeddings=seq_social_emb,
            salience_logits=salience_logits,
            burst_block_sizes=burst_block_sizes,
        )

        if self.generation == "gen1":
            # Scalar empirical-Bayes shrinkage on top of Gen-1 Value-Space bounded pooling
            n_eff = (
                eff_sample_count.unsqueeze(-1)
                if eff_sample_count.ndim == 1
                else eff_sample_count
            ).to(dtype=pooled_soc.dtype)
            scalar_alpha = n_eff / (n_eff + 3.20)
            rep = scalar_alpha * pooled_soc + (1.0 - scalar_alpha) * announcement_prior_emb
            return {
                "representation": rep,
                "ess": pool_diag["ess"],
            }

        if self.generation == "gen2":
            stein_emb, stein_diag = self.subspace_stein(
                social_emb=pooled_soc,
                announcement_prior_emb=announcement_prior_emb,
                eff_sample_count=eff_sample_count,
                noise_var=noise_var,
            )
            return {
                "representation": stein_emb,
                "ess": pool_diag["ess"],
                "alpha_d": stein_diag["alpha_d"],
            }

        # Gen-3 Champion: Streaming Rank-1 Woodbury + Bernoulli Fisher Gate + 64-KC Plasticity
        champ_emb, champ_diag = self.woodbury_fisher(
            social_emb=pooled_soc,
            announcement_prior_emb=announcement_prior_emb,
            eff_sample_count=eff_sample_count,
            jev_calibrated_prob=jev_calibrated_prob,
            volatility_gate_mask=volatility_gate_mask,
            noise_var=noise_var,
            update_covariance=True,
        )
        return {
            "representation": champ_emb,
            "ess": pool_diag["ess"],
            "alpha_d": champ_diag["alpha_d"],
            "fisher_gate": champ_diag["fisher_gate"],
            "kc_sparse_features": champ_diag["kc_sparse_features"],
        }


def build_mutable_rsi_generations(embed_dim: int = 16) -> Dict[str, CandidateRSIOperator]:
    """Instantiate the 3 evolved Financial RSI operator generations (`Row 5..7`)."""
    return {
        "Row_5_RSI_Gen1_ValueSpace_BoundedESS": CandidateRSIOperator(
            generation="gen1", embed_dim=embed_dim, tau=2.5
        ),
        "Row_6_RSI_Gen2_Subspace_Precision_Stein": CandidateRSIOperator(
            generation="gen2", embed_dim=embed_dim, tau=2.5
        ),
        "Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC": CandidateRSIOperator(
            generation="gen3", embed_dim=embed_dim, tau=2.5
        ),
    }
