"""Neural Stock-JEV Multi-Task Decision Network (`neural_jev.py`).

Implements a local PyTorch multi-task neural network adhering 100% to the
TypeSafe JEV decision wire contract (choice, score, noul).
Requires zero external API keys or remote network calls.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fin_skills.model_zoo.jev import (
    JevError,
    _description,
    _json_copy,
    _number,
    _questions,
    _response,
)
from fin_skills.jev.stock_jev import (
    REGIME_CHOICE_OPTIONS,
    ORDINAL_RATING_LEVELS,
    AnalyticalStockJEV,
)


class NeuralStockJEVNet(nn.Module):
    """Tri-Aspect Trunk + 3 Specialized Task Heads for Financial JEV Decisions.

    Trunk inputs:
      - x_macro (dim_macro=8): FX drift, yield slope, market turnover, market vol
      - x_stock (dim_stock=16): price momentum, valuation, earnings quality, spillover
      - x_micro (dim_micro=8): LOB depth, bid-ask spread, order imbalance, liquidity shock
    """

    def __init__(
        self,
        dim_macro: int = 8,
        dim_stock: int = 16,
        dim_micro: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.dim_macro = dim_macro
        self.dim_stock = dim_stock
        self.dim_micro = dim_micro
        self.hidden_dim = hidden_dim

        # Tri-aspect projections
        self.proj_macro = nn.Sequential(
            nn.Linear(dim_macro, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        self.proj_stock = nn.Sequential(
            nn.Linear(dim_stock, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.proj_micro = nn.Sequential(
            nn.Linear(dim_micro, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )

        # Cross-aspect interaction trunk
        total_in = hidden_dim // 2 + hidden_dim + hidden_dim // 2
        self.trunk = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Head 1: Macro Regime (4-class Choice)
        self.head_regime = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.SiLU(),
            nn.Linear(32, 4),
        )

        # Head 2: Disqualification (Binary Noul)
        self.head_disqualify = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # Prior initialization: disqualification base probability ~ 11.9%
        nn.init.constant_(self.head_disqualify[-1].bias, -2.0)

        # Head 3: Ordinal Quality Rating (5-level Score)
        self.head_ordinal = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 5),
        )


    def forward(
        self,
        x_macro: torch.Tensor,
        x_stock: torch.Tensor,
        x_micro: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass across all 3 JEV decision heads."""
        h_m = self.proj_macro(x_macro)
        h_s = self.proj_stock(x_stock)
        h_u = self.proj_micro(x_micro)

        # Head 1: Macro regime is driven primarily by macro + micro features
        h_regime_in = (h_m + h_u) / 2.0
        regime_logits = self.head_regime(h_regime_in)  # [B, 4]
        regime_probs = F.softmax(regime_logits, dim=-1)

        # Combined fused trunk
        fused = torch.cat([h_m, h_s, h_u], dim=-1)
        z = self.trunk(fused)  # [B, hidden_dim]

        # Head 2: Binary Noul disqualification probability
        disqualify_logit = self.head_disqualify(z).squeeze(-1)  # [B]
        disqualify_prob = torch.sigmoid(disqualify_logit)

        # Head 3: 5-level Ordinal rating
        ordinal_logits = self.head_ordinal(z)  # [B, 5]
        ordinal_probs = F.softmax(ordinal_logits, dim=-1)

        return {
            "regime_probs": regime_probs,
            "disqualify_prob": disqualify_prob,
            "ordinal_probs": ordinal_probs,
        }


class NeuralStockJEVAdapter:
    """Wire-compatible JEV Model adapter wrapping NeuralStockJEVNet."""

    def __init__(
        self,
        model: Optional[NeuralStockJEVNet] = None,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        if model is None:
            self.model = NeuralStockJEVNet().to(device=self.device, dtype=self.dtype)
            self.model.eval()
        else:
            self.model = model.to(device=self.device, dtype=self.dtype)

    def predict(self, state: Union[str, dict, list], *, questions: dict) -> dict:
        """Run neural forward pass and format output into valid JEV response format."""
        checked = _questions(questions)
        state_dict = state if isinstance(state, dict) else {"raw_state": state}

        # Extract or pad feature vectors
        dim_m = self.model.dim_macro
        dim_s = self.model.dim_stock
        dim_u = self.model.dim_micro

        raw_macro = state_dict.get("macro_vector", [
            state_dict.get("macro_fx_shock", 0.0),
            state_dict.get("vol_shock_20d", 0.0),
            state_dict.get("market_turnover_zscore", 0.0),
            state_dict.get("yield_slope", 0.0),
        ])
        raw_stock = state_dict.get("stock_vector", [
            state_dict.get("composite_alpha_score", 0.0),
            state_dict.get("momentum_20d", 0.0),
            state_dict.get("valuation_pe_zscore", 0.0),
            state_dict.get("supply_chain_spillover", 0.0),
        ])
        raw_micro = state_dict.get("micro_vector", [
            state_dict.get("lob_resilience_score", 0.5),
            state_dict.get("spread_bps", 10.0),
            state_dict.get("order_imbalance_ratio", 0.0),
            state_dict.get("tail_risk_zscore", 0.0),
        ])

        # Pad or trim to model dimensions
        vec_m = np.zeros(dim_m, dtype=np.float32)
        vec_s = np.zeros(dim_s, dtype=np.float32)
        vec_u = np.zeros(dim_u, dtype=np.float32)

        vec_m[:min(len(raw_macro), dim_m)] = raw_macro[:dim_m]
        vec_s[:min(len(raw_stock), dim_s)] = raw_stock[:dim_s]
        vec_u[:min(len(raw_micro), dim_u)] = raw_micro[:dim_u]

        t_m = torch.tensor(vec_m, dtype=self.dtype, device=self.device).unsqueeze(0)
        t_s = torch.tensor(vec_s, dtype=self.dtype, device=self.device).unsqueeze(0)
        t_u = torch.tensor(vec_u, dtype=self.dtype, device=self.device).unsqueeze(0)

        with torch.no_grad():
            preds = self.model(t_m, t_s, t_u)

        regime_p = preds["regime_probs"][0].cpu().numpy()
        disq_p = float(preds["disqualify_prob"][0].cpu().item())
        ord_p = preds["ordinal_probs"][0].cpu().numpy()

        answers = {}
        for q_id, q_def in checked.items():
            kind = q_def["type"]
            if kind == "choice":
                opts = list(q_def["criteria"].keys())
                # Map available options to probabilities
                if len(opts) == len(REGIME_CHOICE_OPTIONS) and all(o in REGIME_CHOICE_OPTIONS for o in opts):
                    raw_probs = [float(regime_p[REGIME_CHOICE_OPTIONS.index(o)]) for o in opts]
                else:
                    raw_probs = [1.0 / len(opts)] * len(opts)
                norm_probs = np.array(raw_probs) / sum(raw_probs)
                prob_dict = {opt: float(np.round(p, 5)) for opt, p in zip(opts, norm_probs)}
                diff = 1.0 - sum(prob_dict.values())
                prob_dict[opts[0]] = float(np.round(prob_dict[opts[0]] + diff, 5))

                best_choice = max(prob_dict.keys(), key=lambda k: prob_dict[k])
                confidence = float(np.max(list(prob_dict.values())))

                answers[q_id] = {
                    "type": "choice",
                    "choice": best_choice,
                    "confidence": confidence,
                    "probabilities": prob_dict,
                }
            elif kind == "noul":
                prob_val = min(max(disq_p, 0.0001), 0.9999)
                answers[q_id] = {
                    "type": "noul",
                    "noul": float(np.round(prob_val, 4)),
                    "confidence": float(np.round(max(prob_val, 1.0 - prob_val), 4)),
                }
            elif kind == "score":
                n_levels = len(q_def["criteria"])
                if n_levels == 5:
                    raw_p = [float(p) for p in ord_p]
                else:
                    raw_p = [1.0 / n_levels] * n_levels
                norm_p = np.array(raw_p) / sum(raw_p)
                prob_dict = {str(i): float(np.round(p, 5)) for i, p in enumerate(norm_p)}
                diff = 1.0 - sum(prob_dict.values())
                prob_dict["0"] = float(np.round(prob_dict["0"] + diff, 5))

                weighted_score = sum(i * prob_dict[str(i)] for i in range(n_levels))

                answers[q_id] = {
                    "type": "score",
                    "score": float(np.round(weighted_score, 4)),
                    "confidence": float(np.round(np.max(list(prob_dict.values())), 4)),
                    "probabilities": prob_dict,
                    "legend": {str(i): q_def["criteria"][i] for i in range(n_levels)},
                }

        response_payload = {
            "model": "neural-stock-jev-v1",
            "usage": {"input_tokens": 128, "output_tokens": 64},
            "answers": answers,
        }
        return _response(response_payload, checked)
