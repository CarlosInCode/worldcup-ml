"""Modelo de CÓRNERS: total de córners del partido (regresión de Poisson).

Features: ritmo reciente de córners a favor y en contra de cada equipo.
Requiere la tabla Gold `corner_features` (necesita el detalle: wc seed --scope national).
"""

from __future__ import annotations

from worldcup.predictor.poisson_count import train_poisson_count

_FEATURES = [
    "home_corners_for", "home_corners_against",
    "away_corners_for", "away_corners_against",
    "home_shots_for", "away_shots_for",  # volumen ofensivo -> genera córners
    "elo_diff",                            # dominancia
]
_MODEL_PATH = "models/model_corners.pkl"


def train_corners(test_frac: float = 0.2) -> dict:
    return train_poisson_count(
        "corner_features",
        _FEATURES,
        "total_corners",
        _MODEL_PATH,
        require_features=["home_corners_for", "away_corners_for", "home_shots_for"],
        test_frac=test_frac,
    )
