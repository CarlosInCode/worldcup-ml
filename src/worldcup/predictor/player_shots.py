"""Modelo de TIROS A PUERTA por JUGADOR en un partido (regresión de Poisson).

Features: ritmo reciente del jugador (tiros a puerta, tiros totales, minutos) + si juega
de local. Requiere `player_shot_features` -> necesita el endpoint `players`
(wc seed --scope national --endpoints players).
"""

from __future__ import annotations

from worldcup.predictor.poisson_count import train_poisson_count

_FEATURES = ["player_shots_on_avg", "player_shots_total_avg", "player_minutes_avg", "is_home"]
_MODEL_PATH = "models/model_player_shots.pkl"


def train_player_shots(test_frac: float = 0.2) -> dict:
    return train_poisson_count(
        "player_shot_features",
        _FEATURES,
        "shots_on",
        _MODEL_PATH,
        require_features=["player_shots_on_avg"],
        min_rows=500,  # a nivel jugador hay muchas más filas
        test_frac=test_frac,
    )
