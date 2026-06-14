"""Modelo de TARJETAS: total de tarjetas del partido (regresión de Poisson).

Feature estrella: el promedio de tarjetas del ÁRBITRO en partidos anteriores.
Requiere la tabla Gold `card_features` (necesita el detalle: wc seed --scope national).
"""

from __future__ import annotations

from worldcup.predictor.poisson_count import train_poisson_count

_FEATURES = ["ref_avg_cards", "home_avg_cards", "away_avg_cards"]
_MODEL_PATH = "models/model_cards.pkl"


def train_cards(test_frac: float = 0.2) -> dict:
    return train_poisson_count(
        "card_features",
        _FEATURES,
        "total_cards",
        _MODEL_PATH,
        require_features=["ref_avg_cards"],  # exige histórico del árbitro
        test_frac=test_frac,
    )
