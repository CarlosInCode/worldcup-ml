"""Entrenador genérico de modelos de conteo (regresión de Poisson).

Tarjetas, córners y tiros a puerta son todos procesos de conteo: la misma técnica
(Poisson GLM) sirve para los tres. Este módulo evita duplicar el código de entrenamiento.

Buenas prácticas aplicadas:
  - Validación TEMPORAL (corte por fecha, no aleatorio).
  - Imputación de nulos con la media del TRAIN (no del test) -> sin leakage.
  - Métrica MAE comparada contra el baseline "predecir siempre la media".
"""

from __future__ import annotations

from pathlib import Path

from worldcup.common.db import connect


def _table_count(con, table: str) -> int:
    exists = con.sql(
        f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table}'"
    ).fetchone()[0]
    return int(con.sql(f"SELECT count(*) FROM {table}").fetchone()[0]) if exists else 0


def train_poisson_count(
    table: str,
    features: list[str],
    target: str,
    model_path: str | Path,
    *,
    require_features: list[str] | None = None,
    min_rows: int = 200,
    test_frac: float = 0.2,
) -> dict:
    """Entrena un Poisson GLM sobre `table` para predecir `target`. Devuelve métricas.

    require_features: filas sin estos features se descartan (p.ej. exige histórico de árbitro).
    """
    import numpy as np
    import statsmodels.api as sm

    with connect() as con:
        if not _table_count(con, table):
            raise ValueError(
                f"La tabla `{table}` no existe o está vacía. Baja el detalle "
                f"(wc seed --scope national) y reconstruye Gold (wc build-features)."
            )
        df = con.sql(
            f"SELECT * FROM {table} WHERE {target} IS NOT NULL ORDER BY kickoff_utc"
        ).pl()

    if require_features:
        df = df.drop_nulls(subset=require_features)
    if df.height < min_rows:
        raise ValueError(
            f"Solo {df.height} filas usables en `{table}` (mínimo {min_rows}). Baja más detalle."
        )

    cut = int(df.height * (1 - test_frac))
    train, test = df[:cut], df[cut:]

    import polars as pl

    # Imputación con la media del TRAIN (evita leakage del test). Casteamos a Float64
    # para garantizar un array numérico puro (statsmodels falla con dtype object).
    means = {f: (train[f].mean() or 0.0) for f in features}

    def _prep(d):
        return d.select(
            [pl.col(f).cast(pl.Float64).fill_null(means[f]) for f in features]
        ).to_numpy()

    X_train = sm.add_constant(_prep(train), has_constant="add")
    X_test = sm.add_constant(_prep(test), has_constant="add")
    y_train = train[target].cast(pl.Float64).to_numpy()
    y_test = test[target].cast(pl.Float64).to_numpy()

    model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    pred = model.predict(X_test)

    mae = float(np.mean(np.abs(pred - y_test)))
    baseline_mae = float(np.mean(np.abs(y_train.mean() - y_test)))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    metrics = {
        "table": table,
        "target": target,
        "n_train": int(train.height),
        "n_test": int(test.height),
        "mae": round(mae, 4),
        "baseline_mae_media": round(baseline_mae, 4),
        "media": round(float(y_train.mean()), 3),
        "mejora_vs_baseline": round(baseline_mae - mae, 4),
    }
    from worldcup.tracking import track

    track(f"wc-{target}", {"model": "PoissonGLM", "n_features": len(features)}, metrics)
    return metrics
