---
title: World Cup ML 2026
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.58.0
app_file: streamlit_app.py
pinned: false
---

# ⚽ World Cup ML — Plataforma de datos + predicciones para el Mundial FIFA 2026

Predice resultados, tarjetas, córners, over/under y stats por jugador a partir del
historial de api-football. Dos componentes sobre una arquitectura **medallón**
(Bronze → Silver → Gold) y modelos de ML + simulación Monte Carlo.

## Arquitectura

```
api-football ──▶ BRONZE ──▶ SILVER ──▶ GOLD ──▶ MODELOS ──▶ predicción / gráficas
              (crudo JSONL) (tablas   (features  (XGBoost +
                            limpias)   sin leak)  Poisson)

  └──────── dataplatform (crea el dataset) ────┘ └──── predictor (predice) ────┘
```

- **Bronze** (`data/bronze/`): JSON crudo del API, inmutable. Nunca se borra.
- **Silver** (`warehouse.duckdb` → `matches`): tablas limpias y tipadas.
- **Gold** (`warehouse.duckdb` → `match_features`): features listos para ML, **sin data leakage**.

## Stack

Python 3.12 · `uv` · httpx · DuckDB · Polars · XGBoost · statsmodels (Poisson) ·
MLflow · FastAPI (próx.) · Streamlit (próx.). Todo local, cero infra.

## Notas de entorno (macOS) ⚠️

- **Siempre** sincroniza con los extras: `uv sync --extra ml --extra viz --extra dev`
  (o `make setup`). Un `uv sync` pelado **desinstala** los extras de ML.
- XGBoost en Mac necesita OpenMP: `brew install libomp` (solo una vez).
- El comando se llama `wc`, que **colisiona con el `wc` de Unix**. Dentro del proyecto,
  `wc` es el nuestro; para el de Unix usa `/usr/bin/wc`.

## Arranque rápido

```bash
# 1. Instalar dependencias (¡con los extras!)
make setup

# 2. Configurar credenciales
cp .env.example .env        # y pon tu API_FOOTBALL_KEY

# 3. Catálogo de ligas (necesario para `seed`)
uv run wc ingest-leagues

# 4. Ingesta guiada. Empieza barato (solo resultados de selecciones):
uv run wc seed --no-detail
#    Luego el detalle, respetando tu cuota diaria (reanudable, retoma mañana):
uv run wc seed --scope national --max-calls 7000

# 5. Construir dataset y entrenar
uv run wc build-silver && uv run wc build-features && uv run wc train

# 6. Predecir un partido (team_ids de api-football)
uv run wc predict --home 50 --away 33
```

## Comandos

| Comando | Capa | Qué hace |
|---------|------|----------|
| `wc ingest-leagues` | Bronze | Cataloga ligas/competiciones (¡primero!) |
| `wc seed [--scope national\|clubs\|all]` | Bronze | **Ingesta guiada**: resuelve ligas por nombre y baja todo con TOPE de llamadas. Reanudable. |
| `wc ingest-fixtures --league N --season Y` | Bronze | Trae partidos crudos de una liga |
| `wc backfill --league N --season Y` | Bronze | Backfill completo reanudable de UNA liga/temporada |
| `wc build-silver` | Silver | Normaliza → tabla `matches` |
| `wc build-features` | Gold | Calcula Elo, descanso… → `match_features` |
| `wc train` | ML | Entrena 1X2 (validación temporal) |
| `wc train-cards` | ML | Tarjetas (Poisson + árbitro) |
| `wc train-corners` | ML | Córners (Poisson) |
| `wc train-player-shots` | ML | Tiros a puerta por jugador (requiere endpoint `players`) |
| `wc predict --home A --away B [--referee N]` | ML | Predice 1X2, goles, **tarjetas y córners** |
| `wc predict-player --player ID` | ML | Tiros a puerta esperados de un jugador |
| `wc dashboard` | VIZ | Lanza el **dashboard Streamlit** (predicciones con gráficas + exploración) |
| `wc check` | CALIDAD | Valida tablas Silver/Gold (puertas de calidad) |
| `wc pipeline` | ORQUESTACIÓN | Corre todo: Silver→calidad→Gold→calidad→entrenar modelos |

## ⚠️ Reglas de oro (no negociables)

1. **Nunca borres Bronze.** Re-procesas Silver/Gold sin gastar cuota del API.
2. **Cero data leakage.** Todo feature usa solo info anterior al pitazo inicial.
3. **Validación temporal**, nunca aleatoria. Mezclar fechas = métricas mentirosas.
4. **Métricas de probabilidad** (log-loss, Brier, calibración), no solo accuracy.
5. **Benchmark contra las cuotas** de las casas. Si no les ganas, lo sabrás.

## Roadmap

- [x] **Fase 0** — Esqueleto, cliente API (rate limit + reintentos + paginación).
- [x] **Fase 1** — Ingesta histórica → Bronze. *(fixtures, statistics, events, lineups,
      players, odds, teams, standings; backfill reanudable. Pendiente: ranking FIFA externo)*
- [x] **Fase 2** — Silver. Tablas `matches`, `match_statistics`, `match_events`,
      `player_match_stats`. *(verificado: 10.172 partidos reales de selecciones)*
- [x] **Fase 3** — Gold enriquecido + modelos. 1X2 con Elo + **forma reciente + H2H** +
      descanso (log_loss 0.92→0.89). *(pendiente: cuotas como feature)*
- [x] **Fase 4** — Modelos de conteo (Poisson genérico): **tarjetas** (+árbitro), **córners**,
      **tiros a puerta por jugador**. Integrados en `wc predict` (esperados + líneas over/under).
      *(córners/tarjetas necesitan más detalle para superar al baseline; tiros requiere endpoint `players`)*
- [x] **Fase 5** — **Dashboard Streamlit** con gráficas: predicción de partido (donut 1X2,
      mapa de calor de marcadores, barras de tarjetas/córners), ranking Elo, exploración del
      dataset. `wc dashboard`. *(pendiente opcional: API FastAPI on-demand)*
- [x] **Fase 6** — Orquestación + calidad + tracking. `wc pipeline` (grafo Silver→calidad→
      Gold→calidad→modelos con puertas de calidad), `wc check` (validaciones de datos),
      MLflow en SQLite (cada entrenamiento queda registrado). *(Dagster = upgrade futuro;
      hoy bloqueado por Python 3.14)*

### Mejoras de precisión / ingeniería (implementadas)

- **Sedes neutrales** — sin ventaja de local en torneos a sede única (Mundial). `predict --neutral`.
- **Importancia de competición** + **strength-of-schedule** como features del 1X2.
- **Goles ataque/defensa** (Dixon-Coles simplificado) alimentando la simulación.
- **Calibración isotónica** + **walk-forward CV** (TimeSeriesSplit) + **tuning** en el 1X2.
- **Champion/challenger** (`models/registry.json`): promociona solo si mejora el log-loss.
- **Constantes centralizadas** (`common/constants.py`) + **tests** (`pytest`, 11 verdes).
- **Pendiente con razón**: builds incrementales de Silver (prematuro a 10k filas) y
  benchmark vs cuotas (sin datos de odds históricos útiles).

### Operación diaria

```bash
# 1. Seguir nutriendo el detalle (reanudable):
wc seed --scope national --endpoints statistics,events,players --max-calls 70000
# 2. Reconstruir y reentrenar todo con puertas de calidad:
wc pipeline
# 3. Ver evolución de los modelos:
mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://localhost:5000
# 4. Explorar predicciones:
wc dashboard
```

## El Mundial 2026 (importante)

No entrenes solo con Mundiales (pocos partidos, formato nuevo de 48 equipos). Entrena
con **ligas de clubes + clasificatorias + selecciones** de los últimos años para que el
modelo aprenda la fuerza de equipos/jugadores, y aplícalo a los partidos del Mundial.

## Estructura

```
src/worldcup/
├── common/        config.py (settings), db.py (DuckDB)
├── dataplatform/  api_client.py, bronze.py, silver.py, features.py
├── predictor/     train.py, simulate.py (Monte Carlo), predict.py
└── cli.py         interfaz de línea de comandos (typer)
```
