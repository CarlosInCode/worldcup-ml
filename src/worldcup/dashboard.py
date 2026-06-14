"""Dashboard Streamlit: predicciones con gráficas + exploración del dataset.

Ejecuta con:  wc dashboard   (o:  streamlit run src/worldcup/dashboard.py)

Tres secciones (barra lateral):
  1. Predicción de partido — 1X2, mapa de calor de marcadores, tarjetas y córners.
  2. Ranking Elo — fuerza actual de las selecciones según nuestro Elo.
  3. Explorar dataset — volumen, distribución de resultados, goles.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from worldcup.common.db import connect
from worldcup.predictor.predict import predict_match

st.set_page_config(page_title="World Cup ML", page_icon="⚽", layout="wide")


# ───────────────────────── helpers (cacheados) ─────────────────────────


@st.cache_data
def team_options() -> dict[str, int]:
    """{nombre_equipo: team_id} a partir de la tabla matches."""
    with connect() as con:
        df = con.sql(
            """
            SELECT team_id, any_value(name) AS name FROM (
                SELECT home_team_id AS team_id, home_team AS name FROM matches
                UNION ALL
                SELECT away_team_id, away_team FROM matches
            ) GROUP BY team_id ORDER BY name
            """
        ).pl()
    return {r["name"]: r["team_id"] for r in df.iter_rows(named=True)}


@st.cache_data
def latest_elos() -> pl.DataFrame:
    with connect() as con:
        return con.sql(
            """
            WITH all_elo AS (
                SELECT home_team_id AS team_id, home_elo AS elo, kickoff_utc FROM match_features
                UNION ALL
                SELECT away_team_id, away_elo, kickoff_utc FROM match_features
            ), ranked AS (
                SELECT team_id, elo,
                       row_number() OVER (PARTITION BY team_id ORDER BY kickoff_utc DESC) rn
                FROM all_elo
            ), names AS (
                SELECT team_id, any_value(team_name) AS team_name FROM (
                    SELECT home_team_id AS team_id, home_team AS team_name FROM matches
                    UNION ALL
                    SELECT away_team_id AS team_id, away_team AS team_name FROM matches
                ) GROUP BY team_id
            )
            SELECT r.team_id, n.team_name AS name, r.elo
            FROM ranked r
            JOIN names n ON n.team_id = r.team_id
            WHERE r.rn = 1
            ORDER BY r.elo DESC
            """
        ).pl()


@st.cache_data
def dataset_overview() -> dict:
    with connect() as con:
        n = con.sql("SELECT count(*) FROM matches").fetchone()[0]
        res = con.sql(
            "SELECT result_1x2, count(*) AS n FROM matches GROUP BY result_1x2 ORDER BY result_1x2"
        ).pl()
        goals = con.sql(
            "SELECT (home_goals + away_goals) AS total FROM matches WHERE home_goals IS NOT NULL"
        ).pl()
        per_season = con.sql(
            "SELECT season, count(*) AS n FROM matches GROUP BY season ORDER BY season"
        ).pl()
    return {"n": n, "res": res, "goals": goals, "per_season": per_season}


def _count(con, table: str) -> int:
    exists = con.sql(
        f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table}'"
    ).fetchone()[0]
    return int(con.sql(f"SELECT count(*) FROM {table}").fetchone()[0]) if exists else 0


@st.cache_data
def coverage() -> dict:
    """Cuántos partidos tienen cada tipo de detalle (avance de la ingesta)."""
    with connect() as con:
        total = _count(con, "matches")
        out = {"matches (resultados)": total}
        for tbl, label in [("match_statistics", "statistics"),
                           ("match_events", "events"),
                           ("player_match_stats", "players")]:
            if _count(con, tbl):
                d = con.sql(f"SELECT count(DISTINCT match_id) FROM {tbl}").fetchone()[0]
                out[label] = int(d)
            else:
                out[label] = 0
    return out


@st.cache_data
def stats_overview() -> pl.DataFrame | None:
    """Tarjetas/córners/faltas por partido (desde match_statistics)."""
    with connect() as con:
        if not _count(con, "match_statistics"):
            return None
        return con.sql(
            """
            SELECT sum(coalesce(yellow_cards,0)+coalesce(red_cards,0)) AS tarjetas,
                   sum(coalesce(corner_kicks,0)) AS corners,
                   sum(coalesce(fouls,0)) AS faltas
            FROM match_statistics GROUP BY match_id
            """
        ).pl()


@st.cache_data
def player_leaderboards() -> dict | None:
    """Rankings de jugadores (goleadores, tiros a puerta, rating, tarjetas)."""
    with connect() as con:
        if not _count(con, "player_match_stats"):
            return None

        def q(sql):
            return con.sql(sql).pl()

        return {
            "Goleadores": q("SELECT any_value(player_name) AS jugador, sum(goals_total) AS valor "
                            "FROM player_match_stats WHERE player_id!=0 GROUP BY player_id "
                            "ORDER BY valor DESC NULLS LAST LIMIT 15"),
            "Tiros a puerta (promedio, ≥5 partidos)": q(
                "SELECT any_value(player_name) AS jugador, round(avg(shots_on),2) AS valor "
                "FROM player_match_stats WHERE player_id!=0 AND shots_on IS NOT NULL "
                "GROUP BY player_id HAVING count(*)>=5 ORDER BY valor DESC NULLS LAST LIMIT 15"),
            "Mejor rating (promedio, ≥5 partidos)": q(
                "SELECT any_value(player_name) AS jugador, round(avg(rating),2) AS valor "
                "FROM player_match_stats WHERE player_id!=0 AND rating IS NOT NULL "
                "GROUP BY player_id HAVING count(*)>=5 ORDER BY valor DESC NULLS LAST LIMIT 15"),
            "Más tarjetas amarillas": q(
                "SELECT any_value(player_name) AS jugador, sum(cards_yellow) AS valor "
                "FROM player_match_stats WHERE player_id!=0 GROUP BY player_id "
                "ORDER BY valor DESC NULLS LAST LIMIT 15"),
        }


def scoreline_matrix(lh: float, la: float, max_goals: int = 5):
    """Matriz de probabilidad de marcadores (Poisson) para el mapa de calor."""
    from scipy.stats import poisson

    return [[poisson.pmf(i, lh) * poisson.pmf(j, la) for j in range(max_goals + 1)]
            for i in range(max_goals + 1)]


def count_market_figs(data: dict):
    """(fig over/under, fig distribución) para un mercado de conteo (tarjetas/córners/goles)."""
    ou = data["over_under"]
    ou_labels = [k.replace("over_", "+").replace("_", ".") for k in ou]
    fig_ou = px.bar(x=ou_labels, y=list(ou.values()), range_y=[0, 1],
                    text=[f"{v:.0%}" for v in ou.values()],
                    labels={"x": "Línea", "y": "Prob. de superar"},
                    color=list(ou.values()), color_continuous_scale="Blues")
    dist = data["distribucion"]
    fig_d = px.bar(x=list(dist.keys()), y=list(dist.values()),
                   text=[f"{v:.0%}" for v in dist.values()],
                   labels={"x": "Cantidad", "y": "Prob."})
    for f in (fig_ou, fig_d):
        f.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240,
                        coloraxis_showscale=False, yaxis_tickformat=".0%")
    return fig_ou, fig_d


# ───────────────────────── UI ─────────────────────────

section = st.sidebar.radio("Sección", ["🔮 Predicción", "📊 Ranking Elo", "🗂️ Explorar dataset"])
st.sidebar.caption("World Cup 2026 · ML pipeline")


if section == "🔮 Predicción":
    st.title("🔮 Predicción de partido")
    teams = team_options()
    names = list(teams)
    c1, c2, c3 = st.columns(3)
    home = c1.selectbox("Local", names, index=names.index("Brazil") if "Brazil" in names else 0)
    away = c2.selectbox("Visitante", names, index=names.index("Argentina") if "Argentina" in names else 1)
    referee = c3.text_input("Árbitro (opcional)", "")
    neutral = st.checkbox("Sede neutral (Mundial) — sin ventaja de local", value=False)

    if st.button("Predecir", type="primary"):
        pred = predict_match(teams[home], teams[away], referee=referee or None, neutral=neutral)

        # --- Resumen (titular de la predicción) ---
        outcomes = {"prob_home_win": f"Gana {home}", "prob_draw": "Empate",
                    "prob_away_win": f"Gana {away}"}
        best = max(outcomes, key=lambda k: pred[k])
        st.success(
            f"**{outcomes[best]}** ({pred[best]:.0%})  ·  marcador más probable "
            f"**{pred['marcador_mas_probable']}**  ·  goles esp. "
            f"{pred['goles_esperados']['local']}–{pred['goles_esperados']['visitante']}"
        )

        # --- Comparativa de equipos ---
        with st.expander("📋 Comparativa de equipos (contexto)", expanded=True):
            comp = pred["comparativa"]
            metric_labels = {"elo": "Elo", "forma_pts": "Forma (pts/partido)",
                             "goles_favor": "Goles a favor", "goles_contra": "Goles en contra",
                             "sos_elo_rival": "Nivel rivales (Elo)"}
            tabla = [{"Métrica": lbl, home: comp["local"].get(k), away: comp["visitante"].get(k)}
                     for k, lbl in metric_labels.items()]
            st.dataframe(tabla, width="stretch", hide_index=True)
            if comp["h2h_ppg_local"] is not None:
                st.caption(f"Head-to-head: {home} promedia {comp['h2h_ppg_local']} pts/partido "
                           f"en enfrentamientos previos.")

        # --- 1X2 (donut) ---
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Resultado (1X2)")
            fig = go.Figure(go.Pie(
                labels=[f"{home} gana", "Empate", f"{away} gana"],
                values=[pred["prob_home_win"], pred["prob_draw"], pred["prob_away_win"]],
                hole=0.5, textinfo="label+percent", texttemplate="%{label}<br>%{percent:.0%}",
            ))
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Método: {pred['metodo']} · Elo {pred['home_elo']} vs {pred['away_elo']}")

        # --- Mapa de calor de marcadores ---
        with right:
            st.subheader("Marcadores probables")
            ge = pred["goles_esperados"]
            mat = scoreline_matrix(ge["local"], ge["visitante"])
            fig = px.imshow(
                mat, labels=dict(x=f"Goles {away}", y=f"Goles {home}", color="Prob."),
                x=list(range(6)), y=list(range(6)), text_auto=".0%", color_continuous_scale="Blues",
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Marcador más probable: {pred['marcador_mas_probable']}")

        # --- Mercados de goles (detallado) ---
        st.subheader("⚽ Mercados de goles")
        gm = pred["mercados_goles"]
        ge = pred["goles_esperados"]
        st.caption(f"Goles esperados — {home}: {ge['local']} · {away}: {ge['visitante']}")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Over / Under** (prob. de superar la línea)")
            ou = gm["over_under"]
            labels = ["+0.5", "+1.5", "+2.5", "+3.5", "+4.5"]
            vals = [ou[k] for k in ["over_0_5", "over_1_5", "over_2_5", "over_3_5", "over_4_5"]]
            fig = px.bar(x=labels, y=vals, range_y=[0, 1], text=[f"{v:.0%}" for v in vals],
                         labels={"x": "Línea de goles", "y": "Probabilidad"},
                         color=vals, color_continuous_scale="Greens")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260,
                              coloraxis_showscale=False, yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
        with g2:
            st.markdown("**Distribución de goles totales**")
            dist = gm["distribucion_goles"]
            fig = px.bar(x=list(dist.keys()), y=list(dist.values()),
                         text=[f"{v:.0%}" for v in dist.values()],
                         labels={"x": "Goles en el partido", "y": "Probabilidad"})
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260,
                              yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

        g3, g4 = st.columns(2)
        with g3:
            st.markdown("**Marcadores más probables**")
            st.dataframe(
                [{"Marcador": m["marcador"], "Probabilidad": f"{m['prob']:.1%}"}
                 for m in gm["marcadores_top"]],
                width="stretch", hide_index=True,
            )
        with g4:
            st.markdown("**Otros mercados**")
            dc = pred["doble_oportunidad"]
            st.dataframe([
                {"Mercado": "Ambos marcan (BTTS)", "Prob.": f"{gm['btts']:.1%}"},
                {"Mercado": f"Portería a cero {home}", "Prob.": f"{gm['porteria_cero_local']:.1%}"},
                {"Mercado": f"Portería a cero {away}", "Prob.": f"{gm['porteria_cero_visitante']:.1%}"},
                {"Mercado": "Doble oport. 1X (local o empate)", "Prob.": f"{dc['local_o_empate_1X']:.1%}"},
                {"Mercado": "Doble oport. 12 (no empate)", "Prob.": f"{dc['local_o_visita_12']:.1%}"},
                {"Mercado": "Doble oport. X2 (empate o visita)", "Prob.": f"{dc['empate_o_visita_X2']:.1%}"},
            ], width="stretch", hide_index=True)

        # --- Tarjetas ---
        st.subheader("🟨 Tarjetas")
        if "tarjetas" in pred:
            t = pred["tarjetas"]
            st.caption(f"Esperadas: **{t['esperadas']}** · más probable: **{t['mas_probable']}** "
                       f"· árbitro: {t['arbitro_usado']}")
            f1, f2 = count_market_figs(t)
            tc1, tc2 = st.columns(2)
            tc1.plotly_chart(f1, width="stretch")
            tc2.plotly_chart(f2, width="stretch")
        else:
            st.info("Modelo de tarjetas no disponible aún (entrena con `wc train-cards`).")

        # --- Córners ---
        st.subheader("🚩 Córners")
        if "corners" in pred:
            c = pred["corners"]
            st.caption(f"Esperados: **{c['esperados']}** · más probable: **{c['mas_probable']}**")
            f1, f2 = count_market_figs(c)
            cc1, cc2 = st.columns(2)
            cc1.plotly_chart(f1, width="stretch")
            cc2.plotly_chart(f2, width="stretch")
        else:
            st.info("Modelo de córners no disponible aún (entrena con `wc train-corners`).")

        # --- Tiros a puerta por jugador (principales rematadores) ---
        if "tiros_jugadores" in pred:
            st.subheader("🎯 Tiros a puerta — principales rematadores")
            pc1, pc2 = st.columns(2)
            for col, side, label in [(pc1, "local", home), (pc2, "visitante", away)]:
                with col:
                    st.markdown(f"**{label}**")
                    data = pred["tiros_jugadores"][side]
                    if data:
                        fig = px.bar(
                            data, x="shots_on_esperados", y="player", orientation="h",
                            labels={"shots_on_esperados": "Tiros a puerta esperados", "player": ""},
                            color="shots_on_esperados", color_continuous_scale="Oranges",
                        )
                        fig.update_layout(yaxis=dict(autorange="reversed"),
                                          margin=dict(t=10, b=10, l=10, r=10), height=240)
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.caption("Sin datos de jugadores para este equipo.")


elif section == "📊 Ranking Elo":
    st.title("📊 Ranking Elo de selecciones")
    st.caption("Fuerza actual según nuestro Elo (se actualiza partido a partido).")
    top_n = st.slider("Top N", 5, 50, 20)
    elos = latest_elos().head(top_n)
    fig = px.bar(
        elos.to_pandas(), x="elo", y="name", orientation="h",
        labels={"elo": "Elo", "name": ""}, color="elo", color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=max(400, top_n * 22))
    st.plotly_chart(fig, width="stretch")


elif section == "🗂️ Explorar dataset":
    st.title("🗂️ Exploración del dataset")
    tab_m, tab_s, tab_p, tab_c = st.tabs(["⚽ Partidos", "📈 Estadísticas", "👤 Jugadores", "📦 Cobertura"])

    # --- Partidos ---
    with tab_m:
        ov = dataset_overview()
        st.metric("Partidos terminados", f"{ov['n']:,}")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribución de resultados")
            labels = {0: "Gana local", 1: "Empate", 2: "Gana visitante"}
            res = ov["res"].with_columns(
                pl.col("result_1x2").replace_strict(labels, default="?").alias("resultado")
            )
            st.plotly_chart(px.pie(res.to_pandas(), names="resultado", values="n", hole=0.4),
                            width="stretch")
        with c2:
            st.subheader("Goles por partido")
            st.plotly_chart(px.histogram(ov["goals"].to_pandas(), x="total", nbins=12,
                            labels={"total": "Goles totales"}), width="stretch")
        st.subheader("Partidos por temporada")
        st.plotly_chart(px.bar(ov["per_season"].to_pandas(), x="season", y="n",
                        labels={"season": "Temporada", "n": "Partidos"}), width="stretch")

    # --- Estadísticas (match_statistics) ---
    with tab_s:
        stats = stats_overview()
        if stats is None or stats.height == 0:
            st.info("Aún no hay `match_statistics`. Baja el detalle: "
                    "`wc seed --scope national --endpoints statistics`.")
        else:
            pdf = stats.to_pandas()
            m1, m2, m3 = st.columns(3)
            m1.metric("Tarjetas / partido", f"{pdf['tarjetas'].mean():.2f}")
            m2.metric("Córners / partido", f"{pdf['corners'].mean():.2f}")
            m3.metric("Faltas / partido", f"{pdf['faltas'].mean():.1f}")
            g1, g2 = st.columns(2)
            with g1:
                st.subheader("Distribución de tarjetas")
                st.plotly_chart(px.histogram(pdf, x="tarjetas", nbins=15,
                                labels={"tarjetas": "Tarjetas por partido"}), width="stretch")
            with g2:
                st.subheader("Distribución de córners")
                st.plotly_chart(px.histogram(pdf, x="corners", nbins=15,
                                labels={"corners": "Córners por partido"}), width="stretch")

    # --- Jugadores (player_match_stats) ---
    with tab_p:
        boards = player_leaderboards()
        if boards is None:
            st.info("Aún no hay datos de jugadores. Baja el endpoint players: "
                    "`wc seed --scope national --endpoints players`.")
        else:
            pick = st.selectbox("Ranking", list(boards))
            board = boards[pick]
            if board.height == 0:
                st.caption("Sin datos suficientes para este ranking.")
            else:
                fig = px.bar(board.to_pandas(), x="valor", y="jugador", orientation="h",
                             color="valor", color_continuous_scale="Tealgrn")
                fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
                st.plotly_chart(fig, width="stretch")

    # --- Cobertura (avance de ingesta) ---
    with tab_c:
        st.subheader("Cobertura del detalle por partido")
        st.caption("Cuántos partidos tienen cada tipo de dato (te muestra el avance de tu ingesta).")
        cov = coverage()
        total = cov["matches (resultados)"] or 1
        rows = [{"dato": k, "partidos": v, "% del total": round(100 * v / total, 1)}
                for k, v in cov.items()]
        st.plotly_chart(
            px.bar(rows, x="partidos", y="dato", orientation="h", text="% del total"),
            width="stretch",
        )
        st.dataframe(rows, width="stretch")
