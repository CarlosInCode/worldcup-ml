"""Parámetros del proyecto en un solo lugar (antes estaban dispersos como números mágicos).

Tenerlos centralizados permite ajustarlos sin tocar la lógica y documentar su sentido.
"""

# --- Elo ---
ELO_BASE = 1500.0          # rating inicial de un equipo sin historial
ELO_K = 20.0               # cuánto se mueve el rating tras cada partido
HOME_ADVANTAGE_ELO = 65.0  # ventaja de local en puntos Elo (0 en sede neutral)

# --- Ventanas de forma reciente (rolling) ---
FORM_WINDOW = 5            # nº de partidos para forma de equipo
CORNER_WINDOW = 10         # córners son ruidosos -> ventana más larga
SOS_WINDOW = 5             # ventana para strength-of-schedule

# --- Goles / simulación ---
LEAGUE_AVG_GOALS = 1.35    # goles por equipo por partido (media de referencia)
HOME_GOAL_FACTOR = 1.10    # multiplicador de goles del local (1.0 si neutral)
SIM_N = 50_000             # nº de simulaciones Monte Carlo

# --- Importancia de competición (feature ordinal) ---
IMPORTANCE = {"friendly": 0, "qualifier": 1, "nations": 1, "tournament": 2}
