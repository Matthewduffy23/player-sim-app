# app.py — Advanced Scouting + Notes + Comparison Radar + Similar Players + Club Fit
# Single file, drop-in. Requires: streamlit, pandas, numpy, matplotlib.
# scikit-learn is optional; a tiny StandardScaler fallback is included.

import os
import math
from pathlib import Path
import re

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Wedge

# ---- Optional sklearn (fallback provided) ----
try:
    from sklearn.preprocessing import StandardScaler
except Exception:

    class StandardScaler:  # minimal drop-in
        def __init__(self):
            self.mean_ = None; self.scale_ = None
        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = X.mean(axis=0)
            std = X.std(axis=0, ddof=0)
            std[std == 0] = 1.0
            self.scale_ = std
            return self
        def transform(self, X):
            X = np.asarray(X, dtype=float)
            return (X - self.mean_) / self.scale_
        def fit_transform(self, X):
            self.fit(X); return self.transform(X)

# ----------------- PAGE -----------------
st.set_page_config(page_title="Advanced Scouting Suite", layout="wide")
st.title("🔎 Advanced Scouting Suite")

# ----------------- CONFIG -----------------
INCLUDED_LEAGUES = [
    'England 1.','England 2.','England 3.','England 4.','England 5.','England 6.','England 7.','England 8.','England 9.','England 10.',
    'Albania 1.','Algeria 1.','Andorra 1.','Argentina 1.','Armenia 1.','Australia 1.','Austria 1.','Austria 2.','Azerbaijan 1.','Belgium 1.',
    'Belgium 2.','Bolivia 1.','Bosnia 1.','Brazil 1.','Brazil 2.','Brazil 3.','Bulgaria 1.','Canada 1.','Chile 1.','Colombia 1.',
    'Costa Rica 1.','Croatia 1.','Cyprus 1.','Czech 1.','Czech 2.','Denmark 1.','Denmark 2.','Ecuador 1.','Egypt 1.','Estonia 1.',
    'Finland 1.','France 1.','France 2.','France 3.','Georgia 1.','Germany 1.','Germany 2.','Germany 3.','Germany 4.','Greece 1.',
    'Hungary 1.','Iceland 1.','Israel 1.','Israel 2.','Italy 1.','Italy 2.','Italy 3.','Japan 1.','Japan 2.','Kazakhstan 1.',
    'Korea 1.','Latvia 1.','Lithuania 1.','Malta 1.','Mexico 1.','Moldova 1.','Morocco 1.','Netherlands 1.','Netherlands 2.','North Macedonia 1.',
    'Northern Ireland 1.','Norway 1.','Norway 2.','Paraguay 1.','Peru 1.','Poland 1.','Poland 2.','Portugal 1.','Portugal 2.','Portugal 3.',
    'Qatar 1.','Ireland 1.','Romania 1.','Russia 1.','Saudi 1.','Scotland 1.','Scotland 2.','Scotland 3.','Serbia 1.','Serbia 2.',
    'Slovakia 1.','Slovakia 2.','Slovenia 1.','Slovenia 2.','South Africa 1.','Spain 1.','Spain 2.','Spain 3.','Sweden 1.','Sweden 2.',
    'Switzerland 1.','Switzerland 2.','Tunisia 1.','Turkey 1.','Turkey 2.','Ukraine 1.','UAE 1.','USA 1.','USA 2.','Uruguay 1.',
    'Uzbekistan 1.','Venezuela 1.','Wales 1.'
]
PRESET_LEAGUES = {
    "Top 5 Europe": {'England 1.','France 1.','Germany 1.','Italy 1.','Spain 1.'},
    "Top 20 Europe": {
        'England 1.','Italy 1.','Spain 1.','Germany 1.','France 1.',
        'England 2.','Portugal 1.','Belgium 1.','Turkey 1.','Germany 2.','Spain 2.','France 2.',
        'Netherlands 1.','Austria 1.','Switzerland 1.','Denmark 1.','Croatia 1.','Italy 2.','Czech 1.','Norway 1.'
    },
    "EFL (England 2–4)": {'England 2.','England 3.','England 4.'}
}
FEATURES = [
    'Defensive duels per 90','Defensive duels won, %','Aerial duels per 90','Aerial duels won, %','PAdj Interceptions',
    'Non-penalty goals per 90','xG per 90','Shots per 90','Shots on target, %','Goal conversion, %',
    'Crosses per 90','Accurate crosses, %','Dribbles per 90','Successful dribbles, %','Head goals per 90',
    'Key passes per 90','Touches in box per 90','Progressive runs per 90','Accelerations per 90',
    'Passes per 90','Accurate passes, %','xA per 90','Passes to penalty area per 90','Accurate passes to penalty area, %',
    'Deep completions per 90','Smart passes per 90',
]
POLAR_METRICS = [
    "Non-penalty goals per 90","xG per 90","Shots per 90",
    "Dribbles per 90","Passes to penalty area per 90","Touches in box per 90",
    "Aerial duels per 90","Aerial duels won, %","Passes per 90",
    "Accurate passes, %","xA per 90","Progressive runs per 90",
]
# Role buckets
ROLES = {
    'Target Man CF': {'desc': "Aerial outlet, duel dominance, occupy CBs, threaten crosses & second balls.",
                      'metrics': {'Aerial duels per 90': 3,'Aerial duels won, %': 4}},
    'Goal Threat CF': {'desc': "High shot & xG volume, box presence, consistent SoT and finishing.",
                       'metrics': {'Non-penalty goals per 90': 3,'Shots per 90': 1.5,'xG per 90': 3,'Touches in box per 90': 1,'Shots on target, %': 0.5}},
    'Link-Up CF': {'desc': "Combine & create; link play; progress & deliver to the penalty area.",
                   'metrics': {'Passes per 90': 2,'Passes to penalty area per 90': 1.5,'Deep completions per 90': 1,'Smart passes per 90': 1.5,
                               'Accurate passes, %': 1.5,'Key passes per 90': 1,'Dribbles per 90': 2,'Successful dribbles, %': 1,'Progressive runs per 90': 2,'xA per 90': 3}},
    'All in': {'desc': "Blend of creation + scoring; balanced all-round attacking profile.",
               'metrics': {'xA per 90': 2,'Dribbles per 90': 2,'xG per 90': 3,'Non-penalty goals per 90': 3}},
}
LEAGUE_STRENGTHS = {
    'England 1.':100.00,'Italy 1.':97.14,'Spain 1.':94.29,'Germany 1.':94.29,'France 1.':91.43,
    'Brazil 1.':82.86,'England 2.':71.43,'Portugal 1.':71.43,'Argentina 1.':71.43,
    'Belgium 1.':68.57,'Mexico 1.':68.57,'Turkey 1.':65.71,'Germany 2.':65.71,'Spain 2.':65.71,
    'France 2.':65.71,'USA 1.':65.71,'Russia 1.':65.71,'Colombia 1.':62.86,'Netherlands 1.':62.86,
    'Austria 1.':62.86,'Switzerland 1.':62.86,'Denmark 1.':62.86,'Croatia 1.':62.86,
    'Japan 1.':62.86,'Korea 1.':62.86,'Italy 2.':62.86,'Czech 1.':57.14,'Norway 1.':57.14,
    'Poland 1.':57.14,'Romania 1.':57.14,'Israel 1.':57.14,'Algeria 1.':57.14,'Paraguay 1.':57.14,
    'Saudi 1.':57.14,'Uruguay 1.':57.14,'Morocco 1.':57.00,'Brazil 2.':56.00,'Ukraine 1.':55.00,
    'Ecuador 1.':54.29,'Spain 3.':54.29,'Scotland 1.':58.00,'Chile 1.':51.43,'Cyprus 1.':51.43,
    'Portugal 2.':51.43,'Slovakia 1.':51.43,'Australia 1.':51.43,'Hungary 1.':51.43,'Egypt 1.':51.43,
    'England 3.':51.43,'France 3.':48.00,'Japan 2.':48.00,'Bulgaria 1.':48.57,'Slovenia 1.':48.57,
    'Venezuela 1.':48.00,'Germany 3.':45.71,'Albania 1.':44.00,'Serbia 1.':42.86,'Belgium 2.':42.86,
    'Bosnia 1.':42.86,'Kosovo 1.':42.86,'Nigeria 1.':42.86,'Azerbaijan 1.':50.00,'Bolivia 1.':50.00,
    'Costa Rica 1.':50.00,'South Africa 1.':50.00,'UAE 1.':50.00,'Georgia 1.':40.00,'Finland 1.':40.00,
    'Italy 3.':40.00,'Peru 1.':40.00,'Tunisia 1.':40.00,'USA 2.':40.00,'Armenia 1.':40.00,
    'North Macedonia 1.':40.00,'Qatar 1.':40.00,'Uzbekistan 1.':42.00,'Norway 2.':42.00,
    'Kazakhstan 1.':42.00,'Poland 2.':38.00,'Denmark 2.':37.00,'Czech 2.':37.14,'Israel 2.':37.14,
    'Netherlands 2.':37.14,'Switzerland 2.':37.14,'Iceland 1.':34.29,'Ireland 1.':34.29,'Sweden 2.':34.29,
    'Germany 4.':34.29,'Malta 1.':30.00,'Turkey 2.':31.43,'Canada 1.':28.57,'England 4.':28.57,
    'Scotland 2.':28.57,'Moldova 1.':28.57,'Austria 2.':25.71,'Lithuania 1.':25.71,'Brazil 3.':25.00,
    'England 7.':25.00,'Slovenia 2.':22.00,'Latvia 1.':22.86,'Serbia 2.':20.00,'Slovakia 2.':20.00,
    'England 9.':20.00,'England 8.':15.00,'Montenegro 1.':14.29,'Wales 1.':12.00,'Portugal 3.':11.43,
    'Northern Ireland 1.':11.43,'England 10.':10.00,'Scotland 3.':10.00,'England 6.':10.00
}
REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals"}

# ----------------- DATA LOADER -----------------
@st.cache_data(show_spinner=False)
def load_df(csv_name="WORLDJUNE25.csv"):
    p = Path(__file__).with_name(csv_name)
    if p.exists():
        return pd.read_csv(p)
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if not up:
        st.stop()
    return pd.read_csv(up)

df = load_df()

# ----------------- SIDEBAR FILTERS -----------------
with st.sidebar:
    st.header("Filters")
    c1, c2, c3 = st.columns([1,1,1])
    use_top5 = c1.checkbox("Top-5", value=False)
    use_top20 = c2.checkbox("Top-20", value=False)
    use_efl = c3.checkbox("EFL", value=False)

    seed = set()
    if use_top5: seed |= PRESET_LEAGUES["Top 5 Europe"]
    if use_top20: seed |= PRESET_LEAGUES["Top 20 Europe"]
    if use_efl: seed |= PRESET_LEAGUES["EFL (England 2–4)"]

    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df.get("League", pd.Series([])).dropna().unique()))
    default_leagues = sorted(seed) if seed else INCLUDED_LEAGUES
    leagues_sel = st.multiselect("Leagues (add or prune the presets)", leagues_avail, default=default_leagues)

    # numeric coercions
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (500, 5000))
    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age, max_age = st.slider("Age", age_min_data, age_max_data, (16, 33))
    pos_text = st.text_input("Position startswith", "CF")

    # Defaults OFF; league beta default shown as 0.40 but toggle unticked
    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)
    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))
    use_league_weighting = st.checkbox("Use league weighting in role score", value=False)
    beta = st.slider("League weighting beta", 0.0, 1.0, 0.40, 0.05, help="0 = ignore league strength; 1 = only league strength")

    # Market value
    df["Market value"] = pd.to_numeric(df["Market value"], errors="coerce")
    mv_col = "Market value"
    mv_max_raw = int(np.nanmax(df[mv_col])) if df[mv_col].notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    st.markdown("**Market value (€)**")
    use_m = st.checkbox("Adjust in millions", True)
    if use_m:
        max_m = int(mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, max_m))
        min_value = mv_min_m * 1_000_000
        max_value = mv_max_m * 1_000_000
    else:
        min_value, max_value = st.slider("Range (€)", 0, mv_cap, (0, mv_cap), step=100_000)

    value_band_max = st.number_input("Value band (tab 4 max €)", min_value=0, value=min_value if min_value>0 else 5_000_000, step=250_000)

    st.subheader("Minimum performance thresholds")
    enable_min_perf = st.checkbox("Require minimum percentile on selected metrics", value=False)
    sel_metrics = st.multiselect("Metrics to threshold", FEATURES[:], default=['Non-penalty goals per 90','xG per 90'] if enable_min_perf else [])
    min_pct = st.slider("Minimum percentile (0–100)", 0, 100, 60)

    top_n = st.number_input("Top N per table", 5, 200, 50, 5)
    round_to = st.selectbox("Round output percentiles to", [0, 1], index=0)

# ----------------- VALIDATION -----------------
missing = [c for c in REQUIRED_BASE if c not in df.columns]
if missing:
    st.error(f"Dataset missing required base columns: {missing}")
    st.stop()
missing_feats = [c for c in FEATURES if c not in df.columns]
if missing_feats:
    st.error(f"Dataset missing required feature columns: {missing_feats}")
    st.stop()

# ----------------- FILTER POOL -----------------
df_f = df[df["League"].isin(leagues_sel)].copy()
df_f = df_f[df_f["Position"].astype(str).str.startswith(tuple([pos_text]))]
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(min_age, max_age)]
df_f = df_f.dropna(subset=FEATURES)

df_f["Contract expires"] = pd.to_datetime(df_f["Contract expires"], errors="coerce")
if apply_contract:
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]

df_f["League Strength"] = df_f["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
df_f = df_f[(df_f["League Strength"] >= float(min_strength)) & (df_f["League Strength"] <= float(max_strength))]
df_f = df_f[(df_f["Market value"] >= min_value) & (df_f["Market value"] <= max_value)]

if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# ----------------- PERCENTILES FOR TABLES (per league) -----------------
for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)

for feat in FEATURES:
    df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

# ----------------- ROLE SCORING (tables) -----------------
def compute_weighted_role_score(df_in: pd.DataFrame, metrics: dict, beta: float, league_weighting: bool) -> pd.Series:
    total_w = sum(metrics.values()) if metrics else 1.0
    wsum = np.zeros(len(df_in))
    for m, w in metrics.items():
        col = f"{m} Percentile"
        if col in df_in.columns:
            wsum += df_in[col].values * w
    player_score = wsum / total_w  # 0..100
    if league_weighting:
        league_scaled = (df_in["League Strength"].fillna(50) / 100.0) * 100.0
        return (1 - beta) * player_score + beta * league_scaled
    return player_score

for role_name, role_def in ROLES.items():
    df_f[f"{role_name} Score"] = compute_weighted_role_score(df_f, role_def["metrics"], beta=beta, league_weighting=use_league_weighting)

# ----------------- THRESHOLDS -----------------
if enable_min_perf and sel_metrics:
    keep_mask = np.ones(len(df_f), dtype=bool)
    for m in sel_metrics:
        pct_col = f"{m} Percentile"
        if pct_col in df_f.columns:
            keep_mask &= (df_f[pct_col] >= min_pct)
    df_f = df_f[keep_mask]
if df_f.empty:
    st.warning("No players meet the minimum performance thresholds. Loosen thresholds.")
    st.stop()

# ----------------- HELPERS -----------------
def fmt_cols(df_in: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df_in.copy()
    out[score_col] = out[score_col].round(round_to).astype(int if round_to == 0 else float)
    cols = ["Player","Team","League","Age","Contract expires","League Strength", score_col]
    return out[cols]

def top_table(df_in: pd.DataFrame, role: str, head_n: int) -> pd.DataFrame:
    col = f"{role} Score"
    ranked = df_in.dropna(subset=[col]).sort_values(col, ascending=False)
    ranked = fmt_cols(ranked, col).head(head_n).reset_index(drop=True)
    ranked.index = np.arange(1, len(ranked)+1)
    return ranked

def filtered_view(df_in: pd.DataFrame, *, age_max=None, contract_year=None, value_max=None):
    t = df_in.copy()
    if age_max is not None: t = t[t["Age"] <= age_max]
    if contract_year is not None: t = t[t["Contract expires"].dt.year <= contract_year]
    if value_max is not None: t = t[t["Market value"] <= value_max]
    return t

# ----------------- TABS (tables) -----------------
tabs = st.tabs(["Overall Top N", "U23 Top N", "Expiring Contracts", "Value Band (≤ max €)"])
for role, role_def in ROLES.items():
    with tabs[0]:
        st.subheader(f"{role} — Overall Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(df_f, role, int(top_n)), use_container_width=True)
        st.divider()

    with tabs[1]:
        u23_cutoff = st.number_input(f"{role} — U23 cutoff", min_value=16, max_value=30, value=23, step=1, key=f"u23_{role}")
        st.subheader(f"{role} — U{u23_cutoff} Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, age_max=u23_cutoff), role, int(top_n)), use_container_width=True)
        st.divider()

    with tabs[2]:
        exp_year = st.number_input(f"{role} — Expiring by year", min_value=2024, max_value=2030, value=cutoff_year, step=1, key=f"exp_{role}")
        st.subheader(f"{role} — Contracts expiring ≤ {exp_year}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, contract_year=exp_year), role, int(top_n)), use_container_width=True)
        st.divider()

    with tabs[3]:
        v_max = st.number_input(f"{role} — Max value (€)", min_value=0, value=value_band_max, step=100_000, key=f"val_{role}")
        st.subheader(f"{role} — Value band ≤ €{v_max:,.0f}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, value_max=v_max), role, int(top_n)), use_container_width=True)
        st.divider()

# ----------------- SINGLE PLAYER ROLE PROFILE -----------------
st.subheader("🎯 Single Player Role Profile")
player_name = st.selectbox("Choose player", sorted(df_f["Player"].unique()))
player_row = df_f[df_f["Player"] == player_name].head(1)

# derive defaults from selected player (to propagate)
default_pos_prefix = str(player_row["Position"].iloc[0])[:2] if not player_row.empty else "CF"
default_league_for_pool = [player_row["League"].iloc[0]] if not player_row.empty else []

# Pool controls (defaults: player's league; age default 16–40 as per request)
st.caption("Percentiles & chart computed against the pool below (defaults to the player's league).")
with st.container():
    c1, c2, c3 = st.columns([2,1,1])
    leagues_pool = c1.multiselect("Comparison leagues", sorted(df["League"].dropna().unique()), default=default_league_for_pool)
    min_minutes_pool, max_minutes_pool = c2.slider("Pool minutes", 0, 5000, (500, 5000))
    age_min_pool, age_max_pool = c3.slider("Pool age", 14, 45, (16, 40))  # default 16–40
    same_pos = st.checkbox("Limit pool to current position prefix", value=True)

    c4, c5 = st.columns([1.2, 2])
    use_player_league_weight = c4.checkbox("Weight player role scores by league", value=False)
    beta_player = c5.slider("Player role beta (league vs. metrics)", 0.0, 1.0, 0.40, 0.05)

def build_pool_df():
    if not leagues_pool:
        return pd.DataFrame([], columns=df.columns)
    pool = df[df["League"].isin(leagues_pool)].copy()
    pool["Minutes played"] = pd.to_numeric(pool["Minutes played"], errors="coerce")
    pool["Age"] = pd.to_numeric(pool["Age"], errors="coerce")
    pool = pool[pool["Minutes played"].between(min_minutes_pool, max_minutes_pool)]
    pool = pool[pool["Age"].between(age_min_pool, age_max_pool)]
    if same_pos and not player_row.empty:
        pref = str(player_row["Position"].iloc[0])[:2]
        pool = pool[pool["Position"].astype(str).str.startswith(pref)]
    pool = pool.dropna(subset=POLAR_METRICS)
    return pool

def clean_attacker_label(s: str) -> str:
    s = s.replace("Non-penalty goals per 90", "Non-Pen Goals")
    s = s.replace("xG per 90", "xG").replace("xA per 90", "xA")
    s = s.replace("Shots per 90", "Shots")
    s = s.replace("Passes per 90", "Passes")
    s = s.replace("Touches in box per 90", "Touches in box")
    s = s.replace("Aerial duels per 90", "Aerial duels")
    s = s.replace("Progressive runs per 90", "Progressive runs")
    s = s.replace("Passes to penalty area per 90", "Passes to Pen area")
    s = s.replace("Accurate passes, %", "Pass %")
    return s

def percentiles_for_player_in_pool(pool_df: pd.DataFrame, ply_row: pd.Series) -> dict:
    if pool_df.empty:
        return {}
    pct_map = {}
    for m in POLAR_METRICS:
        if m not in pool_df.columns or pd.isna(ply_row[m]): continue
        series = pd.to_numeric(pool_df[m], errors="coerce").dropna()
        if series.empty: continue
        rank = (series < float(ply_row[m])).mean() * 100.0
        eq_share = (series == float(ply_row[m])).mean() * 100.0
        pct_map[m] = min(100.0, rank + 0.5 * eq_share)
    return pct_map

def player_role_scores_from_pct(pct_map: dict, *, player_league_strength: float, use_weight: bool, beta: float) -> dict:
    out = {}
    for role, rd in ROLES.items():
        weights = rd["metrics"]; total = sum(weights.values()) or 1.0
        metric_score = sum((pct_map.get(m, np.nan_to_num(0.0)) * w) for m, w in weights.items()) / total
        if use_weight:
            league_scaled = (player_league_strength / 100.0) * 100.0
            out[role] = (1 - beta) * metric_score + beta * league_scaled
        else:
            out[role] = metric_score
    return out

# Style/Strengths/Weaknesses mapping (no percentiles in chip labels)
S_W_MAP = {
    'Defensive duels per 90': {'style': 'High work rate', 'strength': 'Defensive Duels', 'weak': 'Defensive Duels'},
    'Aerial duels per 90': {'style': 'Target Man', 'strength': 'Aerial Presence', 'weak': 'Aerial Presence'},
    'Aerial duels won, %': {'style': None, 'strength': 'Aerial Duels', 'weak': 'Aerial Duels'},
    'Non-penalty goals per 90': {'style': None, 'strength': 'Scoring Goals', 'weak': 'Scoring Goals'},
    'xG per 90': {'style': 'Good shot locations', 'strength': 'Attacking Positioning', 'weak': 'Attacking Positioning'},
    'Shots per 90': {'style': 'High shot volume', 'strength': None, 'weak': None},
    'Goal conversion, %': {'style': None, 'strength': 'Finishing', 'weak': 'Finishing'},
    'Crosses per 90': {'style': 'Crossing Volume', 'strength': None, 'weak': None},
    'Accurate crosses, %': {'style': None, 'strength': 'Crossing', 'weak': 'Crossing'},
    'Dribbles per 90': {'style': '1v1 Dribbler', 'strength': None, 'weak': None},
    'Successful dribbles, %': {'style': None, 'strength': 'Dribbling', 'weak': 'Dribbling'},
    'Touches in box per 90': {'style': 'Busy in box', 'strength': None, 'weak': None},
    'Progressive runs per 90': {'style': 'Ball Carrier', 'strength': 'Progressive Runs', 'weak': 'Progressive Runs'},
    'Passes per 90': {'style': 'Active in build-up', 'strength': None, 'weak': None},
    'Accurate passes, %': {'style': None, 'strength': 'Retention', 'weak': 'Retention'},
    'xA per 90': {'style': 'Chance Creator', 'strength': 'Creating Chances', 'weak': 'Creating Chances'},
    'Passes to penalty area per 90': {'style': 'Facilitator', 'strength': 'Passes to Penalty Area', 'weak': 'Passes to Penalty Area'},
    'Deep completions per 90': {'style': 'Value-adding passer', 'strength': None, 'weak': None},
    'Smart passes per 90': {'style': 'Line-breaking passer', 'strength': None, 'weak': None},
}

# Polar chart for attacker metrics
def plot_attacker_polar_chart(labels, vals):
    N = len(labels)
    color_scale = ["#be2a3e", "#e25f48", "#f88f4d", "#f4d166", "#90b960", "#4b9b5f", "#22763f"]
    cmap = LinearSegmentedColormap.from_list("custom_scale", color_scale)
    bar_colors = [cmap(v/100.0) for v in vals]

    angles = np.linspace(0, 2*np.pi, N, endpoint=False)[::-1]
    rotation_shift = np.deg2rad(75) - angles[0]
    ang = (angles + rotation_shift) % (2*np.pi)
    width = 2*np.pi / N

    fig = plt.figure(figsize=(8.2, 6.6), dpi=180)
    fig.patch.set_facecolor('#f3f4f6')
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.74], polar=True)
    ax.set_facecolor('#f3f4f6')
    ax.set_rlim(0, 100)

    for i in range(N):
        ax.bar(ang[i], vals[i], width=width, color=bar_colors[i], edgecolor='black', linewidth=1.0, zorder=3)
        label_pos = max(12, vals[i] * 0.75)
        ax.text(ang[i], label_pos, f"{int(round(vals[i]))}", ha='center', va='center', fontsize=9, weight='bold', color='white', zorder=4)

    outer = plt.Circle((0, 0), 100, transform=ax.transData._b, color='black', fill=False, linewidth=2.2, zorder=5)
    ax.add_artist(outer)

    for i in range(N):
        sep_angle = (ang[i] - width/2) % (2*np.pi)
        is_cross = any(np.isclose(sep_angle, a, atol=0.01) for a in [0, np.pi/2, np.pi, 3*np.pi/2])
        ax.plot([sep_angle, sep_angle], [0, 100], color='black' if is_cross else '#b0b0b0', linewidth=1.6 if is_cross else 1.0, zorder=2)

    label_r = 120
    for i, lab in enumerate(labels):
        ax.text(ang[i], label_r, lab, ha='center', va='center', fontsize=8.5, weight='bold', color='#111827', zorder=6)

    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['polar'].set_visible(False); ax.grid(False)
    return fig

if player_row.empty:
    st.info("Pick a player above.")
else:
    meta = player_row[["Team","League","Age","Contract expires","League Strength","Market value"]].iloc[0]
    st.caption(
        f"**{player_name}** — {meta['Team']} • {meta['League']} • Age {int(meta['Age'])} • "
        f"Contract: {pd.to_datetime(meta['Contract expires']).date() if pd.notna(meta['Contract expires']) else 'N/A'} • "
        f"League Strength {meta['League Strength']:.1f} • Value €{meta['Market value']:,.0f}"
    )

    # Build pool & compute player percentiles within that pool
    pool_df = build_pool_df()
    if pool_df.empty:
        st.warning("Comparison pool is empty. Add at least one league.")
    else:
        ply = player_row.iloc[0]
        pct_map = percentiles_for_player_in_pool(pool_df, ply)

        # Role scores based on pool percentiles (with optional league weighting for the player)
        player_ls = float(LEAGUE_STRENGTHS.get(str(ply["League"]), 50.0))
        role_scores = player_role_scores_from_pct(
            pct_map, player_league_strength=player_ls, use_weight=use_player_league_weight, beta=beta_player
        )

        # Role table with gradient colors
        def score_to_color(v: float) -> str:
            if pd.isna(v): return "background-color: #ffffff"
            if v <= 50:
                r1,g1,b1 = (190,42,62); r2,g2,b2 = (244,209,102); t = v/50
            else:
                r1,g1,b1 = (244,209,102); r2,g2,b2 = (34,197,94); t = (v-50)/50
            r = int(r1 + (r2-r1)*t); g = int(g1 + (g2-g1)*t); b = int(b1 + (b2-b1)*t)
            return f"background-color: rgb({r},{g},{b})"

        rows = [{"Role": r, "Percentile": role_scores.get(r, np.nan)} for r in ROLES.keys()]
        role_df = pd.DataFrame(rows).set_index("Role")
        styled = (
            role_df.style
            .applymap(lambda x: score_to_color(float(x)) if pd.notna(x) else "background-color:#fff", subset=["Percentile"])
            .format({"Percentile": lambda x: f"{int(round(x))}" if pd.notna(x) else "—"})
        )
        st.dataframe(styled, use_container_width=True)

        # ---------- NOTES (Style + strengths/weaknesses from extra metrics) ----------
        def percentile_in_series(value, series: pd.Series) -> float:
            s = pd.to_numeric(series, errors="coerce").dropna()
            if len(s) == 0 or pd.isna(value): return np.nan
            rank = (s < float(value)).mean() * 100.0
            eq_share = (s == float(value)).mean() * 100.0
            return min(100.0, rank + 0.5 * eq_share)

        st.subheader("📝 Notes")
        EXTRA_METRICS = [
            'Defensive duels per 90','Aerial duels per 90','Aerial duels won, %',
            'Non-penalty goals per 90','xG per 90','Shots per 90','Goal conversion, %',
            'Crosses per 90','Accurate crosses, %','Dribbles per 90','Successful dribbles, %',
            'Touches in box per 90','Progressive runs per 90','Passes per 90','Accurate passes, %',
            'xA per 90','Passes to penalty area per 90','Deep completions per 90','Smart passes per 90'
        ]
        STYLE_MAP = {
            'Defensive duels per 90': {'style':'High work rate','sw':'Defensive Duels'},
            'Aerial duels won, %': {'style':None,'sw':'Aerial Duels'},
            'Aerial duels per 90': {'style':'Target Man','sw':'Aerial volume'},
            'Non-penalty goals per 90': {'style':None,'sw':'Scoring Goals'},
            'xG per 90': {'style':'Gets into good goal scoring positions','sw':'Attacking Positioning'},
            'Shots per 90': {'style':'Takes many shots','sw':'Shot Volume'},
            'Goal conversion, %': {'style':None,'sw':'Finishing'},
            'Crosses per 90': {'style':'Moves into wide areas to create','sw':'Crossing Volume'},
            'Accurate crosses, %': {'style':None,'sw':'Crossing Accuracy'},
            'Dribbles per 90': {'style':'1v1 dribbler','sw':'Dribble Volume'},
            'Successful dribbles, %': {'style':None,'sw':'Dribbling Efficiency'},
            'Touches in box per 90': {'style':'Busy in the box','sw':'Penalty-box Coverage'},
            'Progressive runs per 90': {'style':'Gets team up the pitch via carries','sw':'Progressive Runs'},
            'Passes per 90': {'style':'High build-up involvement','sw':'Build-up Volume'},
            'Accurate passes, %': {'style':None,'sw':'Retention'},
            'xA per 90': {'style':'Creates goal scoring chances','sw':'Chance Creation'},
            'Passes to penalty area per 90': {'style':None,'sw':'Passes to Penalty Area'},
            'Deep completions per 90': {'style':None,'sw':'Deep Completions'},
            'Smart passes per 90': {'style':None,'sw':'Smart Passes'},
        }
        HI, LO, STYLE_T = 75, 25, 65  # thresholds

        def chips(items, color):
            if not items: return "_None identified._"
            spans = [
                f"<span style='background:{color};color:#111;padding:2px 6px;border-radius:10px;margin:0 6px 6px 0;display:inline-block'>{txt}</span>"
                for txt in items[:10]
            ]
            return " ".join(spans)

        # Build style/strengths/weaknesses from pool-based (fallback to league percentiles)
        pct_extra = {}
        if isinstance(pool_df, pd.DataFrame) and not pool_df.empty:
            for m in EXTRA_METRICS:
                if m in df.columns and m in pool_df.columns and pd.notna(ply.get(m)):
                    pct_extra[m] = percentile_in_series(ply[m], pool_df[m])
        for m in EXTRA_METRICS:
            if m not in pct_extra or pd.isna(pct_extra[m]):
                col = f"{m} Percentile"
                if col in player_row.columns and pd.notna(player_row[col].iloc[0]):
                    pct_extra[m] = float(player_row[col].iloc[0])

        strengths, weaknesses, styles = [], [], []
        for m, v in pct_extra.items():
            if pd.isna(v): continue
            lab = STYLE_MAP.get(m, {})
            sw_name = lab.get('sw', m)
            style_tag = lab.get('style')
            if v >= HI: strengths.append((sw_name, v))
            elif v <= LO: weaknesses.append((sw_name, v))
            if style_tag and v >= STYLE_T: styles.append((style_tag, v))

        # De-dupe & sort
        if strengths:
            strength_best = {name: max(p for n,p in strengths if n==name) for name,_ in strengths}
            strengths = [name for name,_ in sorted(strength_best.items(), key=lambda kv: -kv[1])]
        if weaknesses:
            weakness_worst = {name: min(p for n,p in weaknesses if n==name) for name,_ in weaknesses}
            weaknesses = [name for name,_ in sorted(weakness_worst.items(), key=lambda kv: kv[1])]
        if styles:
            style_best = {name: max(p for n,p in styles if n==name) for name,_ in styles}
            styles = [name for name,_ in sorted(style_best.items(), key=lambda kv: -kv[1])]

        # Header & best-role line
        st.markdown(
            f"**Profile:** {player_name} — {ply.get('Team','?')} ({ply.get('League','?')}), "
            f"age {int(ply['Age']) if pd.notna(ply.get('Age')) else '—'}, "
            f"minutes {int(ply['Minutes played']) if pd.notna(ply.get('Minutes played')) else '—'}."
        )
        first_three = list(ROLES.keys())[:3]
        best_line = ""
        if isinstance(role_scores, dict) and role_scores:
            subset = {k: v for k, v in role_scores.items() if k in first_three}
            if subset:
                best_role = max(subset.items(), key=lambda kv: kv[1])[0]
                best_line = f"**Best role:** {best_role}."
        else:
            best_role, best_val = None, -1
            for r in first_three:
                col = f"{r} Score"
                if col in player_row.columns and pd.notna(player_row[col].iloc[0]):
                    if float(player_row[col].iloc[0]) > best_val:
                        best_val = float(player_row[col].iloc[0]); best_role = r
            if best_role is not None:
                best_line = f"**Best role (league):** {best_role}."
        if best_line: st.markdown(best_line)

        st.markdown("**Style:**")
        st.markdown(chips(styles, "#bfdbfe"), unsafe_allow_html=True)  # light blue
        st.markdown("**Strengths:**")
        st.markdown(chips(strengths, "#a7f3d0"), unsafe_allow_html=True)  # light green
        st.markdown("**Weaknesses / growth areas:**")
        st.markdown(chips(weaknesses, "#fecaca"), unsafe_allow_html=True)  # light red

        # ---------- POLAR CHART ----------
        labels = [clean_attacker_label(m) for m in POLAR_METRICS if m in pct_map]
        vals = [pct_map[m] for m in POLAR_METRICS if m in pct_map]
        if vals:
            fig = plot_attacker_polar_chart(labels, vals)
            team = str(ply["Team"]); league = str(ply["League"])
            fig.text(0.06, 0.94, f"{player_name} — Performance (pool size: {len(pool_df):,})", fontsize=16, weight='bold', ha='left', color='#111827')
            fig.text(0.06, 0.915, f"Against selected pool • Team: {team} • Native league: {league}", fontsize=9, ha='left', color='#6b7280')
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Not enough metrics to draw the polar chart.")

# =====================================================================
# ============== BELOW THE NOTES: 3 EXTRA FEATURE BLOCKS ==============
# =====================================================================

# ----------------- (A) COMPARISON RADAR (SB-STYLE) -----------------
st.markdown("---")
st.header("📊 Player Comparison — SB Radar")

with st.expander("Radar settings", expanded=False):
    # default pos prefix from selected player
    pos_scope = st.text_input("Position startswith (radar pool)", default_pos_prefix, key="rad_pos")

    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    min_minutes_r, max_minutes_r = st.slider("Minutes filter (radar pool)", 0, 5000, (500, 5000), key="rad_min")

    # default age 16–40 (changed from 16–33)
    age_min_r_bound = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_r_bound = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age_r, max_age_r = st.slider("Age filter (radar pool)", age_min_r_bound, age_max_r_bound, (16, 40), key="rad_age")

    picker_pool = df[df["Position"].astype(str).str.startswith(tuple([pos_scope]))].copy()
    players = sorted(picker_pool["Player"].dropna().unique().tolist())
    if len(players) < 2:
        st.warning("Not enough players for this filter.")
        players = sorted(df["Player"].dropna().unique().tolist())

    # default Player A = selected player if present
    try:
        pA_index = players.index(player_name)
    except Exception:
        pA_index = 0

    pA = st.selectbox("Player A (red)", players, index=pA_index, key="rad_a")

    # default Player B = next one (or index 1)
    pB_default_index = 1 if len(players) > 1 else 0
    if pA_index == pB_default_index and len(players) > 2:
        pB_default_index = 2
    pB = st.selectbox("Player B (blue)", players, index=pB_default_index, key="rad_b")

    DEFAULT_METRICS = [
        "Non-penalty goals per 90","xG per 90","Shots per 90",
        "Dribbles per 90","Successful dribbles, %","Touches in box per 90",
        "Aerial duels per 90","Aerial duels won, %","Passes per 90",
        "Accurate passes, %","xA per 90"
    ]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metrics_default = [m for m in DEFAULT_METRICS if m in df.columns]
    radar_metrics = st.multiselect("Radar metrics", [c for c in df.columns if c in numeric_cols], metrics_default, key="rad_ms")

    sort_by_gap = st.checkbox("Sort axes by biggest gap", False, key="rad_sort")
    show_avg = st.checkbox("Show pool average (thin line)", True, key="rad_avg")

    # Build radar pool and draw
    def clean_label_r(s: str) -> str:
        s = s.replace("Non-penalty goals per 90", "Non-Pen Goals")
        s = s.replace("xG per 90", "xG").replace("xA per 90", "xA")
        s = s.replace("Shots per 90", "Shots").replace("Passes per 90", "Passes")
        s = s.replace("Touches in box per 90", "Touches in box").replace("Aerial duels per 90", "Aerial duels")
        s = s.replace("Successful dribbles, %", "Dribble %").replace("Accurate passes, %", "Pass %")
        s = re.sub(r"\s*per\s*90", "", s, flags=re.I); return s

    if radar_metrics:
        try:
            rowA = df[df["Player"] == pA].iloc[0]; rowB = df[df["Player"] == pB].iloc[0]
            union_leagues = {rowA["League"], rowB["League"]}
            pool = df[(df["League"].isin(union_leagues)) &
                      (df["Position"].astype(str).str.startswith(tuple([pos_scope]))) &
                      (df["Minutes played"].between(min_minutes_r, max_minutes_r)) &
                      (df["Age"].between(min_age_r, max_age_r))].copy()

            for m in radar_metrics:
                pool[m] = pd.to_numeric(pool[m], errors="coerce")
            pool = pool.dropna(subset=radar_metrics)

            if not pool.empty:
                labels = [clean_label_r(m) for m in radar_metrics]
                pool_pct = pool[radar_metrics].rank(pct=True) * 100.0

                def pct_for(player):
                    sub_idx = pool[pool["Player"] == player].index
                    if len(sub_idx)==0:
                        return np.full(len(radar_metrics), np.nan)
                    return pool_pct.loc[sub_idx, :].mean(axis=0).values

                A_r = pct_for(pA); B_r = pct_for(pB); AVG_r = np.full(len(radar_metrics), 50.0)

                axis_min = pool[radar_metrics].min().values
                axis_max = pool[radar_metrics].max().values
                pad = (axis_max - axis_min) * 0.07
                axis_min = axis_min - pad; axis_max = axis_max + pad

                ring_radii = np.linspace(10, 100, 11)
                axis_ticks = [np.linspace(axis_min[i], axis_max[i], 11) for i in range(len(labels))]

                if sort_by_gap:
                    order = np.argsort(-np.abs(A_r - B_r))
                    labels = [labels[i] for i in order]
                    A_r = A_r[order]; B_r = B_r[order]; AVG_r = AVG_r[order]
                    axis_ticks = [axis_ticks[i] for i in order]

                # draw
                COL_A = "#C81E1E"; COL_B = "#1D4ED8"
                FILL_A = (200/255, 30/255, 30/255, 0.60)
                FILL_B = (29/255, 78/255, 216/255, 0.60)
                PAGE_BG = AX_BG = "#FFFFFF"
                GRID_BAND_A = "#FFFFFF"; GRID_BAND_B = "#E5E7EB"
                RING_COLOR = "#D1D5DB"; RING_LW = 1.0
                LABEL_COLOR = "#0F172A"; TITLE_FS = 26; SUB_FS = 12; AXIS_FS = 10
                TICK_FS = 7; TICK_COLOR = "#9CA3AF"; INNER_HOLE = 10

                def draw_radar(labels, A_r, B_r, ticks, headerA, subA, subA2, headerB, subB, subB2, show_avg=False, AVG_r=None):
                    N = len(labels)
                    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
                    theta_closed = np.concatenate([theta, theta[:1]])
                    Ar = np.concatenate([A_r, A_r[:1]])
                    Br = np.concatenate([B_r, B_r[:1]])

                    fig = plt.figure(figsize=(13.2, 8.0), dpi=260); fig.patch.set_facecolor(PAGE_BG)
                    ax = plt.subplot(111, polar=True); ax.set_facecolor(AX_BG)
                    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)

                    ax.set_xticks(theta); ax.set_xticklabels(labels, fontsize=AXIS_FS, color=LABEL_COLOR, fontweight=600)
                    ax.set_yticks([]); ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]

                    for i in range(10):
                        r0, r1 = np.linspace(INNER_HOLE,100,11)[i], np.linspace(INNER_HOLE,100,11)[i+1]
                        band = GRID_BAND_A if i % 2 == 0 else GRID_BAND_B
                        ax.add_artist(Wedge((0,0), r1, 0, 360, width=(r1-r0),
                                            transform=ax.transData._b, facecolor=band, edgecolor="none", zorder=0.8))

                    ring_t = np.linspace(0, 2*np.pi, 361)
                    for r in np.linspace(INNER_HOLE,100,11):
                        ax.plot(ring_t, np.full_like(ring_t, r), color=RING_COLOR, lw=RING_LW, zorder=0.9)

                    start_idx = 2
                    for i, ang in enumerate(theta):
                        vals = ticks[i][start_idx:]
                        for rr, v in zip(np.linspace(INNER_HOLE,100,11)[start_idx:], vals):
                            ax.text(ang, rr-1.8, f"{v:.1f}", ha="center", va="center", fontsize=TICK_FS, color=TICK_COLOR, zorder=1.1)

                    ax.add_artist(Circle((0,0), radius=INNER_HOLE-0.6, transform=ax.transData._b, color=PAGE_BG, zorder=1.2, ec="none"))

                    if show_avg and AVG_r is not None:
                        Avg = np.concatenate([AVG_r, AVG_r[:1]])
                        ax.plot(theta_closed, Avg, lw=1.5, color="#94A3B8", ls="--", alpha=0.9, zorder=2.2)

                    ax.plot(theta_closed, Ar, color=COL_A, lw=2.2, zorder=3); ax.fill(theta_closed, Ar, color=FILL_A, zorder=2.5)
                    ax.plot(theta_closed, Br, color=COL_B, lw=2.2, zorder=3); ax.fill(theta_closed, Br, color=FILL_B, zorder=2.5)

                    ax.set_rlim(0, 105)

                    minsA = f"{int(pd.to_numeric(rowA.get('Minutes played',0))):,} mins" if pd.notna(rowA.get('Minutes played')) else "Minutes: N/A"
                    minsB = f"{int(pd.to_numeric(rowB.get('Minutes played',0))):,} mins" if pd.notna(rowB.get('Minutes played')) else "Minutes: N/A"

                    fig.text(0.12, 0.96, f"{pA}", color=COL_A, fontsize=TITLE_FS, fontweight="bold", ha="left")
                    fig.text(0.12, 0.935, f"{rowA['Team']} — {rowA['League']}", color=COL_A, fontsize=SUB_FS, ha="left")
                    fig.text(0.12, 0.915, minsA, color="#374151", fontsize=10, ha="left")

                    fig.text(0.88, 0.96, f"{pB}", color=COL_B, fontsize=TITLE_FS, fontweight="bold", ha="right")
                    fig.text(0.88, 0.935, f"{rowB['Team']} — {rowB['League']}", color=COL_B, fontsize=SUB_FS, ha="right")
                    fig.text(0.88, 0.915, minsB, color="#374151", fontsize=10, ha="right")
                    return fig

                figr = draw_radar(labels, A_r, B_r, axis_ticks, pA, "", "", pB, "", "", show_avg=show_avg, AVG_r=AVG_r)
                st.pyplot(figr, use_container_width=True)
            else:
                st.info("No players remain in radar pool after filters.")
        except Exception as e:
            st.info(f"Radar could not be drawn: {e}")

# ----------------- (B) SIMILAR PLAYERS (adjustable pool) -----------------
st.markdown("---")
st.header("🧭 Similar players (within adjustable pool)")

with st.expander("Similarity settings", expanded=False):
    # leagues for candidates = the same main selection by default
    sim_leagues = st.multiselect(
        "Candidate leagues",
        sorted(set(INCLUDED_LEAGUES) | set(df["League"].dropna().unique())),
        default=leagues_sel,
        key="sim_leagues"
    )

    sim_min_minutes, sim_max_minutes = st.slider("Minutes played (candidates)", 0, 5000, (500, 5000), key="sim_min")
    sim_min_age, sim_max_age = st.slider("Age (candidates)", 14, 45, (16, 33), key="sim_age")

    percentile_weight = st.slider("Percentile weight", 0.0, 1.0, 0.7, 0.05, key="sim_pw")
    league_weight_sim = st.slider("League weight (difficulty adj.)", 0.0, 1.0, 0.2, 0.05, key="sim_lw")
    top_n_sim = st.number_input("Show top N", min_value=5, max_value=200, value=50, step=5, key="sim_top")

    # similarity computation (light version: uses a fixed feature basket)
    SIM_FEATURES = [
        'Defensive duels per 90','Aerial duels per 90','Aerial duels won, %',
        'Non-penalty goals per 90','xG per 90','Shots per 90',
        'Crosses per 90','Dribbles per 90','Successful dribbles, %',
        'Touches in box per 90','Progressive runs per 90','Passes per 90',
        'Accurate passes, %','xA per 90','Smart passes per 90',
        'Passes to penalty area per 90','Deep completions per 90'
    ]

    if not player_row.empty:
        target_row_full = df[df['Player'] == player_name].head(1).iloc[0]
        target_league = target_row_full['League']

        df_candidates = df[df['League'].isin(sim_leagues)].copy()
        df_candidates = df_candidates[
            (df_candidates['Minutes played'].between(sim_min_minutes, sim_max_minutes)) &
            (df_candidates['Age'].between(sim_min_age, sim_max_age))
        ]
        df_candidates = df_candidates.dropna(subset=SIM_FEATURES)
        df_candidates = df_candidates[df_candidates['Player'] != player_name]

        if not df_candidates.empty:
            # percentile ranks within candidate pool (per-league for robustness)
            percl = df_candidates.groupby('League')[SIM_FEATURES].rank(pct=True)

            # target percentiles computed on df global per-league
            target_percentiles = df.groupby('League')[SIM_FEATURES].rank(pct=True).loc[df['Player'] == player_name]

            # standardize on candidate pool
            scaler = StandardScaler()
            standardized_features = scaler.fit_transform(df_candidates[SIM_FEATURES])
            target_features_standardized = scaler.transform([target_row_full[SIM_FEATURES].values])

            weights = np.ones(len(SIM_FEATURES), dtype=float)

            percentile_distances = np.linalg.norm((percl.values - target_percentiles.values) * weights, axis=1)
            actual_value_distances = np.linalg.norm((standardized_features - target_features_standardized) * weights, axis=1)

            combined = percentile_distances * percentile_weight + actual_value_distances * (1.0 - percentile_weight)

            # robust normalization -> similarity 0..100
            arr = np.asarray(combined, dtype=float).ravel()
            if arr.size == 0:
                norm = arr
            else:
                rng = np.ptp(arr)
                norm = (arr - arr.min()) / (rng if rng != 0 else 1.0)
            similarities = ((1.0 - norm) * 100.0).round(2)

            out = df_candidates[['Player','Team','League','Age','Minutes played','Market value']].copy()
            out['League strength'] = out['League'].map(LEAGUE_STRENGTHS).fillna(0.0)

            tgt_ls = LEAGUE_STRENGTHS.get(target_league, 1.0)
            league_ratio = (out['League strength'] / tgt_ls).clip(lower=0.5, upper=1.2)

            out['Similarity'] = similarities
            out['Adjusted Similarity'] = out['Similarity'] * (1 - league_weight_sim) + out['Similarity'] * league_ratio * league_weight_sim

            out = out.sort_values('Adjusted Similarity', ascending=False).reset_index(drop=True)
            out.insert(0, 'Rank', np.arange(1, len(out) + 1))

            st.dataframe(out.head(int(top_n_sim)), use_container_width=True)
        else:
            st.info("No candidates after similarity filters.")
    else:
        st.caption("Pick a player to see similar players.")

# ---------------------------- (C) CLUB FIT — self-contained block ----------------------------
st.markdown("---")
st.header("🏟️ Club Fit Finder")

# ---------- SAFE FALLBACKS (use existing globals if present; use UPPERCASE names) ----------
# leagues list
if 'INCLUDED_LEAGUES' in globals():
    _included_leagues_cf = list(INCLUDED_LEAGUES)
else:
    _included_leagues_cf = sorted(pd.Series(df.get('League', pd.Series([]))).dropna().unique().tolist())

# presets (add Top 20 + EFL explicitly)
if 'PRESET_LEAGUES' in globals():
    _PRESETS_CF = {
        "All listed leagues": _included_leagues_cf,
        "Top 5 Europe": sorted(list(PRESET_LEAGUES.get("Top 5 Europe", []))),
        "Top 20 Europe": sorted(list(PRESET_LEAGUES.get("Top 20 Europe", []))),
        "EFL (England 2–4)": sorted(list(PRESET_LEAGUES.get("EFL (England 2–4)", []))),
        "Custom": None,
    }
else:
    _PRESETS_CF = {
        "All listed leagues": _included_leagues_cf,
        "Top 5 Europe": [],
        "Top 20 Europe": [],
        "EFL (England 2–4)": [],
        "Custom": None,
    }

# default per-metric weights (preloaded, not all 1s)
_DEFAULT_W_CF = {
    'Passes per 90': 2,
    'Accurate passes, %': 2,
    'Dribbles per 90': 2,
    'Non-penalty goals per 90': 2,
    'Shots per 90': 2,
    'Successful dribbles, %': 2,
    'Aerial duels won, %': 2,
    'xA per 90': 2,
    'xG per 90': 2,
    'Touches in box per 90': 2,
}

# league strengths (use your table; no neutral fallback unless truly unknown)
if 'LEAGUE_STRENGTHS' in globals():
    _LS_CF = dict(LEAGUE_STRENGTHS)
else:
    _LS_CF = {lg: 50.0 for lg in _included_leagues_cf}

# weight dials
DEFAULT_LEAGUE_WEIGHT = 0.4
DEFAULT_MARKET_WEIGHT  = 0.2

# features (fixed)
CF_FEATURES = [
    'Defensive duels per 90','Aerial duels per 90','Aerial duels won, %','PAdj Interceptions',
    'Non-penalty goals per 90','xG per 90','Shots per 90','Shots on target, %','Goal conversion, %',
    'Crosses per 90','Accurate crosses, %','Dribbles per 90','Successful dribbles, %',
    'Offensive duels per 90','Touches in box per 90','Progressive runs per 90','Accelerations per 90',
    'Passes per 90','Accurate passes, %','xA per 90','Smart passes per 90','Key passes per 90',
    'Passes to final third per 90','Passes to penalty area per 90','Accurate passes to penalty area, %',
    'Deep completions per 90'
]

required_cols_cf = {'Player','Team','League','Age','Position','Minutes played','Market value', *CF_FEATURES}
missing_cf = [c for c in required_cols_cf if c not in df.columns]

if missing_cf:
    st.error(f"Club Fit: dataset missing required columns: {missing_cf}")
else:
    # -------------------- Controls --------------------
    with st.expander("Club-fit settings", expanded=False):
        leagues_available_cf = sorted(set(_included_leagues_cf) | set(df.get('League', pd.Series([])).dropna().unique()))

        # Target leagues (only affects target player list)
        target_leagues_cf = st.multiselect(
            "Target leagues (choose target from here)",
            leagues_available_cf,
            default=leagues_available_cf,
            key="cf_target_leagues"
        )

        # Candidate pool via preset + extras
        if 'candidate_leagues_cf' not in st.session_state:
            st.session_state.candidate_leagues_cf = list(_included_leagues_cf)

        preset_name_cf = st.selectbox("Candidate pool preset", list(_PRESETS_CF.keys()), index=0, key="cf_preset_name")
        c1a, c1b = st.columns([1,2])
        if c1a.button("Apply preset", key="cf_apply_preset"):
            if _PRESETS_CF.get(preset_name_cf) is not None:
                st.session_state.candidate_leagues_cf = list(_PRESETS_CF[preset_name_cf])

        extra_candidate_leagues_cf = c1b.multiselect(
            "Extra leagues to add", leagues_available_cf, default=[], key="cf_extra_leagues"
        )
        leagues_selected_cf = sorted(set(st.session_state.candidate_leagues_cf) | set(extra_candidate_leagues_cf))
        st.caption(f"Candidate pool leagues: **{len(leagues_selected_cf)}** selected.")

        # default position prefix + default target = selected player
        pos_scope_cf = st.text_input("Position startswith (club fit)", default_pos_prefix, key="cf_pos_scope")

        # Target player selector (from target leagues) default to selected player
        target_pool_cf = df[df['League'].isin(target_leagues_cf)]
        target_pool_cf = target_pool_cf[target_pool_cf['Position'].astype(str).str.startswith(tuple([pos_scope_cf]))]
        target_options_cf = sorted(target_pool_cf['Player'].dropna().unique())
        try:
            default_target_idx = target_options_cf.index(player_name)
        except Exception:
            default_target_idx = 0 if target_options_cf else 0

        target_player_cf = st.selectbox(
            "Target player", target_options_cf,
            index=default_target_idx if target_options_cf else 0, key="cf_target_player"
        )

        # Minutes / age filters for candidate pool (teams built from these players)
        max_minutes_in_data_cf = int(pd.to_numeric(df.get('Minutes played', pd.Series([0])), errors='coerce').fillna(0).max())
        slider_max_minutes_cf = int(max(1000, max_minutes_in_data_cf))
        min_minutes_cf, max_minutes_cf = st.slider(
            "Minutes filter (candidates)", 0, slider_max_minutes_cf, (500, slider_max_minutes_cf), key="cf_minutes_slider"
        )

        age_series_cf = pd.to_numeric(df.get('Age', pd.Series([16, 45])), errors='coerce')
        age_min_data_cf = int(np.nanmin(age_series_cf)) if age_series_cf.notna().any() else 14
        age_max_data_cf = int(np.nanmax(age_series_cf)) if age_series_cf.notna().any() else 45
        # default age 16–40 (changed)
        min_age_cf, max_age_cf = st.slider(
            "Age filter (candidates)", age_min_data_cf, age_max_data_cf, (16, 40), key="cf_age_slider"
        )

        min_strength_cf, max_strength_cf = st.slider("League quality (strength)", 0, 101, (0, 101), key="cf_strength")

        # Weights
        league_weight_cf = st.slider("League weight", 0.0, 1.0, DEFAULT_LEAGUE_WEIGHT, 0.05, key="cf_league_w")
        market_value_weight_cf = st.slider("Market value weight", 0.0, 1.0, DEFAULT_MARKET_WEIGHT, 0.05, key="cf_market_w")
        manual_override_cf = st.number_input("Target market value override (€)", min_value=0, value=0, step=100_000, key="cf_mv_override")

        # Advanced feature weights (preloaded defaults)
        st.subheader("Advanced feature weights")
        st.caption("Unlisted features default to weight = 1.")
        weights_ui_cf = {}
        for f in CF_FEATURES:
            default_val = _DEFAULT_W_CF.get(f, 1)
            weights_ui_cf[f] = st.slider(f"• {f}", 0, 5, int(default_val), key=f"cf_w_{f}")

        top_n_cf = st.number_input("Show top N teams", 5, 100, 20, 5, key="cf_topn")

    # -------------------- Compute --------------------
    if target_player_cf and (target_player_cf in df['Player'].values):
        # Candidate player pool
        df_candidates_cf = df[df['League'].isin(leagues_selected_cf)].copy()
        df_candidates_cf = df_candidates_cf[df_candidates_cf['Position'].astype(str).str.startswith(tuple([pos_scope_cf]))]

        # Numerics + filters
        df_candidates_cf['Minutes played'] = pd.to_numeric(df_candidates_cf['Minutes played'], errors='coerce')
        df_candidates_cf['Age'] = pd.to_numeric(df_candidates_cf['Age'], errors='coerce')
        df_candidates_cf['Market value'] = pd.to_numeric(df_candidates_cf['Market value'], errors='coerce')

        df_candidates_cf = df_candidates_cf[
            df_candidates_cf['Minutes played'].between(min_minutes_cf, max_minutes_cf, inclusive='both')
        ]
        df_candidates_cf = df_candidates_cf[
            df_candidates_cf['Age'].between(min_age_cf, max_age_cf, inclusive='both')
        ]
        df_candidates_cf = df_candidates_cf.dropna(subset=CF_FEATURES)

        if df_candidates_cf.empty:
            st.info("No candidate players after filters. Widen candidate leagues or relax filters.")
        else:
            # Target (from target leagues)
            df_target_pool_cf = df[df['League'].isin(target_leagues_cf)].copy()
            df_target_pool_cf = df_target_pool_cf[df_target_pool_cf['Position'].astype(str).str.startswith(tuple([pos_scope_cf]))]

            if target_player_cf not in df_target_pool_cf['Player'].values:
                st.info("Target player not found in selected target leagues.")
            else:
                df_target_pool_cf['Market value'] = pd.to_numeric(df_target_pool_cf['Market value'], errors='coerce')

                target_row_cf = df_target_pool_cf.loc[df_target_pool_cf['Player'] == target_player_cf].iloc[0]
                target_vector_cf = target_row_cf[CF_FEATURES].values
                target_ls_cf = float(_LS_CF.get(target_row_cf['League'], 50.0))

                # Target MV (override if provided)
                target_market_value_cf = (
                    float(manual_override_cf) if manual_override_cf and manual_override_cf > 0 else
                    (float(target_row_cf['Market value']) if pd.notna(target_row_cf['Market value']) and target_row_cf['Market value'] > 0 else 2_000_000.0)
                )

                # Team profiles (mean of players that survived candidate filters)
                club_profiles_cf = df_candidates_cf.groupby('Team')[CF_FEATURES].mean().reset_index()

                # Team league & average team MV from same pool
                team_league_cf = df_candidates_cf.groupby('Team')['League'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
                team_market_cf = df_candidates_cf.groupby('Team')['Market value'].mean()

                club_profiles_cf['League'] = club_profiles_cf['Team'].map(team_league_cf)
                club_profiles_cf['Avg Team Market Value'] = club_profiles_cf['Team'].map(team_market_cf)
                club_profiles_cf = club_profiles_cf.dropna(subset=['Avg Team Market Value'])

                # Standardize & weighted distance
                scaler_cf = StandardScaler()
                X_team = scaler_cf.fit_transform(club_profiles_cf[CF_FEATURES])
                x_tgt = scaler_cf.transform([target_vector_cf])[0]

                weights_vec_cf = np.array([weights_ui_cf.get(f, 1) for f in CF_FEATURES], dtype=float)

                # Distance (lower = more similar)
                dist_cf = np.linalg.norm((X_team - x_tgt) * weights_vec_cf, axis=1)
                rng = float(dist_cf.max() - dist_cf.min())
                club_fit_base = (1 - (dist_cf - float(dist_cf.min())) / (rng if rng > 0 else 1.0)) * 100.0
                club_profiles_cf['Club Fit %'] = club_fit_base.round(2)

                # League strength adjustment & filter
                club_profiles_cf['League strength'] = club_profiles_cf['League'].map(_LS_CF).fillna(50.0)
                club_profiles_cf = club_profiles_cf[
                    (club_profiles_cf['League strength'] >= float(min_strength_cf)) &
                    (club_profiles_cf['League strength'] <= float(max_strength_cf))
                ]

                if club_profiles_cf.empty:
                    st.info("No teams remain after league-strength filter.")
                else:
                    # Difficulty ratio vs target league
                    ratio_cf = (club_profiles_cf['League strength'] / target_ls_cf).clip(0.5, 1.2)
                    club_profiles_cf['Adjusted Fit %'] = (
                        club_profiles_cf['Club Fit %'] * (1 - league_weight_cf) +
                        club_profiles_cf['Club Fit %'] * ratio_cf * league_weight_cf
                    )

                    # Mild penalty if destination league >> target
                    league_gap_cf = (club_profiles_cf['League strength'] - target_ls_cf).clip(lower=0)
                    penalty_cf = (1 - (league_gap_cf / 100)).clip(lower=0.7)
                    club_profiles_cf['Adjusted Fit %'] = club_profiles_cf['Adjusted Fit %'] * penalty_cf

                    # Market value fit (closer team MV to target MV gets rewarded)
                    value_fit_ratio_cf = (club_profiles_cf['Avg Team Market Value'] / target_market_value_cf).clip(0.5, 1.5)
                    value_fit_score_cf = (1 - abs(1 - value_fit_ratio_cf)) * 100.0

                    club_profiles_cf['Final Fit %'] = (
                        club_profiles_cf['Adjusted Fit %'] * (1 - market_value_weight_cf) +
                        value_fit_score_cf * market_value_weight_cf
                    )

                    # ---------------- Results ----------------
                    results_cf = club_profiles_cf[[
                        'Team','League','League strength','Avg Team Market Value',
                        'Club Fit %','Adjusted Fit %','Final Fit %'
                    ]].copy()

                    results_cf = results_cf.sort_values('Final Fit %', ascending=False).reset_index(drop=True)
                    results_cf.insert(0, 'Rank', np.arange(1, len(results_cf) + 1))

                    st.caption(
                        f"Target: {target_player_cf} — {target_row_cf.get('Team','Unknown')} ({target_row_cf['League']}) • "
                        f"Target MV used: €{target_market_value_cf:,.0f} • Target LS {target_ls_cf:.2f} • "
                        f"Candidates: {len(leagues_selected_cf)} leagues (preset: {preset_name_cf})"
                    )
                    st.dataframe(results_cf.head(int(top_n_cf)), use_container_width=True)

                    # Export
                    csv_cf = results_cf.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download all results (CSV)", data=csv_cf, file_name="club_fit_results.csv", mime="text/csv")

                    # Debug panel
                    with st.expander("Debug / Repro details"):
                        st.write({
                            "preset": preset_name_cf,
                            "candidate_leagues_count": len(leagues_selected_cf),
                            "target_leagues_count": len(target_leagues_cf),
                            "league_weight": float(league_weight_cf),
                            "market_value_weight": float(market_value_weight_cf),
                            "target_market_value_used": float(target_market_value_cf),
                            "minutes_range": (int(min_minutes_cf), int(max_minutes_cf)),
                            "age_range": (int(min_age_cf), int(max_age_cf)),
                            "strength_range": (int(min_strength_cf), int(max_strength_cf)),
                            "n_teams": int(results_cf.shape[0]),
                        })





