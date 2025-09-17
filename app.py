# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import math

st.set_page_config(page_title="Player Similarity Finder", layout="wide")

# --- Load dataset directly from repo ---
df = pd.read_csv("WORLDJUNE25.csv")

st.title("⚽ Player Similarity Finder")

# ---------------------------
# CONSTANTS
# ---------------------------
included_leagues = [
    'England 1.', 'England 2.', 'England 3.', 'England 4.', 'England 5.',
    'England 6.', 'England 7.', 'England 8.', 'England 9.', 'England 10.',
    'Albania 1.', 'Algeria 1.', 'Andorra 1.', 'Argentina 1.', 'Armenia 1.',
    'Australia 1.', 'Austria 1.', 'Austria 2.', 'Azerbaijan 1.', 'Belgium 1.',
    'Belgium 2.', 'Bolivia 1.', 'Bosnia 1.', 'Brazil 1.', 'Brazil 2.', 'Brazil 3.',
    'Bulgaria 1.', 'Canada 1.', 'Chile 1.', 'Colombia 1.', 'Costa Rica 1.',
    'Croatia 1.', 'Cyprus 1.', 'Czech 1.', 'Czech 2.', 'Denmark 1.', 'Denmark 2.',
    'Ecuador 1.', 'Egypt 1.', 'Estonia 1.', 'Finland 1.', 'France 1.', 'France 2.',
    'France 3.', 'Georgia 1.', 'Germany 1.', 'Germany 2.', 'Germany 3.',
    'Germany 4.', 'Greece 1.', 'Hungary 1.', 'Iceland 1.', 'Israel 1.',
    'Israel 2.', 'Italy 1.', 'Italy 2.', 'Italy 3.', 'Japan 1.', 'Japan 2.',
    'Kazakhstan 1.', 'Korea 1.', 'Latvia 1.', 'Lithuania 1.', 'Malta 1.',
    'Mexico 1.', 'Moldova 1.', 'Morocco 1.', 'Netherlands 1.', 'Netherlands 2.',
    'North Macedonia 1.', 'Northern Ireland 1.', 'Norway 1.', 'Norway 2.',
    'Paraguay 1.', 'Peru 1.', 'Poland 1.', 'Poland 2.', 'Portugal 1.',
    'Portugal 2.', 'Portugal 3.', 'Qatar 1.', 'Ireland 1.', 'Romania 1.',
    'Russia 1.', 'Saudi 1.', 'Scotland 1.', 'Scotland 2.', 'Scotland 3.',
    'Serbia 1.', 'Serbia 2.', 'Slovakia 1.', 'Slovakia 2.', 'Slovenia 1.',
    'Slovenia 2.', 'South Africa 1.', 'Spain 1.', 'Spain 2.', 'Spain 3.',
    'Sweden 1.', 'Sweden 2.', 'Switzerland 1.', 'Switzerland 2.', 'Tunisia 1.',
    'Turkey 1.', 'Turkey 2.', 'Ukraine 1.', 'UAE 1.', 'USA 1.', 'USA 2.',
    'Uruguay 1.', 'Uzbekistan 1.', 'Venezuela 1.', 'Wales 1.'
]

# League presets for the candidate pool
PRESETS = {
    "Top 5 Europe": [
        'England 1.', 'France 1.', 'Germany 1.', 'Italy 1.', 'Spain 1.'
    ],
    "Top 20 Europe": [
        'England 1.', 'Italy 1.', 'Spain 1.', 'Germany 1.', 'France 1.',
        'England 2.', 'Portugal 1.', 'Belgium 1.',
        'Turkey 1.', 'Germany 2.', 'Spain 2.', 'France 2.',
        'Netherlands 1.', 'Austria 1.', 'Switzerland 1.', 'Denmark 1.', 'Croatia 1.',
        'Italy 2.', 'Czech 1.', 'Norway 1.'
    ],
    "EFL (England 2–4)": ['England 2.', 'England 3.', 'England 4.'],
    "All listed leagues": included_leagues,
    "Custom": None,
}

features = [
    'Defensive duels per 90', 'Aerial duels per 90', 'Aerial duels won, %',
    'Non-penalty goals per 90', 'xG per 90', 'Shots per 90', 'Shots on target, %',
    'Crosses per 90',  'Dribbles per 90', 'Successful dribbles, %',
    'Offensive duels per 90', 'Touches in box per 90', 'Progressive runs per 90',
    'Passes per 90', 'Accurate passes, %', 'xA per 90', 'Smart passes per 90',
    'Passes to final third per 90', 'Passes to penalty area per 90',
    'Deep completions per 90'
]

weight_factors = {
    'Passes per 90': 3,
    'Dribbles per 90': 3,
    'Non-penalty goals per 90': 3,
    'Aerial duels won, %': 2,
    'Aerial duels per 90': 3,
    'xA per 90': 2,
    'xG per 90': 3,
    'Touches in box per 90': 2,
}

DEFAULT_PERCENTILE_WEIGHT = 0.7
DEFAULT_LEAGUE_WEIGHT = 0.3  # default 0.3 per request

league_strengths = {
    'England 1.': 100.00, 'Italy 1.': 97.14, 'Spain 1.': 94.29, 'Germany 1.': 94.29,
    'France 1.': 91.43, 'Brazil 1.': 82.86, 'England 2.': 71.43, 'Portugal 1.': 71.43,
    'Argentina 1.': 71.43, 'Belgium 1.': 68.57, 'Mexico 1.': 68.57, 'Turkey 1.': 65.71,
    'Germany 2.': 65.71, 'Spain 2.': 65.71, 'France 2.': 65.71, 'USA 1.': 65.71,
    'Russia 1.': 65.71, 'Colombia 1.': 62.86, 'Netherlands 1.': 62.86, 'Austria 1.': 62.86,
    'Switzerland 1.': 62.86, 'Denmark 1.': 62.86, 'Croatia 1.': 62.86, 'Japan 1.': 62.86,
    'Korea 1.': 62.86, 'Italy 2.': 62.86, 'Czech 1.': 57.14, 'Norway 1.': 57.14,
    'Poland 1.': 57.14, 'Romania 1.': 57.14, 'Israel 1.': 57.14, 'Algeria 1.': 57.14,
    'Paraguay 1.': 57.14, 'Saudi 1.': 57.14, 'Uruguay 1.': 57.14, 'Morocco 1.': 57.00,
    'Brazil 2.': 56.00, 'Ukraine 1.': 54.29, 'Ecuador 1.': 54.29, 'Spain 3.': 54.29,
    'Scotland 1.': 54.29, 'Chile 1.': 51.43, 'Cyprus 1.': 51.43, 'Portugal 2.': 51.43,
    'Slovakia 1.': 51.43, 'Australia 1.': 51.43, 'Hungary 1.': 51.43, 'Egypt 1.': 51.43,
    'England 3.': 51.43, 'France 3.': 48.00, 'Japan 2.': 48.00, 'Bulgaria 1.': 48.57,
    'Slovenia 1.': 48.57, 'Venezuela 1.': 48.00, 'Germany 3.': 45.71, 'Albania 1.': 44.00,
    'Serbia 1.': 42.86, 'Belgium 2.': 42.86, 'Bosnia 1.': 42.86, 'Kosovo 1.': 42.86,
    'Nigeria 1.': 42.86, 'Azerbaijan 1.': 50.00, 'Bolivia 1.': 50.00, 'Costa Rica 1.': 50.00,
    'South Africa 1.': 50.00, 'UAE 1.': 50.00, 'Georgia 1.': 40.00, 'Finland 1.': 40.00,
    'Italy 3.': 40.00, 'Peru 1.': 40.00, 'Tunisia 1.': 40.00, 'USA 2.': 40.00,
    'Armenia 1.': 40.00, 'North Macedonia 1.': 40.00, 'Qatar 1.': 40.00, 'Uzbekistan 1.': 42.00,
    'Norway 2.': 42.00, 'Kazakhstan 1.': 42.00, 'Poland 2.': 38.00, 'Denmark 2.': 37.00,
    'Czech 2.': 37.14, 'Israel 2.': 37.14, 'Netherlands 2.': 37.14, 'Switzerland 2.': 37.14,
    'Iceland 1.': 34.29, 'Macedonia 1.': 34.29, 'Ireland 1.': 34.29, 'Sweden 2.': 34.29,
    'Germany 4.': 34.29, 'Malta 1.': 30.00, 'Turkey 2.': 31.43, 'Canada 1.': 28.57,
    'England 4.': 28.57, 'Scotland 2.': 28.57, 'Moldova 1.': 28.57, 'Austria 2.': 25.71,
    'Lithuania 1.': 25.71, 'Brazil 3.': 25.00, 'England 7.': 25.00, 'Slovenia 2.': 22.00,
    'Latvia 1.': 22.86, 'Serbia 2.': 20.00, 'Slovakia 2.': 20.00, 'England 9.': 20.00,
    'England 8.': 15.00, 'Montenegro 1.': 14.29, 'Wales 1.': 12.00, 'Portugal 3.': 11.43,
    'Northern Ireland 1.': 11.43, 'England 5.': 11.43, 'Andorra 1.': 10.00, 'Estonia 1.': 8.57,
    'England 10.': 5.00, 'Scotland 3.': 0.00, 'England 6.': 0.00
}

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
with st.sidebar:
    st.header("Leagues")

    # --- Quick toggles like in the other apps ---
    c1, c2, c3 = st.columns(3)
    use_top5 = c1.checkbox("Top-5", value=False)
    use_top20 = c2.checkbox("Top-20", value=False)
    use_efl = c3.checkbox("EFL", value=False)

    toggle_seed = set()
    if use_top5:
        toggle_seed |= set(PRESETS["Top 5 Europe"])
    if use_top20:
        toggle_seed |= set(PRESETS["Top 20 Europe"])
    if use_efl:
        toggle_seed |= set(PRESETS["EFL (England 2–4)"])

    # --- Target selection leagues (choose target from anywhere) ---
    target_leagues = st.multiselect(
        "Target leagues (for choosing the target player)",
        sorted(set(included_leagues) | set(df.get('League', pd.Series([])).dropna().unique())),
        default=included_leagues
    )

    # --- Candidate pool leagues with preset + extras ---
    if 'candidate_leagues' not in st.session_state:
        st.session_state.candidate_leagues = included_leagues.copy()

    preset_name = st.selectbox("Candidate pool preset", list(PRESETS.keys()), index=0)
    if st.button("Apply preset"):
        preset = PRESETS[preset_name]
        if preset is not None:
            st.session_state.candidate_leagues = preset

    extra_leagues = st.multiselect(
        "Extra leagues to include (added to preset)",
        sorted(set(included_leagues) | set(df.get('League', pd.Series([])).dropna().unique())),
        default=[]
    )

    # Base candidate set = current preset in state, union with quick-toggles
    base_candidate = set(st.session_state.candidate_leagues)
    if toggle_seed:
        base_candidate |= toggle_seed

    # Final candidate league set = base ∪ extras
    leagues_selected = sorted(base_candidate | set(extra_leagues))

    st.caption(f"Candidate pool has **{len(leagues_selected)}** leagues.")

    # --- Target player from *target_leagues* ---
    player_names = df[df['League'].isin(target_leagues)]['Player'].dropna().unique()
    target_player = st.selectbox("Target player", sorted(player_names))

    st.header("Filters")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (500, 5000))
    min_age, max_age = st.slider("Age", 14, 45, (16, 33))

    mv_col = 'Market value'
    mv_max_raw = int(np.nanmax(df[mv_col])) if mv_col in df.columns and df[mv_col].notna().any() else 150_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)

    st.markdown("**Market value (€)**")
    use_millions = st.checkbox("Adjust in millions", True)
    if use_millions:
        max_m = int(mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, max_m))
        min_value = mv_min_m * 1_000_000
        max_value = mv_max_m * 1_000_000
        st.caption(f"Selected: €{min_value:,.0f} – €{max_value:,.0f}")
    else:
        min_value, max_value = st.slider("Range (€)", 0, mv_cap, (0, mv_cap), step=100_000)
        st.caption(f"Selected: €{min_value:,.0f} – €{max_value:,.0f}")

    with st.expander("Exact market value range (override)"):
        c1, c2 = st.columns(2)
        min_value = c1.number_input("Min (€)", value=min_value, min_value=0, max_value=mv_cap, step=50_000, format="%d")
        max_value = c2.number_input("Max (€)", value=max_value, min_value=0, max_value=mv_cap, step=50_000, format="%d")
        if min_value > max_value:
            st.warning("Min value is greater than max value; swapping.")
            min_value, max_value = max_value, min_value

    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))

    st.subheader("Weights")
    percentile_weight = st.slider("Percentile weight", 0.0, 1.0, DEFAULT_PERCENTILE_WEIGHT, 0.05)
    actual_value_weight = 1.0 - percentile_weight
    st.caption(f"Actual value weight is set to {actual_value_weight:.2f}")

    # NEW: Toggle for league weight (default ON) and slider default 0.3
    use_league_weight = st.checkbox("Apply league weight adjustment", value=True)
    league_weight = st.slider(
        "League weight (difficulty adjustment)", 0.0, 1.0, DEFAULT_LEAGUE_WEIGHT, 0.05,
        disabled=not use_league_weight
    )
    effective_league_weight = league_weight if use_league_weight else 0.0

    with st.expander("Advanced feature weights"):
        wf = weight_factors.copy()
        wf['Passes per 90']            = st.slider("Passes per 90 weight", 1, 5, wf['Passes per 90'])
        wf['Dribbles per 90']          = st.slider("Dribbles per 90 weight", 1, 5, wf['Dribbles per 90'])
        wf['Non-penalty goals per 90'] = st.slider("Non-penalty goals per 90 weight", 1, 5, wf['Non-penalty goals per 90'])
        wf['Aerial duels per 90']      = st.slider("Aerial duels per 90 weight", 1, 5, wf['Aerial duels per 90'])
        wf['Aerial duels won, %']      = st.slider("Aerial duels won % weight", 1, 5, wf['Aerial duels won, %'])
        wf['xA per 90']                = st.slider("xA per 90 weight", 1, 5, wf['xA per 90'])
        wf['xG per 90']                = st.slider("xG per 90 weight", 1, 5, wf['xG per 90'])
        wf['Touches in box per 90']    = st.slider("Touches in box per 90 weight", 1, 5, wf['Touches in box per 90'])

    top_n = st.number_input("Show top N", min_value=5, max_value=200, value=50, step=5)

# ---------------------------
# COMPUTATION
# ---------------------------
required_cols = {
    'Player','Team','League','Age','Position','Goals','Minutes played','Market value', *features
}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your data is missing required columns: {missing}")
    st.stop()

# Candidate pool (used for scaler + results)
df_candidates = df[df['League'].isin(leagues_selected)].copy()
df_candidates = df_candidates.dropna(subset=features)
df_candidates = df_candidates[df_candidates['Goals'] > 0]
df_candidates = df_candidates[df_candidates['Dribbles per 90'] > 0]
df_candidates = df_candidates[df_candidates['Position'].astype(str).str.startswith(('CF',))]

# Target row (can come from leagues outside candidate pool)
df_target_pool = df[df['League'].isin(target_leagues)].copy()
if target_player not in df_target_pool['Player'].values:
    st.warning("Target player not found in target leagues. Adjust 'Target leagues'.")
    st.stop()

target_row_full = df_target_pool.loc[df_target_pool['Player'] == target_player].iloc[0]
target_league = target_row_full['League']

# Safety: ensure candidate pool not empty
if df_candidates.empty:
    st.warning("No candidates after filters/preset. Widen candidate leagues or relax filters.")
    st.stop()

# Build arrays
target_features = target_row_full[features].values.reshape(1, -1)
target_percentiles_in_candidates = (
    df_candidates.groupby('League')[features]
    .rank(pct=True)
)

# Percentiles for all candidates
percentile_ranks = target_percentiles_in_candidates.values

# Percentiles for the target — compute over its own league (full df, so it's stable)
target_percentiles = (
    df.groupby('League')[features]
    .rank(pct=True)
    .loc[df['Player'] == target_player]
    .values
)

# Feature weights
weights = np.array([wf.get(f, 1) for f in features], dtype=float)

# Standardize over the candidate pool
scaler = StandardScaler()
standardized_features = scaler.fit_transform(df_candidates[features])
target_features_standardized = scaler.transform(target_features)

# Distances
percentile_distances = np.linalg.norm((percentile_ranks - target_percentiles) * weights, axis=1)
actual_value_distances = np.linalg.norm((standardized_features - target_features_standardized) * weights, axis=1)

combined = percentile_distances * percentile_weight + actual_value_distances * (1.0 - percentile_weight)

# Normalize to similarity 0..100
norm = (combined - np.min(combined)) / (np.ptp(combined) if np.ptp(combined) != 0 else 1.0)
similarities = ((1 - norm) * 100).round(2)

# Assemble result frame
similarity_df = df_candidates.copy()
similarity_df['Similarity'] = similarities
similarity_df = similarity_df[similarity_df['Player'] != target_player]

# User filters
similarity_df = similarity_df[
    (similarity_df['Market value'] >= min_value) &
    (similarity_df['Market value'] <= max_value) &
    (similarity_df['Minutes played'] >= min_minutes) &
    (similarity_df['Minutes played'] <= max_minutes) &
    (similarity_df['Age'] >= min_age) &
    (similarity_df['Age'] <= max_age)
]

# League strength & range
similarity_df['League strength'] = similarity_df['League'].map(league_strengths).fillna(0.0)
target_league_strength = league_strengths.get(target_league, 1.0)

similarity_df = similarity_df[
    (similarity_df['League strength'] >= float(min_strength)) &
    (similarity_df['League strength'] <= float(max_strength))
]

# Difficulty adjustment with toggle
league_ratio = (similarity_df['League strength'] / target_league_strength).clip(lower=0.5, upper=1.2)
similarity_df['Adjusted Similarity'] = (
    similarity_df['Similarity'] * (1 - effective_league_weight) +
    similarity_df['Similarity'] * league_ratio * effective_league_weight
)

# Rank
similarity_df = similarity_df.sort_values('Adjusted Similarity', ascending=False).reset_index(drop=True)
similarity_df.insert(0, 'Rank', np.arange(1, len(similarity_df) + 1))

# ---------------------------
# UI OUTPUT
# ---------------------------
st.subheader(f"Similar to: {target_player} — target league {target_league} (strength {target_league_strength:.2f})")

cols_to_show = ['Rank', 'Player', 'Team', 'League', 'Age', 'Minutes played',
                'Market value', 'League strength', 'Similarity', 'Adjusted Similarity']
cols_to_show = [c for c in cols_to_show if c in similarity_df.columns]

st.dataframe(similarity_df[cols_to_show].head(int(top_n)), use_container_width=True)

csv = similarity_df[cols_to_show].to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download full results (CSV)", data=csv, file_name="similarity_results.csv", mime="text/csv")

with st.expander("Debug / Repro details"):
    st.write({
        "candidate_preset": preset_name,
        "toggles": {"Top-5": use_top5, "Top-20": use_top20, "EFL": use_efl},
        "candidate_leagues_final_count": len(leagues_selected),
        "extra_leagues_selected": extra_leagues,
        "target_leagues_count": len(target_leagues),
        "percentile_weight": float(percentile_weight),
        "use_league_weight": bool(use_league_weight),
        "league_weight": float(league_weight if use_league_weight else 0.0),
        "target_league_strength": float(target_league_strength),
        "n_candidates": int(len(similarity_df)),
        "market_value_range": (int(min_value), int(max_value)),
        "minutes_range": (int(min_minutes), int(max_minutes)),
    })




