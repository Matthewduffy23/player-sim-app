# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Player Similarity Finder", layout="wide")

# --- Load dataset directly from repo ---
df = pd.read_csv("WORLDJUNE25.csv")

st.title("⚽ Player Similarity Finder")


# ---------------------------
# 2) CONSTANTS (same as your notebook)
# ---------------------------
included_leagues = [ 'England 1.', 'England 2.', 'England 3.', 'England 4.', 'England 5.',
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

features = [
    'Defensive duels per 90', 'Aerial duels per 90', 'Aerial duels won, %', 
    'Non-penalty goals per 90', 'xG per 90', 'Shots per 90', 'Shots on target, %',
    'Crosses per 90',  'Dribbles per 90', 'Successful dribbles, %',
    'Offensive duels per 90', 'Touches in box per 90', 'Progressive runs per 90',
    'Passes per 90', 'Accurate passes, %', 'xA per 90', 'Smart passes per 90', 
    'Passes to final third per 90', 'Passes to penalty area per 90', 
    'Deep completions per 90'
]

# NOTE: fixed key to match the feature name exactly: "Non-penalty goals per 90"
weight_factors = {
    'Passes per 90': 3, 
    'Dribbles per 90': 3,
    'Non-penalty goals per 90': 3,
    'Aerial duels won, %': 1, 
    'Aerial duels per 90': 3,
    'xA per 90': 2, 
    'xG per 90': 3,
    'Touches in box per 90': 2,
}

# Defaults
DEFAULT_PERCENTILE_WEIGHT = 0.7
DEFAULT_ACTUAL_WEIGHT = 0.3
DEFAULT_LEAGUE_WEIGHT = 0.2

league_strengths = {
'England 1.': 100.00,
    'Italy 1.': 97.14,
    'Spain 1.': 94.29,
    'Germany 1.': 94.29,
    'France 1.': 91.43,
    'Brazil 1.': 82.86,
    'England 2.': 71.43,
    'Portugal 1.': 71.43,
    'Argentina 1.': 71.43,
    'Belgium 1.': 68.57,
    'Mexico 1.': 68.57,
    'Turkey 1.': 65.71,
    'Germany 2.': 65.71,
    'Spain 2.': 65.71,
    'France 2.': 65.71,
    'USA 1.': 65.71,
    'Russia 1.': 65.71,
    'Colombia 1.': 62.86,
    'Netherlands 1.': 62.86,
    'Austria 1.': 62.86,
    'Switzerland 1.': 62.86,
    'Denmark 1.': 62.86,
    'Croatia 1.': 62.86,
    'Japan 1.': 62.86,
    'Korea 1.': 62.86,
    'Italy 2.': 62.86,
    'Czech 1.': 57.14,
    'Norway 1.': 57.14,
    'Poland 1.': 57.14,
    'Romania 1.': 57.14,
    'Israel 1.': 57.14,
    'Algeria 1.': 57.14,
    'Paraguay 1.': 57.14,
    'Saudi 1.': 57.14,
    'Uruguay 1.': 57.14,
    'Morocco 1.': 57.00,
    'Brazil 2.': 56.00,
    'Ukraine 1.': 54.29,
    'Ecuador 1.': 54.29,
    'Spain 3.': 54.29,
    'Scotland 1.': 54.29,
    'Chile 1.': 51.43,
    'Cyprus 1.': 51.43,
    'Portugal 2.': 51.43,
    'Slovakia 1.': 51.43,
    'Australia 1.': 51.43,
    'Hungary 1.': 51.43,
    'Egypt 1.': 51.43,
    'England 3.': 51.43,
    'France 3.': 48.00,
    'Japan 2.': 48.00,
    'Bulgaria 1.': 48.57,
    'Slovenia 1.': 48.57,
    'Venezuela 1.': 48.00,
    'Germany 3.': 45.71,
    'Albania 1.': 44.00,
    'Serbia 1.': 42.86,
    'Belgium 2.': 42.86,
    'Bosnia 1.': 42.86,
    'Kosovo 1.': 42.86,
    'Nigeria 1.': 42.86,
    'Azerbaijan 1.': 50.00,
    'Bolivia 1.': 50.00,
    'Costa Rica 1.': 50.00,
    'South Africa 1.': 50.00,
    'UAE 1.': 50.00,
    'Georgia 1.': 40.00,
    'Finland 1.': 40.00,
    'Italy 3.': 40.00,
    'Peru 1.': 40.00,
    'Tunisia 1.': 40.00,
    'USA 2.': 40.00,
    'Armenia 1.': 40.00,
    'North Macedonia 1.': 40.00,
    'Qatar 1.': 40.00,
    'Uzbekistan 1.': 42.00,
    'Norway 2.': 42.00,
    'Kazakhstan 1.': 42.00,
    'Poland 2.': 38.00,
    'Denmark 2.': 37.00,
    'Czech 2.': 37.14,
    'Israel 2.': 37.14,
    'Netherlands 2.': 37.14,
    'Switzerland 2.': 37.14,
    'Iceland 1.': 34.29,
    'Macedonia 1.': 34.29,
    'Ireland 1.': 34.29,
    'Sweden 2.': 34.29,
    'Germany 4.': 34.29,
    'Malta 1.': 30.00,
    'Turkey 2.': 31.43,
    'Canada 1.': 28.57,
    'England 4.': 28.57,
    'Scotland 2.': 28.57,
    'Moldova 1.': 28.57,
    'Austria 2.': 25.71,
    'Lithuania 1.': 25.71,
    'Brazil 3.': 25.00,
    'England 7.': 25.00,
    'Slovenia 2.': 22.00,
    'Latvia 1.': 22.86,
    'Serbia 2.': 20.00,
    'Slovakia 2.': 20.00,
    'England 9.': 20.00,
    'England 8.': 15.00,
    'Montenegro 1.': 14.29,
    'Wales 1.': 12.00,
    'Portugal 3.': 11.43,
    'Northern Ireland 1.': 11.43,
    'England 5.': 11.43,
    'Andorra 1.': 10.00,
    'Estonia 1.': 8.57,
    'England 10.': 5.00,
    'Scotland 3.': 0.00,
    'England 6.': 0.00
}

# ---------------------------
# 3) SIDEBAR CONTROLS
# ---------------------------
with st.sidebar:
    st.header("Controls")

    leagues_selected = st.multiselect(
        "Leagues included",
        sorted(list(set(included_leagues) | set(df.get('League', pd.Series([])).unique()))),
        default=included_leagues
    )

    player_names = df[df['League'].isin(leagues_selected)]['Player'].dropna().unique()
    target_player = st.selectbox("Target player", sorted(player_names))

    min_minutes, max_minutes = st.slider("Minutes played", 0, 12000, (500, 999999))
    min_age, max_age = st.slider("Age", 14, 45, (16, 33))
    min_value, max_value = st.slider(
    "Market value (€)",
    0, 150_000_000, (0, 150_000_000)
)


    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))

    st.subheader("Weights")
    percentile_weight = st.slider("Percentile weight", 0.0, 1.0, DEFAULT_PERCENTILE_WEIGHT, 0.05)
    actual_value_weight = 1.0 - percentile_weight
    st.caption(f"Actual value weight is set to {actual_value_weight:.2f} (1 - percentile weight)")

    league_weight = st.slider("League weight (difficulty adjustment)", 0.0, 1.0, DEFAULT_LEAGUE_WEIGHT, 0.05)

    with st.expander("Advanced feature weights"):
        # Build sliders only for the features you custom-weighted
        wf = weight_factors.copy()
        wf['Passes per 90'] = st.slider("Passes per 90 weight", 1, 5, wf['Passes per 90'])
        wf['Dribbles per 90'] = st.slider("Dribbles per 90 weight", 1, 5, wf['Dribbles per 90'])
        wf['Non-penalty goals per 90'] = st.slider("Non-penalty goals per 90 weight", 1, 5, wf['Non-penalty goals per 90'])
        wf['Aerial duels per 90'] = st.slider("Aerial duels per 90 weight", 1, 5, wf['Aerial duels per 90'])
        wf['Aerial duels won, %'] = st.slider("Aerial duels won % weight", 1, 5, wf['Aerial duels won, %'])
        wf['xA per 90'] = st.slider("xA per 90 weight", 1, 5, wf['xA per 90'])
        wf['xG per 90'] = st.slider("xG per 90 weight", 1, 5, wf['xG per 90'])
        wf['Touches in box per 90'] = st.slider("Touches in box per 90 weight", 1, 5, wf['Touches in box per 90'])

    top_n = st.number_input("Show top N", min_value=5, max_value=200, value=50, step=5)

# ---------------------------
# 4) COMPUTATION
# ---------------------------
required_cols = {
    'Player','Team','League','Age','Position','Goals','Minutes played','Market value',
    *features
}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your data is missing required columns: {missing}")
    st.stop()

# Filter leagues first so the percentiles are computed inside the chosen competitions
df_filtered = df[df['League'].isin(leagues_selected)].copy()

# Cleanup & basic eligibility filters
df_filtered = df_filtered.dropna(subset=features)
df_filtered = df_filtered[df_filtered['Goals'] > 0]
df_filtered = df_filtered[df_filtered['Dribbles per 90'] > 0]
df_filtered = df_filtered[df_filtered['Position'].astype(str).str.startswith(('CF',))]

if target_player not in df_filtered['Player'].values:
    st.warning("Target player not found in the filtered set. Try expanding leagues or filters.")
    st.stop()

# Target vectors
target_features = df_filtered.loc[df_filtered['Player'] == target_player, features].values
target_percentiles = (
    df_filtered.groupby('League')[features]
    .rank(pct=True)
    .loc[df_filtered['Player'] == target_player]
    .values
)

# Feature weights array
weights = np.array([wf.get(f, 1) for f in features], dtype=float)

# Standardize actual values
scaler = StandardScaler()
standardized_features = scaler.fit_transform(df_filtered[features])
target_features_standardized = scaler.transform(target_features)

# Distances
percentile_ranks = df_filtered.groupby('League')[features].rank(pct=True).values
percentile_distances = np.linalg.norm((percentile_ranks - target_percentiles) * weights, axis=1)
actual_value_distances = np.linalg.norm((standardized_features - target_features_standardized) * weights, axis=1)

combined = percentile_distances * percentile_weight + actual_value_distances * actual_value_weight
# Normalize to similarity 0..100
norm = (combined - np.min(combined)) / (np.ptp(combined) if np.ptp(combined) != 0 else 1.0)
similarities = ((1 - norm) * 100).round(2)

# Build frame
similarity_df = df_filtered.copy()
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
target_league = df_filtered.loc[df_filtered['Player'] == target_player, 'League'].iloc[0]
target_league_strength = league_strengths.get(target_league, 1.0)

similarity_df = similarity_df[
    (similarity_df['League strength'] >= float(min_strength)) &
    (similarity_df['League strength'] <= float(max_strength))
]

# League difficulty adjustment
league_ratio = (similarity_df['League strength'] / target_league_strength).clip(lower=0.5, upper=1.2)
similarity_df['Adjusted Similarity'] = (
    similarity_df['Similarity'] * (1 - league_weight) +
    similarity_df['Similarity'] * league_ratio * league_weight
)

# Rank and display
similarity_df = similarity_df.sort_values('Adjusted Similarity', ascending=False).reset_index(drop=True)
similarity_df.insert(0, 'Rank', np.arange(1, len(similarity_df) + 1))

# ---------------------------
# 5) UI OUTPUT
# ---------------------------
st.subheader(f"Similar to: {target_player}  —  League: {target_league} (strength {target_league_strength:.2f})")

cols_to_show = ['Rank', 'Player', 'Team', 'League', 'Age', 'Minutes played',
                'Market value', 'League strength', 'Similarity', 'Adjusted Similarity']
cols_to_show = [c for c in cols_to_show if c in similarity_df.columns]

st.dataframe(
    similarity_df[cols_to_show].head(int(top_n)),
    use_container_width=True
)

csv = similarity_df[cols_to_show].to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download full results (CSV)", data=csv, file_name="similarity_results.csv", mime="text/csv")

with st.expander("Debug / Repro details"):
    st.write({
        "percentile_weight": percentile_weight,
        "actual_value_weight": actual_value_weight,
        "league_weight": league_weight,
        "target_league_strength": float(target_league_strength),
        "n_candidates": int(len(similarity_df))
    })
