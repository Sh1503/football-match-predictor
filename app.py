
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Football Predictor", layout="centered")
st.title("⚽ Football Match Predictor")

# --- Load Data ---
@st.cache_data
def load_league_data():
    leagues = {
        "Premier League": "epl.csv",
        "La Liga": "laliga.csv",
        "Serie A": "seriea.csv",
        "Bundesliga": "bundesliga.csv",
        "Ligue 1": "ligue1.csv"
    }
    data = {}
    for league, file in leagues.items():
        df = pd.read_csv(file)
        df = df.dropna(subset=['FTHG', 'FTAG'])
        data[league] = df
    return data

data = load_league_data()
league = st.selectbox("Select a League", list(data.keys()))
df = data[league]

teams = sorted(set(df['HomeTeam']).union(df['AwayTeam']))
home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", [t for t in teams if t != home_team])

# --- Compute averages ---
def get_team_avg(df, team, is_home):
    if is_home:
        matches = df[df['HomeTeam'] == team]
        return matches['FTHG'].mean()
    else:
        matches = df[df['AwayTeam'] == team]
        return matches['FTAG'].mean()

home_avg = get_team_avg(df, home_team, is_home=True)
away_avg = get_team_avg(df, away_team, is_home=False)

if st.button("Predict Outcome"):
    st.markdown(f"📊 **{home_team} home avg goals:** {home_avg:.2f}")
    st.markdown(f"📊 **{away_team} away avg goals:** {away_avg:.2f}")

    if abs(home_avg - away_avg) < 0.2:
        prediction = "🤝 Draw"
    elif home_avg > away_avg:
        prediction = f"🏠 {home_team} likely to win"
    else:
        prediction = f"🚗 {away_team} likely to win"

    st.success(f"🔮 Prediction: {prediction}")
