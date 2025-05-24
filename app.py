import streamlit as st
import pandas as pd

st.set_page_config(page_title="Football Predictor", layout="centered")
st.title("⚽ Football Match Predictor")

# --- קבוצות קבועות לכל ליגה ---
LEAGUE_TEAMS = {
    "Premier League": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
        "Liverpool", "Luton", "Manchester City", "Manchester United",
        "Newcastle", "Nottingham Forest", "Sheffield United", "Tottenham",
        "West Ham", "Wolves"
    ],
    "La Liga": [
        "Alavés", "Almería", "Athletic Club", "Atlético Madrid", "Barcelona",
        "Cádiz", "Celta Vigo", "Getafe", "Girona", "Granada", "Las Palmas",
        "Mallorca", "Osasuna", "Rayo Vallecano", "Real Betis", "Real Madrid",
        "Real Sociedad", "Sevilla", "Valencia", "Villarreal"
    ],
    "Serie A": [
        "Atalanta", "Bologna", "Cagliari", "Empoli", "Fiorentina", "Frosinone",
        "Genoa", "Hellas Verona", "Inter", "Juventus", "Lazio", "Lecce",
        "Milan", "Monza", "Napoli", "Roma", "Salernitana", "Sassuolo",
        "Torino", "Udinese"
    ],
    "Bundesliga": [
        "Augsburg", "Bayer Leverkusen", "Bayern Munich", "Bochum", "Borussia Dortmund",
        "Borussia M'gladbach", "Darmstadt", "Eintracht Frankfurt", "Freiburg",
        "Heidenheim", "Hoffenheim", "Mainz", "RB Leipzig", "Stuttgart",
        "Union Berlin", "Werder Bremen", "Wolfsburg", "Köln"
    ],
    "Ligue 1": [
        "Brest", "Clermont", "Le Havre", "Lens", "Lille", "Lorient",
        "Lyon", "Marseille", "Metz", "Monaco", "Montpellier", "Nantes",
        "Nice", "Paris SG", "Reims", "Rennes", "Strasbourg", "Toulouse"
    ]
}

# --- טעינת נתוני משחקים ---
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
teams = sorted(LEAGUE_TEAMS[league])

home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", [t for t in teams if t != home_team])

# --- חישוב ממוצעים ---
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
