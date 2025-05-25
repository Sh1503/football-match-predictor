import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime

st.set_page_config(page_title="Football Predictor Pro", layout="centered", page_icon="⚽")
st.title("⚽ Football Match Predictor Pro")

# ----------------------------
# קבועים: קבוצות לפי ליגה
# ----------------------------
LEAGUE_TEAMS = {
    "Premier League": [...],  # קיצרתי כאן - שים את הרשימה המלאה שלך
    "La Liga": [...],
    "Serie A": [...],
    "Bundesliga": [...],
    "Ligue 1": [...]
}

# ----------------------------
# טעינת קבצי CSV
# ----------------------------
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
        try:
            df = pd.read_csv(file)
            data[league] = df
        except Exception as e:
            st.warning(f"לא ניתן לטעון נתונים עבור {league}: {e}")
    return data

# ----------------------------
# פונקציית חיזוי בהתבסס על פואסון
# ----------------------------
def predict_match(home, away, df):
    home_data = df[df['HomeTeam'] == home]
    away_data = df[df['AwayTeam'] == away]

    if home_data.empty or away_data.empty:
        raise ValueError("אין מספיק נתונים לקבוצות שנבחרו.")

    home_goals_avg = home_data['FTHG'].mean()
    away_goals_avg = away_data['FTAG'].mean()

    home_attack = home_goals_avg
    away_attack = away_goals_avg

    max_goals = 5
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    home_win_prob = draw_prob = away_win_prob = 0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, home_attack) * poisson.pmf(j, away_attack)
            matrix[i][j] = p
            if i > j:
                home_win_prob += p
            elif i == j:
                draw_prob += p
            else:
                away_win_prob += p

    return {
        "home_win": round(home_win_prob, 3),
        "draw": round(draw_prob, 3),
        "away_win": round(away_win_prob, 3)
    }

# ----------------------------
# פונקציית בדיקה לאחור (Backtest)
# ----------------------------
def backtest_strategy(df, confidence=0.6):
    correct = 0
    total_bets = 0

    for _, match in df.iterrows():
        try:
            prediction = predict_match(match['HomeTeam'], match['AwayTeam'], df)
            result = match['FTR']
            if prediction['home_win'] > confidence:
                total_bets += 1
                if result == 'H':
                    correct += 1
            elif prediction['away_win'] > confidence:
                total_bets += 1
                if result == 'A':
                    correct += 1
        except:
            continue

    if total_bets == 0:
        return 0, 0, 0
    accuracy = round(correct / total_bets * 100, 2)
    return correct, total_bets, accuracy

# ----------------------------
# ממשק משתמש
# ----------------------------
data = load_league_data()

selected_league = st.selectbox("בחר ליגה", list(LEAGUE_TEAMS.keys()))

if selected_league:
    teams = LEAGUE_TEAMS[selected_league]
    home_team = st.selectbox("Home Team", teams, key="home")
    away_team = st.selectbox("Away Team", [team for team in teams if team != home_team], key="away")

    if st.button("חשב חיזוי"):
        df = data[selected_league]
        try:
            prediction = predict_match(home_team, away_team, df)
            st.subheader("🔮 תוצאות חיזוי:")
            st.write(f"ניצחון לקבוצה הביתית **{home_team}**: {prediction['home_win']*100:.1f}%")
            st.write(f"תיקו: {prediction['draw']*100:.1f}%")
            st.write(f"ניצחון לקבוצה האורחת **{away_team}**: {prediction['away_win']*100:.1f}%")
        except Exception as e:
            st.error(f"שגיאה: {e}")

    st.markdown("---")
    st.subheader("📊 Backtesting על עונת עבר")
    confidence = st.slider("רף ביטחון להימור (אחוז)", 50, 90, 60)
    if st.button("הרץ Backtest"):
        df = data[selected_league]
        correct, total_bets, acc = backtest_strategy(df, confidence=confidence/100)
        st.write(f"ניחושים נכונים: {correct} מתוך {total_bets}")
        st.write(f"אחוז הצלחה: **{acc}%**")

