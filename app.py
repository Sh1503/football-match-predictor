import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# הגדרות דף
st.set_page_config(
    page_title="Football Predictor Pro",
    page_icon="⚽",
    layout="centered"
)
st.title("⚽ Football Match Predictor Pro")

# ----------------------------
# קבוצות לפי ליגה (מותאם לקבצי ה-CSV שלך)
# ----------------------------
LEAGUE_TEAMS = {
    'Bundesliga': [
        'Augsburg', 'Bayern Munich', 'Bochum', 'Dortmund', 'Ein Frankfurt',
        'Freiburg', 'Heidenheim', 'Hoffenheim', 'Holstein Kiel', 'Leverkusen',
        "M'gladbach", 'Mainz', 'RB Leipzig', 'St Pauli', 'Stuttgart',
        'Union Berlin', 'Werder Bremen', 'Wolfsburg'
    ],
    'Premier League': [
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
        'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
        "Nott'm Forest", 'Southampton', 'Tottenham', 'West Ham', 'Wolves'
    ],
    'La Liga': [
        'Alaves', 'Ath Bilbao', 'Ath Madrid', 'Barcelona', 'Betis', 'Celta',
        'Espanol', 'Getafe', 'Girona', 'Las Palmas', 'Leganes', 'Mallorca',
        'Osasuna', 'Real Madrid', 'Sevilla', 'Sociedad', 'Valencia',
        'Valladolid', 'Vallecano', 'Villarreal'
    ],
    'Ligue 1': [
        'Angers', 'Auxerre', 'Brest', 'Le Havre', 'Lens', 'Lille', 'Lyon',
        'Marseille', 'Monaco', 'Montpellier', 'Nantes', 'Nice', 'Paris SG',
        'Reims', 'Rennes', 'St Etienne', 'Strasbourg', 'Toulouse'
    ],
    'Serie A': [
        'Atalanta', 'Bologna', 'Cagliari', 'Como', 'Empoli', 'Fiorentina',
        'Genoa', 'Inter', 'Juventus', 'Lazio', 'Lecce', 'Milan', 'Monza',
        'Napoli', 'Parma', 'Roma', 'Torino', 'Udinese', 'Venezia', 'Verona'
    ]
}

# ----------------------------
# טעינת נתונים
# ----------------------------
@st.cache_data
def load_league_data():
    return {
        "Premier League": pd.read_csv("epl.csv"),
        "La Liga": pd.read_csv("laliga.csv"),
        "Serie A": pd.read_csv("seriea.csv"),
        "Bundesliga": pd.read_csv("bundesliga.csv"),
        "Ligue 1": pd.read_csv("ligue1.csv")
    }

# ----------------------------
# חיזוי תוצאות
# ----------------------------
def predict_match(home_team, away_team, df):
    # חישוב ממוצעי שערים
    home_avg = df[df['HomeTeam'] == home_team]['FTHG'].mean()
    away_avg = df[df['AwayTeam'] == away_team]['FTAG'].mean()
    
    # חישוב הסתברויות עם התפלגות פואסון
    max_goals = 5
    home_win = draw = away_win = 0.0
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, home_avg) * poisson.pmf(j, away_avg)
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
    
    return {
        "home_win": round(home_win, 3),
        "draw": round(draw, 3),
        "away_win": round(away_win, 3)
    }

# ----------------------------
# בדיקת ביצועי המודל
# ----------------------------
def backtest_strategy(df, confidence=0.6):
    correct = total = 0
    for _, row in df.iterrows():
        try:
            pred = predict_match(row['HomeTeam'], row['AwayTeam'], df)
            actual = row['FTR']
            
            if pred['home_win'] > confidence and actual == 'H':
                correct += 1
                total += 1
            elif pred['away_win'] > confidence and actual == 'A':
                correct += 1
                total += 1
        except:
            continue
    
    return correct, total, round((correct / total) * 100, 2) if total > 0 else 0

# ----------------------------
# ממשק משתמש
# ----------------------------
data = load_league_data()
selected_league = st.selectbox("בחר ליגה", options=list(LEAGUE_TEAMS.keys()))

if selected_league:
    teams = LEAGUE_TEAMS[selected_league]
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("קבוצה ביתית", options=teams)
    
    with col2:
        away_team = st.selectbox("קבוצה אורחת", options=[t for t in teams if t != home_team])
    
    if st.button("חשב חיזוי ⚡"):
        try:
            prediction = predict_match(home_team, away_team, data[selected_league])
            st.subheader("תוצאות החיזוי:")
            st.metric(label=f"ניצחון ל־{home_team}", value=f"{prediction['home_win']*100:.1f}%")
            st.metric(label="תיקו", value=f"{prediction['draw']*100:.1f}%")
            st.metric(label=f"ניצחון ל־{away_team}", value=f"{prediction['away_win']*100:.1f}%")
        except Exception as e:
            st.error(f"שגיאה: {str(e)}")
    
    st.divider()
    st.subheader("בדיקת דיוק המודל")
    confidence = st.slider("רף ביטחון (%)", 50, 90, 60, help="המודל יחשב רק ניחושים עם הסתברות מעל ערך זה")
    
    if st.button("הרץ בדיקה"):
        correct, total, acc = backtest_strategy(data[selected_league], confidence/100)
        st.write(f"**ניחושים נכונים:** {correct} מתוך {total}")
        st.write(f"**דיוק:** {acc}%")
