import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Football Predictor Pro", layout="centered", page_icon="⚽")
st.title("⚽ Football Match Predictor Pro")

# ----------------------------
# Backtesting System
# ----------------------------
def backtest_strategy(dataframe, confidence=0.6):
    correct = 0
    total_bets = 0
    
    for index, match in dataframe.iterrows():
        try:
            prediction = predict_match(match['HomeTeam'], match['AwayTeam'], dataframe)
            actual_result = match['FTR']
            
            if prediction['home_win'] > confidence:
                total_bets += 1
                if actual_result == 'H':
                    correct += 1
            elif prediction['away_win'] > confidence:
                total_bets += 1
                if actual_result == 'A':
                    correct += 1
        except:
            continue
                
    return correct, total_bets

# ----------------------------
# הגדרות קבועות
# ----------------------------
LEAGUE_TEAMS = {
    "Premier League": ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Burnley", "Chelsea", 
                      "Crystal Palace", "Everton", "Fulham", "Liverpool", "Luton", "Manchester City", 
                      "Manchester United", "Newcastle", "Nottingham Forest", "Sheffield United", "Tottenham", 
                      "West Ham", "Wolves"],
    "La Liga": ["Alavés", "Almería", "Athletic Club", "Atlético Madrid", "Barcelona", "Cádiz", "Celta Vigo", 
               "Getafe", "Girona", "Granada", "Las Palmas", "Mallorca", "Osasuna", "Rayo Vallecano", 
               "Real Betis", "Real Madrid", "Real Sociedad", "Sevilla", "Valencia", "Villarreal"],
    "Serie A": ["Atalanta", "Bologna", "Cagliari", "Empoli", "Fiorentina", "Frosinone", "Genoa", "Hellas Verona", 
               "Inter", "Juventus", "Lazio", "Lecce", "Milan", "Monza", "Napoli", "Roma", "Salernitana", 
               "Sassuolo", "Torino", "Udinese"],
    "Bundesliga": ["Augsburg", "Bayer Leverkusen", "Bayern Munich", "Bochum", "Borussia Dortmund", 
                  "Borussia M'gladbach", "Darmstadt", "Eintracht Frankfurt", "Freiburg", "Heidenheim", 
                  "Hoffenheim", "Mainz", "RB Leipzig", "Stuttgart", "Union Berlin", "Werder Bremen", 
                  "Wolfsburg", "Köln"],
    "Ligue 1": ["Brest", "Clermont", "Le Havre", "Lens", "Lille", "Lorient", "Lyon", "Marseille", "Metz", 
               "Monaco", "Montpellier", "Nantes", "Nice", "Paris SG", "Reims", "Rennes", "Strasbourg", "Toulouse"]
}

# ----------------------------
# טעינת נתונים
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
            df = df.dropna(subset=['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST'])
            data[league] = df
        except Exception as e:
            st.error(f"שגיאה בטעינת קובץ {file}: {str(e)}")
    return data

data = load_league_data()

# ----------------------------
# פונקציות חיזוי מתקדמות
# ----------------------------
def dixon_coles_correction(home_avg, away_avg, rho=0.2):
    return home_avg * (1 + rho * away_avg), away_avg * (1 + rho * home_avg)

def predict_match(home_team, away_team, df):
    home_matches = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)]
    away_matches = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)]
    
    home_avg = home_matches['FTHG'].mean() if not home_matches.empty else 0
    away_avg = away_matches['FTAG'].mean() if not away_matches.empty else 0
    
    home_adj, away_adj = dixon_coles_correction(home_avg, away_avg)
    
    home_prob = sum(poisson.pmf(i, home_adj) for i in range(0, 6))
    away_prob = sum(poisson.pmf(i, away_adj) for i in range(0, 6))
    
    return {
        'home_win': home_prob * (1 - away_prob),
        'draw': home_prob * away_prob,
        'away_win': (1 - home_prob) * away_prob,
        'home_xG': home_adj,
        'away_xG': away_adj
    }

# ----------------------------
# ממשק משתמש
# ----------------------------
league = st.selectbox("בחר ליגה", list(data.keys()))
df = data[league]
teams = sorted(LEAGUE_TEAMS[league])

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("קבוצה ביתית", teams)
with col2:
    away_team = st.selectbox("קבוצה אורחת", [t for t in teams if t != home_team])

# ----------------------------
# Backtesting Section - מיקום מתוקן!
# ----------------------------
with st.expander("🔄 בדוק את האסטרטגיה על עונות קודמות"):
    confidence_level = st.slider("בחר סף ביטחון להימור (%)", 50, 90, 60)
    if st.button("הרץ סימולציית הימורים"):
        correct, total = backtest_strategy(df, confidence_level/100)
        if total > 0:
            st.success(f"אחוז הצלחה: {correct}/{total} ({correct/total*100:.1f}%)")
        else:
            st.warning("לא נמצאו הימורים שעברו את סף הביטחון")

# ----------------------------
# ניתוח סטטיסטי
# ----------------------------
with st.expander("📊 ניתוח סטטיסטי מתקדם"):
    try:
        home_stats = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)]
        away_stats = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"ממוצע שערים ({home_team})", f"{home_stats['FTHG'].mean():.2f}")
            st.metric("בעיטות למשחק", f"{home_stats['HS'].mean():.1f}")
        with col2:
            st.metric(f"ממוצע שערים ({away_team})", f"{away_stats['FTAG'].mean():.2f}")
            st.metric("בעיטות למשחק", f"{away_stats['AS'].mean():.1f}")
        with col3:
            st.metric("מפגשים היסטוריים", len(df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)]))
        
        fig = px.line_polar(
            r=[home_stats['HST'].mean(), away_stats['AST'].mean(), home_stats['HC'].mean(), away_stats['AC'].mean()],
            theta=['בעיטות למסגרת', 'בעיטות למסגרת', 'קרנות', 'קרנות'],
            line_close=True
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"שגיאה בניתוח סטטיסטי: {str(e)}")

# ----------------------------
# חיזוי והמלצות
# ----------------------------
if st.button("בצע חיזוי"):
    try:
        prediction = predict_match(home_team, away_team, df)
        
        st.subheader("תוצאות חיזוי")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ניצחון בית", f"{prediction['home_win']*100:.1f}%")
        with col2:
            st.metric("תיקו", f"{prediction['draw']*100:.1f}%")
        with col3:
            st.metric("ניצחון חוץ", f"{prediction['away_win']*100:.1f}%")
        
        st.subheader("המלצות הימורים")
        if prediction['home_win'] > 0.6:
            st.success(f"המלצה: הימור על ניצחון ביתי עם ביטחון {prediction['home_win']*100:.0f}%")
        elif prediction['away_win'] > 0.55:
            st.success(f"המלצה: הימור על ניצחון חוץ עם ביטחון {prediction['away_win']*100:.0f}%")
        else:
            st.info("אין המלצת הימור ברורה - שקול להימנע מהימור")
            
    except Exception as e:
        st.error(f"שגיאה בחיזוי: {str(e)}")

# ----------------------------
# עדכון תאריך
# ----------------------------
st.markdown(f"*נתונים מעודכנים לאחרונה: {datetime.now().strftime('%d/%m/%Y %H:%M')}*")
