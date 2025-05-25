import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Football Predictor Pro", layout="centered", page_icon="⚽")
st.title("⚽ Football Match Predictor Pro")

# ----------------------------
# Backtesting System (מתוקן)
# ----------------------------
def backtest_strategy(dataframe, confidence=0.6):
    correct = 0
    total_bets = 0
    
    for index, match in dataframe.iterrows():
        try:
            # חיזוי תוצאה
            prediction = predict_match(match['HomeTeam'], match['AwayTeam'], dataframe)
            
            # בדיקת תוצאה אמיתית
            actual_result = match['FTR']
            
            # בדיקת תנאי הימור
            if prediction['home_win'] > confidence:
                total_bets += 1
                if actual_result == 'H':
                    correct += 1
            elif prediction['away_win'] > confidence:
                total_bets += 1
                if actual_result == 'A':
                    correct += 1
                    
        except Exception as e:
            st.error(f"שגיאה במשחק {match['HomeTeam']} vs {match['AwayTeam']}: {str(e)}")
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
# טעינת נתונים (מתוקן עם ולידציה)
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
            df = pd.read
