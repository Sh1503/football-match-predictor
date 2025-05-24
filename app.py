
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Football Match Predictor")

# Load example dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/english-premier-league/2023-2024.csv'
    df = pd.read_csv(url)
    df = df.dropna(subset=['FTHG', 'FTAG'])
    df['result'] = df.apply(lambda row: 'H' if row['FTHG'] > row['FTAG'] else ('A' if row['FTAG'] > row['FTHG'] else 'D'), axis=1)
    return df

df = load_data()
X = df[['FTHG', 'FTAG']]
X.columns = ['home_goals', 'away_goals']
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.subheader("Enter Match Details:")
home_goals = st.number_input("Home Team Goals", min_value=0, max_value=10, value=1)
away_goals = st.number_input("Away Team Goals", min_value=0, max_value=10, value=1)

if st.button("Predict Outcome"):
    pred = model.predict([[home_goals, away_goals]])[0]
    outcome = {'H': '🏠 Home Win', 'D': '🤝 Draw', 'A': '🚗 Away Win'}[pred]
    st.success(f"Predicted Outcome: {outcome}")

st.markdown(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
