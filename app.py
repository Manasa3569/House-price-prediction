import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------------------------------
# APP TITLE
# ---------------------------------
st.title("MANASA 🏠 Hyderabad House Price Prediction")

# ---------------------------------
# LOAD DATA
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Hyderabad_House_Data.csv")

data = load_data()

# ---------------------------------
# DATA PREPROCESSING
# ---------------------------------

# Washrooms
data['Washrooms'] = pd.to_numeric(data['Washrooms'], errors='coerce')
data['Washrooms'] = data['Washrooms'].fillna(data['Washrooms'].median())

# Area
data['Area'] = data['Area'].astype(str).str.extract('(\d+)')
data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
data['Area'] = data['Area'].fillna(data['Area'].median())

# Price
data['Price'] = (
    data['Price']
    .astype(str)
    .str.replace(r'[^\d.]', '', regex=True)
)
data['Price'] = data['Price'].astype(float)

# ---------------------------------
# FEATURES & TARGET
# ---------------------------------
X = data.drop('Price', axis=1)
y = data['Price']

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------
# MODEL TRAINING
# ---------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------
# USER INPUT
# ---------------------------------
st.subheader("🔢 Enter House Details")

area = st.number_input(
    "Area (in sq ft)", min_value=100, max_value=10000, value=1400
)
bedrooms = st.number_input(
    "Bedrooms", min_value=1, max_value=10, value=3
)
washrooms = st.number_input(
    "Washrooms", min_value=1, max_value=10, value=2
)

# ---------------------------------
# PREDICTION
# ---------------------------------
if st.button("Predict House Price"):
    new_house = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Washrooms': [washrooms]
    })

    new_house = pd.get_dummies(new_house)
    new_house = new_house.reindex(columns=X.columns, fill_value=0)

    new_scaled = scaler.transform(new_house)
    predicted_price = model.predict(new_scaled)

    st.success(
        f"🏷️ Predicted House Price: ₹ {predicted_price[0]:,.2f}"
    )

