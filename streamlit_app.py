import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, dan fitur
model = joblib.load("dropout_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Prediksi Dropout Siswa - Jaya Jaya Institut", layout="centered")
st.title("🎓 Prediksi Dropout Siswa")

# Input user
st.subheader("Masukkan Data Siswa")

# Buat input untuk semua fitur
input_data = {}
for col in feature_columns:
    if "grade" in col or "GPA" in col or "rate" in col or "Charges" in col or "Age" in col:
        input_data[col] = st.number_input(col, value=0.0)
    else:
        input_data[col] = st.number_input(col, value=0)

# Konversi ke DataFrame
input_df = pd.DataFrame([input_data])

# Pastikan urutan kolom sesuai
input_df = input_df[feature_columns]

# Transformasi scaling
input_scaled = scaler.transform(input_df)

# Prediksi
if st.button("Prediksi Dropout"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    status = "Dropout" if prediction == 1 else "Lanjut"
    st.markdown(f"### 🚨 Hasil Prediksi: **{status}**")
    st.markdown(f"**Probabilitas:** `{probability:.2%}`")
