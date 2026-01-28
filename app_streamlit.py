import joblib 
import streamlit as st
import pandas as pd

model = joblib.load("model.joblib")

st.title("Regresi Nilai TKA")
st.markdown("Regresi nilai TKA berdasarkan jam belajar per hari, persen kehadiran, bimbel")

jam_belajar_per_hari = st.slider("Jam belajar per hari", 1, 24, 5)
persen_kehadiran = st.slider("Persen kehadiran", 0, 100, 60)
bimbel = st.pills("Bimbel", ["ya", "tidak"], default="ya")

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[jam_belajar_per_hari, persen_kehadiran, bimbel]],columns=["jam_belajar_per_hari", "persen_kehadiran", "bimbel"])
	prediksi = model.predict(data_baru)[0]
	prediksi = prediksi.clip(0,100)
	st.success(f"Model memprediksi nilai TKA : {prediksi:.0f}")
	st.balloons()

