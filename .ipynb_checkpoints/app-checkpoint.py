import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('gaji_prediksi_model.joblib')
le_kategori = joblib.load('label_encoder_kategori.joblib')
le_pendidikan = joblib.load('label_encoder_pendidikan.joblib')
imputer_umur = joblib.load('imputer_umur.joblib')
imputer_pend = joblib.load('imputer_pendidikan.joblib')

job_data = pd.read_csv('job_data_for_streamlit.csv')
st.set_page_config(page_title="Cariin")
st.title('Cariin')

kategori_options = le_kategori.classes_
pendidikan_options = le_pendidikan.classes_

kategori_user = st.selectbox('Pilih Kategori Pekerjaan', kategori_options)
umur_user = st.number_input('Masukkan Umur Anda', min_value=18, max_value=65, value=30)
pendidikan_user = st.selectbox('Pilih Pendidikan Terakhir', pendidikan_options)

kategori_encoded = le_kategori.transform([kategori_user])[0]
pendidikan_encoded = le_pendidikan.transform([pendidikan_user])[0]
umur_imputed = imputer_umur.transform(np.array([[umur_user]]))[0][0]

features = np.array([[kategori_encoded, umur_imputed, pendidikan_encoded]])
gaji_pred = model.predict(features)[0]

st.markdown(f"### Estimasi Gaji: Rp {gaji_pred:,.0f}")

relevant_jobs = job_data[job_data['Kategori Lowongan'] == kategori_user].head(10)

st.markdown('### Lowongan Pekerjaan Relevan:')
for _, row in relevant_jobs.iterrows():
    st.markdown(f"- [{row['Title']}]({row['Link']})")
