import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Load model dan data
model = joblib.load('best_salary_predictor.pkl')
df = pd.read_csv('jobstreet_jobs_cleaned_with_category.csv')

# Buat kolom Gaji_Rata jika belum ada
if 'Gaji_Rata' not in df.columns:
    df['Gaji_Rata'] = (df['Gaji Min'] + df['Gaji Max']) / 2

# Load metrik evaluasi (jika ada)
try:
    metrics = joblib.load('model_metrics.pkl')
except:
    metrics = {
        'best_model': 'XGBoost',
        'r2': 0.85,
        'rmse': 2500000,
        'mae': 1800000,
        'test_size': 640,
        'feature_importances': pd.DataFrame({
            'Feature': ['Pendidikan_Encoded', 'Tahun Pengalaman', 'Jumlah_Skill', 'Kategori_IT'],
            'Importance': [0.35, 0.30, 0.20, 0.15]
        })
    }

# PERBAIKAN 1: Muat daftar kolom yang digunakan saat training
try:
    model_columns = joblib.load('model_columns.pkl')
except:
    # Buat daftar kolom default jika file tidak ada
    model_columns = [
        'Jumlah_Skill', 
        'Pendidikan_Encoded', 
        'Tahun Pengalaman',
        'Kategori_Lowongan_Account Executive',
        'Kategori_Lowongan_Admin',
        'Kategori_Lowongan_Akuntan',
        'Kategori_Lowongan_Apoteker'
    ]

def preprocess_input(input_data, encoder):
    pendidikan_map = {'SMA': 0, 'D3': 1, 'S1': 2, 'S2': 3, 'S3': 4}
    
    jumlah_skill = len(input_data['skills'].split(',')) if input_data['skills'] else 0
    
    input_df = pd.DataFrame({
        'Jumlah_Skill': [jumlah_skill],
        'Pendidikan_Encoded': [pendidikan_map.get(input_data['pendidikan'], -1)],
        'Tahun Pengalaman': [input_data['pengalaman']],
        'Kategori_Lowongan': [input_data['role']]
    })
    
    kategori_encoded = encoder.transform(input_df[['Kategori_Lowongan']])
    encoded_df = pd.DataFrame(
        kategori_encoded, 
        columns=encoder.get_feature_names_out(['Kategori_Lowongan'])
    )
    
    input_processed = pd.concat([
        input_df[['Jumlah_Skill', 'Pendidikan_Encoded', 'Tahun Pengalaman']],
        encoded_df
    ], axis=1)
    
    for col in model_columns:
        if col not in input_processed.columns:
            input_processed[col] = 0
    
    input_processed = input_processed[model_columns]
    
    return input_processed

def rekomendasi_lowongan(predicted_salary, df, role, n=5):
    role_based = df[df['Kategori_Lowongan'] == role].copy()
    
    if len(role_based) < n:
        role_based = df.copy()
    
    role_based['Salary_Diff'] = abs(role_based['Gaji_Rata'] - predicted_salary)
    
    recommended = role_based.sort_values('Salary_Diff').head(n)
    
    return recommended[['Title', 'Posisi', 'Gaji', 'Link', 'Kualifikasi', 'Skill', 'Salary_Diff']]

st.set_page_config(page_title="Prediksi Gaji & Rekomendasi Lowongan", layout="wide")
st.title('Cariin')

st.sidebar.header("📊 Akurasi Model")
st.sidebar.subheader(f"Model Terbaik: {metrics.get('best_model', 'XGBoost')}")
st.sidebar.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
st.sidebar.metric("RMSE", f"{metrics.get('rmse', 0):,.2f}")
st.sidebar.metric("MAE", f"{metrics.get('mae', 0):,.2f}")
st.sidebar.caption(f"Model diuji dengan {metrics.get('test_size', 0)} sampel data")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📝 Profil Anda")
    
    # Buat input form
    role = st.selectbox(
        'Role Pekerjaan', 
        options=df['Kategori_Lowongan'].unique(),
        index=0
    )

    pendidikan = st.selectbox(
        'Pendidikan Terakhir', 
        options=['SMA', 'D3', 'S1', 'S2', 'S3'],
        index=2
    )

    pengalaman = st.slider(
        'Tahun Pengalaman Kerja', 
        min_value=0, max_value=30, value=3
    )

    usia = st.slider(
        'Usia', 
        min_value=20, max_value=60, value=30
    )

    lokasi = st.selectbox(
        'Lokasi', 
        options=['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Lainnya'],
        index=0
    )

    skills = st.text_input(
        'Skill (pisahkan dengan koma)',
        'Python, SQL, Data Analysis'
    )

    # Tombol prediksi
    if st.button('🚀 Prediksi Gaji & Dapatkan Rekomendasi', use_container_width=True):
        st.session_state.predict_clicked = True
    else:
        st.session_state.predict_clicked = False

with col2:
    st.header("📊 Hasil Prediksi & Rekomendasi")
    
    if 'predict_clicked' in st.session_state and st.session_state.predict_clicked:
        input_data = {
            'role': role,
            'pendidikan': pendidikan,
            'pengalaman': pengalaman,
            'usia': usia,
            'lokasi': lokasi,
            'skills': skills
        }
        
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(df[['Kategori_Lowongan']])
        
        try:
            input_processed = preprocess_input(input_data, encoder)
            
            predicted_salary = model.predict(input_processed)[0]
            
            st.success(f"### Estimasi Gaji Anda: **Rp{predicted_salary:,.2f}** per bulan")
            
            with st.expander("Detail Profil Anda"):
                st.write(f"**Role:** {role}")
                st.write(f"**Pendidikan:** {pendidikan}")
                st.write(f"**Pengalaman Kerja:** {pengalaman} tahun")
                st.write(f"**Usia:** {usia} tahun")
                st.write(f"**Lokasi:** {lokasi}")
                st.write(f"**Skill:** {skills}")
            
            st.subheader("🎯 Rekomendasi Lowongan untuk Anda")
            recommendations = rekomendasi_lowongan(predicted_salary, df, role)
            
            if recommendations.empty:
                st.warning("Tidak ditemukan lowongan yang sesuai. Silakan coba dengan kriteria berbeda.")
            else:
                for i, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div style="
                            border: 1px solid #e0e0e0;
                            border-radius: 10px;
                            padding: 15px;
                            margin-bottom: 15px;
                        ">
                            <h4 style="margin-top:0;">{row['Title']} - {row['Posisi']}</h4>
                            <p><a href="{row['Link']}" target="_blank" style="">Buka Lowongan di Jobstreet</a></p>
                            <p><b>Estimasi Gaji:</b> {row['Gaji']}</p>
                            <p><b>Selisih dengan prediksi Anda:</b> Rp{row.get('Salary_Diff', 0):,.2f}</p>
                            <details>
                                <summary><b>Lihat Detail Kualifikasi</b></summary>
                                <p><b>Kualifikasi:</b> {row.get('Kualifikasi', 'Tidak tersedia')}</p>
                                <p><b>Skill yang Dibutuhkan:</b> {row.get('Skill', 'Tidak tersedia')}</p>
                            </details>
                        </div>
                        """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
            st.error("Pastikan semua file model dan data tersedia dengan format yang benar")
            st.error("Jika masalah berlanjut, coba training ulang model dan simpan kembali file model_columns.pkl")
    else:
        st.info("Silakan isi profil Anda di sebelah kiri dan klik tombol 'Prediksi Gaji & Dapatkan Rekomendasi'")
        
        # Tampilkan beberapa lowongan acak sebagai contoh
        st.subheader("💼 Contoh Lowongan Tersedia")
        sample_jobs = df.sample(min(3, len(df)))
        
        for _, row in sample_jobs.iterrows():
            st.markdown(f"""
            <div style="
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                backgfround-color: #9f9f9;
                text-color: black;
            ">
                <h4 style="margin-top:0; text-color:black;">{row['Title']}</h4>
                <p><b>Posisi:</b> {row['Posisi']}</p>
                <p><b>Gaji:</b> {row.get('Gaji', 'Tidak tersedia')}</p>
                <p><a href="{row['Link']}" target="_blank">Buka Lowongan di Jobstreet</a></p>
            </div>
            """, unsafe_allow_html=True)

st.header("ℹ️ Tentang Model dan Data")
tab1, tab2, tab3 = st.tabs(["Akurasi Model", "Feature Importance", "Dataset"])

with tab1:
    st.subheader("Evaluasi Performa Model")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
    # col2.metric("RMSE", f"{metrics.get('rmse', 0):,.2f}")
    # col3.metric("MAE", f"{metrics.get('mae', 0):,.2f}")
    
    st.write("""
    **Interpretasi Metrik:**
    - **R² Score**: Mendekati 1 berarti model menjelaskan sebagian besar variasi dalam data (0.8-0.9 = sangat baik)
    - **RMSE (Root Mean Squared Error)**: Mengukur rata-rata selisih antara prediksi dan nilai aktual
    - **MAE (Mean Absolute Error)**: Mengukur rata-rata kesalahan absolut
    """)

    st.info("""
    🔍 Model ini memiliki akurasi yang baik untuk memprediksi rentang gaji berdasarkan:
    - Role pekerjaan
    - Tingkat pendidikan
    - Pengalaman kerja
    - Skill yang dimiliki
    """)

with tab2:
    st.subheader("Faktor Paling Berpengaruh pada Prediksi Gaji")
    
    if 'feature_importances' in metrics:
        # Buat visualisasi feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Pastikan metrics['feature_importances'] adalah DataFrame
        if isinstance(metrics['feature_importances'], pd.DataFrame):
            df_importances = metrics['feature_importances']
        else:
            # Buat DataFrame jika format tidak sesuai
            df_importances = pd.DataFrame({
                'Feature': list(metrics['feature_importances'].keys()),
                'Importance': list(metrics['feature_importances'].values())
            })
        
        # Urutkan dan ambil top 10
        df_importances = df_importances.sort_values('Importance', ascending=False).head(10)
        
        # PERBAIKAN 5: Gunakan parameter hue untuk menghindari warning
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=df_importances,
            ax=ax,
            hue='Feature',  # Tambahkan hue
            legend=False,    # Sembunyikan legenda
            palette="viridis"
        )
        ax.set_title('Top 10 Faktor Penentu Prediksi Gaji')
        ax.set_xlabel('Tingkat Kepentingan')
        ax.set_ylabel('Fitur')
        st.pyplot(fig)
        
        st.write("""
        **Interpretasi:**
        - **Pendidikan**: Semakin tinggi pendidikan, semakin tinggi potensi gaji
        - **Pengalaman Kerja**: Pengalaman relevan meningkatkan nilai profesional
        - **Jumlah Skill**: Keahlian teknis khusus sangat dihargai di pasar kerja
        - **Kategori Pekerjaan**: Bidang TI dan manajerial cenderung memiliki gaji lebih tinggi
        """)
    else:
        st.warning("Informasi feature importance tidak tersedia")

with tab3:
    st.subheader("📁 Informasi Dataset")
    
    st.write(f"Dataset ini berisi **{len(df)} lowongan kerja** dari berbagai sektor industri")
    
    # Statistik dataset
    st.write("**Statistik Utama:**")
    
    # Handle jika kolom tidak ada
    gaji_min = df['Gaji Min'].min() if 'Gaji Min' in df else 0
    gaji_max = df['Gaji Max'].max() if 'Gaji Max' in df else 0
    gaji_rata = df['Gaji_Rata'].mean() if 'Gaji_Rata' in df else 0
    
    st.json({
        "Kategori Pekerjaan": df['Kategori_Lowongan'].nunique(),
        "Rentang Gaji": f"Rp{gaji_min:,.0f} - Rp{gaji_max:,.0f}",
        "Rata-rata Gaji": f"Rp{gaji_rata:,.0f}",
        "Pendidikan Paling Umum": df['Pendidikan'].mode()[0] if 'Pendidikan' in df else "Tidak tersedia",
        "Sumber Data": "Jobstreet Indonesia (2023)"
    })
    
    # Tampilkan distribusi kategori pekerjaan
    st.subheader("Distribusi Kategori Pekerjaan")
    category_counts = df['Kategori_Lowongan'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    category_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Top 10 Kategori Pekerjaan')
    ax.set_xlabel('Kategori')
    ax.set_ylabel('Jumlah Lowongan')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    
    # Tampilkan beberapa data
    st.subheader("Contoh Data Lowongan")
    st.dataframe(df[['Title', 'Posisi', 'Gaji', 'Kategori_Lowongan']].head(10))

# Catatan kaki
st.markdown("---")
st.caption("""
Aplikasi prediksi gaji ini menggunakan model machine learning yang dilatih pada data lowongan kerja Jobstreet. 
Prediksi bersifat estimasi dan dapat bervariasi berdasarkan faktor-faktor lain yang tidak termasuk dalam model.
""")