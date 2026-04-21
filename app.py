import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL & DATA
# =========================
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

df = pd.read_csv('DataSpotify.csv', sep=';', low_memory=False)

# =========================
# BERSIHKAN KOLOM
# =========================
df.columns = df.columns.str.strip().str.lower()

# =========================
# DETEKSI KOLOM
# =========================
name_cols = ['name', 'track_name', 'song_name', 'title']
artist_cols = ['artist', 'artists', 'artist_name']

name_col = next((c for c in name_cols if c in df.columns), None)
artist_col = next((c for c in artist_cols if c in df.columns), None)

if name_col is None:
    st.error("❌ Kolom nama lagu tidak ditemukan!")
    st.stop()

if artist_col is None:
    st.error("❌ Kolom artis tidak ditemukan!")
    st.stop()

# =========================
# CEK KOLOM WAJIB
# =========================
required_cols = [name_col, artist_col, 'energy', 'acousticness']

for col in required_cols:
    if col not in df.columns:
        st.error(f"❌ Kolom '{col}' tidak ditemukan!")
        st.stop()

# =========================
# KONVERSI NUMERIK
# =========================
df['energy'] = df['energy'].astype(str).str.replace(',', '.')
df['acousticness'] = df['acousticness'].astype(str).str.replace(',', '.')

df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
df['acousticness'] = pd.to_numeric(df['acousticness'], errors='coerce')

# =========================
# FILTER DATA
# =========================
df = df[[name_col, artist_col, 'energy', 'acousticness']].dropna()

# =========================
# UI
# =========================
st.title("🎧 Clustering Lagu Spotify")
st.write("Cluster 0 → Happy")
st.write("Cluster 1 → Galau")
st.write("Cluster 2 → Santai")

# =========================
# PILIH ARTIS
# =========================
st.subheader("🎤 Pilih Artis")
artist_name = st.selectbox("Pilih Artis", sorted(df[artist_col].unique()))

# =========================
# FILTER LAGU
# =========================
filtered_df = df[df[artist_col] == artist_name]

# =========================
# PILIH LAGU
# =========================
st.subheader("🔎 Pilih Lagu")
song_name = st.selectbox("Pilih Lagu", filtered_df[name_col].unique())

# =========================
# AMBIL DATA
# =========================
song_data = filtered_df[filtered_df[name_col] == song_name].iloc[0]

energy = float(song_data['energy'])
acousticness = float(song_data['acousticness'])

# =========================
# PREDIKSI
# =========================
input_data = np.array([[energy, acousticness]])
input_scaled = scaler.transform(input_data)
cluster = model.predict(input_scaled)

# =========================
# OUTPUT
# =========================
st.subheader("📊 Hasil")

st.write("Artis:", artist_name)
st.write("Judul Lagu:", song_name)
st.write("Energy:", round(energy, 3))
st.write("Acousticness:", round(acousticness, 3))
st.write("Cluster:", int(cluster[0]))

# =========================
# INTERPRETASI EMOSI
# =========================
st.subheader("🧠 Kategori Emosi")

cluster_map = {
    0: "🎉 Happy / Energetic",
    1: "😢 Galau / Sedih",
    2: "😌 Santai / Chill"
}

label = cluster_map.get(int(cluster[0]), "🎧 Tidak diketahui")

st.write("Kategori Lagu:", label)
