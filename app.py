
import streamlit as st
import joblib
import re
import pandas as pd # Import pandas for loading the dataset
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize # Using simple split instead
import nltk

# Ensure NLTK resources are available in the Streamlit environment
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Define the preprocessing function (same as used during training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split() # Using simple split for consistency
    stop_words_id = set(stopwords.words('indonesian'))
    filtered_tokens = [word for word in tokens if word not in stop_words_id]
    return ' '.join(filtered_tokens)

# Load the trained model and vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
svm_model = joblib.load('svm_model.pkl')

st.title('Aplikasi Deteksi Emosi')
st.write('Masukkan teks untuk mendeteksi emosi yang terkandung di dalamnya.')

# --- Emotion Distribution Visualization ---
st.header('Distribusi Emosi dalam Dataset')

# Load the original dataset to get emotion distribution for visualization
try:
    original_df = pd.read_excel('/content/ISEAR ID 100.xlsx')
    emotion_counts = original_df['Field'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Count']
    st.bar_chart(emotion_counts.set_index('Emotion'))
    st.write('Grafik di atas menunjukkan jumlah entri untuk setiap emosi dalam dataset pelatihan.')
except FileNotFoundError:
    st.error("Dataset 'ISEAR ID 100.xlsx' tidak ditemukan. Tidak dapat menampilkan distribusi emosi.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat atau menampilkan distribusi emosi: {e}")

# --- Prediction Section ---
st.header('Deteksi Emosi Teks Baru')

user_input = st.text_area('Masukkan teks Anda di sini:', '')

if st.button('Deteksi Emosi'):
    if user_input:
        # Preprocess the input text
        cleaned_text = preprocess_text(user_input)
        
        # Vectorize the preprocessed text
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = svm_model.predict(vectorized_text)
        
        st.success(f'Emosi yang terdeteksi: **{prediction[0]}**')
    else:
        st.warning('Mohon masukkan teks terlebih dahulu.')
