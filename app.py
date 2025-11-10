# ---------------------------------------------------------
# 🎙️ Accent-Aware Cuisine Recommendation System
# ---------------------------------------------------------

import streamlit as st
import numpy as np
import librosa
import torch
import joblib
from transformers import AutoFeatureExtractor, HubertModel

# ------------------------------
# Load Model, States, and HuBERT
# ------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("hubert_model.pkl")          # your trained Random Forest model
    states = joblib.load("hubert_states.pkl")        # same folder/state order used in training

    extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.eval()

    return model, states, extractor, hubert

model, states, extractor, hubert = load_assets()

# ------------------------------
# Accent → Cuisine Mapping
# ------------------------------
accent_to_cuisine = {
    "Andhra Pradesh": "Pulihora, Pesarattu",
    "Gujarat": "Dhokla, Thepla",
    "Jharkhand": "Litti Chokha",
    "Karnataka": "Bisi Bele Bath, Neer Dosa",
    "Kerala": "Appam, Avial, Puttu",
    "Tamil Nadu": "Pongal, Idli, Dosa"
}

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Accent-Aware Cuisine Recommendation", page_icon="🎧")
st.title("🎙️ Accent-Aware Cuisine Recommendation System")
st.markdown(
    "Upload an English speech sample, and this app will detect your regional accent "
    "and recommend popular cuisines from that region! 🍛"
)

uploaded_file = st.file_uploader("📂 Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

# ------------------------------
# Process Audio and Predict Accent
# ------------------------------
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🎧 Extracting HuBERT features and predicting accent..."):
        try:
            # Load and normalize audio
            y, sr = librosa.load(uploaded_file, sr=16000)
            y = librosa.util.normalize(y)

            # Handle long/silent audio
            if len(y) > 16000 * 5:  # keep only 5 seconds
                y = y[:16000 * 5]
            if np.mean(np.abs(y)) < 0.01:
                st.warning("⚠️ Audio too quiet or silent. Please upload a clearer file.")
                st.stop()

            # HuBERT feature extraction
            inputs = extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = hubert(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

            # Predict accent
            pred = model.predict([emb])[0]
            accent = states[pred].replace("_", " ").title()
            cuisine = accent_to_cuisine.get(accent, "Cuisine recommendation not available 🍽️")

            # Display results
            st.success(f"🗣️ **Detected Accent:** {accent}")
            st.info(f"🍲 **Recommended Cuisine:** {cuisine}")

            # Optional confidence
            if hasattr(model, "predict_proba"):
                conf = np.max(model.predict_proba([emb])) * 100
                st.caption(f"Confidence: {conf:.2f}%")

        

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
