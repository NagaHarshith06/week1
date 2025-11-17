import os
import io
import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Aircraft Engine Detection", layout="centered")
st.title("Aircraft Engine Detection")

MODEL_FILE = "engine_classifier.h5"
LE_FILE = "label_encoder.npy"

def load_artifacts():
    model = None
    classes = None
    if os.path.exists(MODEL_FILE):
        try:
            model = load_model(MODEL_FILE)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    if os.path.exists(LE_FILE):
        try:
            classes = np.load(LE_FILE, allow_pickle=True)
        except Exception as e:
            st.warning(f"Failed to load label encoder: {e}")
    return model, classes

model, classes = load_artifacts()

st.markdown("Upload a WAV file recorded similarly to the AeroSonic dataset (mono or stereo). The app uses a saved Keras model (`engine_classifier.h5`) and `label_encoder.npy`. If these files are not present, run `python train_model.py` locally to create them.")

uploaded = st.file_uploader("Upload WAV file", type=["wav"])

def extract_mfcc_mean_from_bytes(wav_bytes, n_mfcc=20):
    try:
        data, sr = librosa.load(io.BytesIO(wav_bytes), sr=None)
    except Exception:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(wav_bytes))
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

if uploaded is not None:
    wav_bytes = uploaded.read()
    st.audio(wav_bytes, format='audio/wav')
    try:
        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, duration=5.0)
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, len(y)/sr, len(y)), y)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Couldn't plot waveform: {e}")

    if model is None or classes is None:
        st.error('Model or label encoder not found. Run `python train_model.py` to create them.')
    else:
        feat = extract_mfcc_mean_from_bytes(wav_bytes, n_mfcc=20)
        inp = feat.reshape(1, -1)
        probs = model.predict(inp)[0]
        top_idx = np.argsort(probs)[::-1]
        # bar chart of probabilities
        fig2, ax2 = plt.subplots()
        ax2.bar(range(len(probs)), probs)
        ax2.set_xticks(range(len(probs)))
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.set_ylabel('Probability')
        ax2.set_title('Predicted probabilities')
        st.pyplot(fig2)

        pred_label = classes[top_idx[0]]
        st.success(f'Predicted engine type: {pred_label} (prob={probs[top_idx[0]]:.3f})')
        st.markdown('**Top 5 predictions**')
        for i in top_idx[:5]:
            st.write(f'{classes[i]} — {probs[i]:.4f}')

st.markdown('---')
st.markdown('If you do not have `engine_classifier.h5`, run `python train_model.py` to train and save the model.')
