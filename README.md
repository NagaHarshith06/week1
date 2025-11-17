# AeroSonic — Neural model + Streamlit app bundle

This bundle contains:
- `train_model.py` — trains a neural network on the AeroSonic-style dataset, saves `engine_classifier.h5` and `label_encoder.npy`
- `app.py` — Streamlit app that loads the saved model and classifies uploaded WAV files
- `requirements.txt` — Python dependencies

## Quick start

1. Create a virtual environment and install requirements:
```bash
pip install -r requirements.txt
```

2. Place your dataset so that metadata is at `dataset/sample_meta.csv` and audio files are under `dataset/audio/audio/*/<filename>.wav`

3. Train (this will create `engine_classifier.h5` and `label_encoder.npy`):
```bash
python train_model.py
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

Notes:
- The trainer uses mean MFCC features and a small MLP for speed. For higher accuracy, consider a CNN on MFCC sequences or data augmentation.
- Training can take time depending on the dataset size and your machine.
