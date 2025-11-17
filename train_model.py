import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = os.getcwd()
META_FILE = os.path.join("dataset", "sample_meta.csv")
AUDIO_ROOT = os.path.join("dataset", "audio", "audio")  # searches subfolders under this
MFCC_FEATURES = 20
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 16

# ----------------------------
# Utility: find audio file by filename within AUDIO_ROOT
# ----------------------------
def find_audio_path(filename):
    for root, dirs, files in os.walk(AUDIO_ROOT):
        if filename in files:
            return os.path.join(root, filename)
    return None

# ----------------------------
# Load metadata
# ----------------------------
if not os.path.exists(META_FILE):
    raise FileNotFoundError(f"Metadata file not found: {META_FILE}")

meta_df = pd.read_csv(META_FILE)

if 'engtype' not in meta_df.columns:
    raise ValueError("sample_meta.csv must contain 'engtype' column")

# ----------------------------
# CLEAN ENGINE TYPE COLUMN
# ----------------------------
# 1. Drop NaN engine types
meta_df = meta_df.dropna(subset=['engtype'])

# 2. Convert to string (avoids numeric/float issues)
meta_df['engtype'] = meta_df['engtype'].astype(str).str.strip()

# 3. Remove empty strings, "nan", "None"
meta_df = meta_df[meta_df['engtype'].isin(['', 'nan', 'None', '0']) == False]

# 4. Remove rows where engtype == 0 (already string now)
meta_df = meta_df[meta_df['engtype'] != "0"]

# ----------------------------
# Keep rows that have audio files
# ----------------------------
meta_df['audio_path'] = meta_df['filename'].apply(find_audio_path)
meta_df = meta_df[meta_df['audio_path'].notna()].reset_index(drop=True)
print(f"Samples available after filtering: {len(meta_df)}")

if len(meta_df) == 0:
    raise ValueError("No usable audio files found under dataset/audio/audio. Check files and paths.")

# ----------------------------
# Feature extraction (mean MFCC)
# ----------------------------
def extract_mfcc_mean(path, n_mfcc=MFCC_FEATURES):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

X = []
y = []

for idx, row in meta_df.iterrows():
    path = row['audio_path']
    try:
        feat = extract_mfcc_mean(path)
        X.append(feat)
        y.append(str(row['engtype']))
    except Exception as e:
        print("Error processing", path, e)

X = np.array(X)
y = np.array(y)

print("Feature matrix shape:", X.shape)

# ----------------------------
# Encode labels
# ----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ----------------------------
# Train/test split (stratify by integer labels)
# ----------------------------
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)

# convert to categorical for Keras
y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=len(le.classes_))
y_test = tf.keras.utils.to_categorical(y_test_int, num_classes=len(le.classes_))

# ----------------------------
# Build model (simple MLP)
# ----------------------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(MFCC_FEATURES,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
checkpoint = ModelCheckpoint('engine_classifier.h5', monitor='val_accuracy', save_best_only=True)
early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early]
)

# Save final artifacts
# model is already saved by checkpoint as engine_classifier.h5 (best model)
np.save('label_encoder.npy', le.classes_)
print('Saved engine_classifier.h5 and label_encoder.npy')
print('Training complete')
