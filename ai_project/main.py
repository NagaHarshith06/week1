import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PATHS = [
    r"C:\Users\Lenovo\Desktop\ai_project\dataset\audio\audio\0",
    r"C:\Users\Lenovo\Desktop\ai_project\dataset\audio\audio\1"
]

META_FILE = r"C:\Users\Lenovo\Desktop\ai_project\dataset\sample_meta.csv"
MFCC_FEATURES = 20  # reduce for speed

# ----------------------------
# LOAD METADATA
# ----------------------------
meta_df = pd.read_csv(META_FILE)
if 'engtype' not in meta_df.columns:
    raise ValueError("sample_meta.csv must contain 'engtype' column")

meta_df = meta_df[meta_df['engtype'] != 0]

# ----------------------------
# LOAD AUDIO AND EXTRACT FEATURES
# ----------------------------
X = []
y = []

valid_files = set(meta_df['filename'].values)

for folder_path in DATASET_PATHS:
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    for file in os.listdir(folder_path):
        if file.endswith(".wav") and file in valid_files:
            file_path = os.path.join(folder_path, file)
            try:
                audio, sr = librosa.load(file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_FEATURES)
                mfcc_mean = np.mean(mfcc, axis=1)
                X.append(mfcc_mean)

                engtype_row = meta_df[meta_df['filename'] == file]
                y.append(str(engtype_row.iloc[0]['engtype']))
            except Exception as e:
                print(f"Error processing {file}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Total files used: {len(X)}, Total labels used: {len(y)}")

if len(X) == 0:
    raise ValueError("No audio files found. Check dataset path and filenames.")

# ----------------------------
# CLASS DISTRIBUTION
# ----------------------------
plt.figure(figsize=(8,5))
unique, counts = np.unique(y, return_counts=True)
plt.bar(unique, counts, color='skyblue')
plt.xlabel("Engine Type")
plt.ylabel("Number of Samples")
plt.title("Class Distribution")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ----------------------------
# LABEL ENCODING
# ----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# ----------------------------
# TRAIN-TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# ----------------------------
# NEURAL NETWORK
# ----------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(MFCC_FEATURES,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=30,  # increase/decrease for speed/accuracy
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ----------------------------
# ACCURACY PLOT
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# PREDICTIONS & CONFUSION MATRIX
# ----------------------------
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_, yticklabels=le.classes_,
            cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Engine Classification")
plt.tight_layout()
plt.show()
