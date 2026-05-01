# Trains an MLP on MediaPipe hand landmarks to classify ASL fingerspelling
# Reads landmarks.csv outputs model.pkl
# has  mirror augmentation, pose jitter, and extra weight on R, U, V

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from landmark_features import (
    FEATURE_DIM,
    expand_feature_matrix,
    mirror_base_x,
    random_pose_augment,
)

LANDMARKS_CSV = "./landmarks.csv"
MODEL_OUT = "./model.pkl"
ENCODER_OUT = "./label_encoder.pkl"
TEST_SIZE = 0.15
RANDOM_STATE = 42

# R  U  V are easy to confuse so gave more synthetic views and loss weight
RUV_LABELS = frozenset({"R", "U", "V"})
RUV_EXTRA_AUG_PER_SAMPLE = 6
RUV_SAMPLE_WEIGHT = 2.25
POSE_NOISE_STD = 0.018
POSE_MAX_ANGLE_RAD = 0.42

# ================ Load data ================
print("Loading landmarks.csv...")
df = pd.read_csv(LANDMARKS_CSV)
print(f"  {len(df)} rows, {df['label'].nunique()} classes: {sorted(df['label'].unique())}")

X63 = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

# ================ Augment: mirror ================
X_mirrored = mirror_base_x(X63)
X63 = np.vstack([X63, X_mirrored])
y = np.concatenate([y, y], axis=0)
print(f"  After mirroring: {len(X63)} samples")

# ================ Extra pose jitter for R, U, V ================
rng = np.random.default_rng(RANDOM_STATE)
ruv_mask = np.isin(y, list(RUV_LABELS))
extra_X = []
extra_y = []
for i in np.where(ruv_mask)[0]:
    for _ in range(RUV_EXTRA_AUG_PER_SAMPLE):
        aug = random_pose_augment(
            X63[i],
            rng,
            noise_std=POSE_NOISE_STD,
            max_angle_rad=POSE_MAX_ANGLE_RAD,
        )
        extra_X.append(aug)
        extra_y.append(y[i])
if extra_X:
    X63 = np.vstack([X63, np.stack(extra_X, axis=0)])
    y = np.concatenate([y, np.array(extra_y)], axis=0)
    print(f"  After R/U/V pose augmentation: {len(X63)} samples (+{len(extra_y)} synthetic)")

# ================ Global light jitter (all classes) for robustness ================
n_jitter = min(12_000, len(X63) // 4)
if n_jitter > 0:
    idx = rng.choice(len(X63), size=n_jitter, replace=False)
    jitter_blocks = [
        random_pose_augment(X63[i], rng, noise_std=POSE_NOISE_STD * 0.7, max_angle_rad=0.22)
        for i in idx
    ]
    X63 = np.vstack([X63, np.stack(jitter_blocks, axis=0)])
    y = np.concatenate([y, y[idx]], axis=0)
    print(f"  After global light jitter: {len(X63)} samples")

# ================ Expand 63 → 68 ================
X = expand_feature_matrix(X63)
print(f"  Feature shape: {X.shape[1]} (= {FEATURE_DIM})")

# ================ Encode labels ================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"  Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ================ Train/val split ================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"\nTrain: {len(X_train)} samples | Val: {len(X_val)} samples")

ruv_class_idx = np.array(
    [le.transform([c])[0] for c in RUV_LABELS if c in le.classes_],
    dtype=int,
)
sample_weight = np.ones(len(y_train), dtype=np.float64)
for j in ruv_class_idx:
    sample_weight[y_train == j] = RUV_SAMPLE_WEIGHT

# ================ Train MLP ================
print("\nTraining MLP...")
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    alpha=1e-4,
    max_iter=900,
    random_state=RANDOM_STATE,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    learning_rate_init=0.001,
)
model.fit(X_train, y_train, sample_weight=sample_weight)

# ================ Evaluate ================
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\nValidation accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nPer-class report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# Highlight R, U, V confusion
ruv_names = sorted(RUV_LABELS & set(le.classes_))
if ruv_names:
    print("\nR / U / V subset report:")
    mask = np.isin(le.inverse_transform(y_val), ruv_names)
    if mask.any():
        print(
            classification_report(
                y_val[mask],
                y_pred[mask],
                labels=[le.transform([n])[0] for n in ruv_names],
                target_names=ruv_names,
                zero_division=0,
            )
        )

# ================ Confusion matrix ================
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
)
plt.title("Confusion Matrix — Validation Set")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved to confusion_matrix.png")

# ================ Save ================
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)
with open(ENCODER_OUT, "wb") as f:
    pickle.dump(le, f)
print(f"Model saved to {MODEL_OUT}")
print(f"Label encoder saved to {ENCODER_OUT}")
