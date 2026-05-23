import os
import random
import joblib
import numpy as np
import cv2
import librosa

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# 1. PATHS
# =========================================================
IMAGE_BASE = r"archive (3)"
FRAME_BASE = r"archive (4)"
AUDIO_BASE = r"archive (2)"

# =========================================================
# 2. SETTINGS
# =========================================================
IMG_SIZE = (128, 128)

# keep subsets for faster training in VS Code
MAX_IMAGE_PER_CLASS = 3000
MAX_FRAME_PER_CLASS = 1500
MAX_AUDIO_PER_CLASS = 2000

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

random.seed(42)
np.random.seed(42)

# =========================================================
# 3. HELPERS
# =========================================================
def is_image_file(fname):
    return fname.lower().endswith(IMAGE_EXTS)

def is_audio_file(fname):
    return fname.lower().endswith(AUDIO_EXTS)

def get_label_from_path(path):
    p = path.lower().replace("\\", "/")

    if "/fake/" in p or "training_fake" in p or "validation_fake" in p or "test/fake" in p:
        return 1
    if "/real/" in p or "training_real" in p or "validation_real" in p or "test/real" in p:
        return 0

    return None

def collect_files(base_path, checker, max_per_class=None):
    real_files = []
    fake_files = []

    for root, _, files in os.walk(base_path):
        for file in files:
            if checker(file):
                full_path = os.path.join(root, file)
                label = get_label_from_path(full_path)

                if label == 0:
                    real_files.append(full_path)
                elif label == 1:
                    fake_files.append(full_path)

    random.shuffle(real_files)
    random.shuffle(fake_files)

    if max_per_class is not None:
        real_files = real_files[:max_per_class]
        fake_files = fake_files[:max_per_class]

    return real_files, fake_files

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    return gray

def extract_hog_features(gray_img):
    return hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

def extract_mfcc_features(audio_path, max_len=300):
    try:
        audio, sr = librosa.load(audio_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
        mfcc = mfcc.flatten()

        if len(mfcc) < max_len:
            mfcc = np.pad(mfcc, (0, max_len - len(mfcc)))
        else:
            mfcc = mfcc[:max_len]

        return mfcc
    except Exception as e:
        print("Audio error:", audio_path, e)
        return None

# =========================================================
# 4. LOAD DATASETS
# =========================================================
def load_visual_dataset(base_path, max_per_class):
    real_files, fake_files = collect_files(base_path, is_image_file, max_per_class)

    X, y = [], []

    print(f"\nLoading visual data from: {base_path}")
    print("Real images found:", len(real_files))
    print("Fake images found:", len(fake_files))

    for path in real_files:
        img = load_image(path)
        if img is not None:
            X.append(extract_hog_features(img))
            y.append(0)

    for path in fake_files:
        img = load_image(path)
        if img is not None:
            X.append(extract_hog_features(img))
            y.append(1)

    return np.array(X), np.array(y)

def load_audio_dataset(base_path, max_per_class):
    real_files, fake_files = collect_files(base_path, is_audio_file, max_per_class)

    X, y = [], []

    print(f"\nLoading audio data from: {base_path}")
    print("Real audio found:", len(real_files))
    print("Fake audio found:", len(fake_files))

    for path in real_files:
        feat = extract_mfcc_features(path)
        if feat is not None:
            X.append(feat)
            y.append(0)

    for path in fake_files:
        feat = extract_mfcc_features(path)
        if feat is not None:
            X.append(feat)
            y.append(1)

    return np.array(X), np.array(y)

# =========================================================
# 5. TRAIN BEST MODEL
# =========================================================
def train_best_model(X, y, data_name):
    print(f"\n================ {data_name} ================")

    if len(X) == 0:
        raise ValueError(f"No data loaded for {data_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "SVM": SVC(C=5, kernel="rbf", probability=True),
        "RF": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        "XGB": XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    best_model = None
    best_name = None
    best_acc = 0.0

    for name, model in models.items():
        print(f"\nTraining {data_name} with {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        print(f"{name} Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    print(f"\nBest model for {data_name}: {best_name} ({best_acc:.4f})")
    print(classification_report(y_test, best_model.predict(X_test)))

    return best_model, scaler, best_name, best_acc

# =========================================================
# 6. MAIN
# =========================================================
print("Starting dataset loading...")

X_img, y_img = load_visual_dataset(IMAGE_BASE, MAX_IMAGE_PER_CLASS)
X_frm, y_frm = load_visual_dataset(FRAME_BASE, MAX_FRAME_PER_CLASS)
X_aud, y_aud = load_audio_dataset(AUDIO_BASE, MAX_AUDIO_PER_CLASS)

print("\nShapes:")
print("Image:", X_img.shape, y_img.shape)
print("Frame:", X_frm.shape, y_frm.shape)
print("Audio:", X_aud.shape, y_aud.shape)

img_model, img_scaler, img_best, img_acc = train_best_model(X_img, y_img, "IMAGE")
frm_model, frm_scaler, frm_best, frm_acc = train_best_model(X_frm, y_frm, "FRAME")
aud_model, aud_scaler, aud_best, aud_acc = train_best_model(X_aud, y_aud, "AUDIO")

# =========================================================
# 7. SAVE MODELS
# =========================================================
os.makedirs("saved_models", exist_ok=True)

joblib.dump(img_model, "saved_models/img_model.pkl")
joblib.dump(img_scaler, "saved_models/img_scaler.pkl")

joblib.dump(frm_model, "saved_models/frm_model.pkl")
joblib.dump(frm_scaler, "saved_models/frm_scaler.pkl")

joblib.dump(aud_model, "saved_models/aud_model.pkl")
joblib.dump(aud_scaler, "saved_models/aud_scaler.pkl")

print("\nTraining completed.")
print("Saved in folder: saved_models")
print(f"Best image model: {img_best} | acc={img_acc:.4f}")
print(f"Best frame model: {frm_best} | acc={frm_acc:.4f}")
print(f"Best audio model: {aud_best} | acc={aud_acc:.4f}")