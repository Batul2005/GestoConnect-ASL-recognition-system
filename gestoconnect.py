#!/usr/bin/env python3
"""
GestoConnect - end-to-end script:
- prepare: extract mediapipe landmarks for dataset images (optional)
- train_cnn: train CNN on images (saves cnn_model.h5 and label_encoder.pkl)
- train_rf: train RandomForest on landmarks CSV (saves rf_model.pkl and label_encoder.pkl)
- evaluate: evaluate saved models
- demo: run real-time webcam demo (ensemble: avg prob of CNN & RF)
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import pyttsx3
from tqdm import tqdm

# --------------------------
# Config - change as needed
# --------------------------
DATA_DIR = "dataset/images"    # images organized by class folders
LANDMARK_CSV = "dataset/landmarks.csv"
CNN_MODEL_PATH = "cnn_model.h5"
RF_MODEL_PATH = "rf_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
RANDOM_STATE = 42
# --------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def list_classes(data_dir):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()
    return classes

# --------------------------
# Utility: load image dataset
# --------------------------
def load_image_dataset(data_dir, img_size=IMG_SIZE):
    X = []
    y = []
    classes = list_classes(data_dir)
    for cls in classes:
        folder = os.path.join(data_dir, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg','.png','.jpeg')):
                path = os.path.join(folder, fname)
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(cls)
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)
    print(f"[DATA] Loaded images: X.shape={X.shape}, labels={len(np.unique(y))}")
    return X, y, classes

# --------------------------
# Utility: extract mediapipe landmarks from images and save CSV
# --------------------------
def extract_landmarks_from_images(data_dir, csv_out=LANDMARK_CSV, img_size=IMG_SIZE):
    print("[LANDMARKS] Extracting landmarks (this may take some time)...")
    rows = []
    classes = list_classes(data_dir)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for cls in classes:
            folder = os.path.join(data_dir, cls)
            for fname in tqdm(os.listdir(folder), desc=f"Class {cls}"):
                if not fname.lower().endswith(('.jpg','.png','.jpeg')):
                    continue
                path = os.path.join(folder, fname)
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                res = hands.process(img_rgb)
                if not res.multi_hand_landmarks:
                    lm_flat = [0.0] * 63
                else:
                    hand = res.multi_hand_landmarks[0]
                    lm = []
                    for p in hand.landmark:
                        lm.extend([p.x, p.y, p.z])
                    lm_flat = lm  # length 63
                # store label + path + landmarks
                rows.append([cls, path] + lm_flat)

    # build dataframe
    if rows:
        cols = ["label", "path"] + [f"l{i}" for i in range(len(rows[0]) - 2)]
    else:
        cols = ["label", "path"]

    df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"[LANDMARKS] saved to {csv_out}. shape: {df.shape}")
    return df


# --------------------------
# Build CNN model (simple but effective)
# --------------------------
def build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), n_classes=26):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64,(3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128,(3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# --------------------------
# Train CNN
# --------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_cnn(data_dir, out_model=CNN_MODEL_PATH, label_encoder_path=LABEL_ENCODER_PATH):
    # data generators (stream from disk)
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation'
    )

    # build CNN
    model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), n_classes=train_gen.num_classes)

    # save class labels
    import joblib
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(list(train_gen.class_indices.keys()))
    joblib.dump(le, label_encoder_path)
    print(f"[LABELS] Saved label encoder to {label_encoder_path}")

    # training
    cb = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[cb]
    )

    model.save(out_model)
    print(f"[CNN] Saved model to {out_model}")


# --------------------------
# Train Random Forest on landmarks CSV
# --------------------------
def train_rf(landmark_csv=LANDMARK_CSV, out_model=RF_MODEL_PATH, label_encoder_path=LABEL_ENCODER_PATH):
    if not os.path.exists(landmark_csv):
        raise FileNotFoundError(f"{landmark_csv} not found. Run 'prepare' first to extract landmarks.")
    df = pd.read_csv(landmark_csv)
    df = df.dropna()

    # 🔹 Drop non-numeric columns (label + path)
    X = df.drop(columns=['label', 'path']).values
    y = df['label'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, label_encoder_path)
    print(f"[LABELS] saved label encoder to {label_encoder_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, out_model)
    print(f"[RF] Saved RF to {out_model}")

    # evaluate
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    acc = (y_pred == y_test).mean()
    print(f"[RF] Test accuracy: {acc:.4f}")


# --------------------------
# Helper: predict RF on mediapipe landmarks array (len=63)
# --------------------------
def rf_predict_proba(rf_model, landmark_array):
    # input should be 1D list/array of length 63. if zeros or missing, still ok.
    arr = np.array(landmark_array).reshape(1, -1)
    if hasattr(rf_model, "predict_proba"):
        probs = rf_model.predict_proba(arr)[0]
        return probs
    else:
        # fallback: one-hot like
        pred = rf_model.predict(arr)[0]
        probs = np.zeros(len(rf_model.classes_))
        probs[pred] = 1.0
        return probs

# --------------------------
# Evaluate ensemble on held-out test sets
# --------------------------
def evaluate_models(data_dir=DATA_DIR, landmark_csv=LANDMARK_CSV, cnn_model_path=CNN_MODEL_PATH, rf_model_path=RF_MODEL_PATH, label_encoder_path=LABEL_ENCODER_PATH):
    # Load label encoder
    le = joblib.load(label_encoder_path)
    classes = le.classes_
    # Load CNN if exists
    cnn = None
    if os.path.exists(cnn_model_path):
        cnn = tf.keras.models.load_model(cnn_model_path)
        print("[EVAL] Loaded CNN")
    rf = None
    if os.path.exists(rf_model_path):
        rf = joblib.load(rf_model_path)
        print("[EVAL] Loaded RF")

    # Build test set from images
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(data_dir, landmark_csv, cnn_model_path, rf_model_path, label_encoder_path):
    # CNN data generator for validation
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )

    # Load CNN
    cnn = models.load_model(cnn_model_path)
    print("[EVAL] Loaded CNN")

    # CNN predictions
    y_pred_cnn = cnn.predict(val_gen)
    y_pred_cnn = np.argmax(y_pred_cnn, axis=1)
    y_true = val_gen.classes
    print(f"[EVAL] CNN predictions done on {len(y_true)} images")

    # --- RF predictions (aligned to validation set) ---
    rf = joblib.load(rf_model_path)
    le = joblib.load(label_encoder_path)
    df = pd.read_csv(landmark_csv)

    # Extract only landmark rows corresponding to validation images
    val_filenames = [val_gen.filepaths[i].split("\\")[-2:] for i in range(len(val_gen.filepaths))]
    # Format: [class_folder, filename]
    val_lookup = {f"{cls}/{fname}": idx for idx, (cls, fname) in enumerate(val_filenames)}

    # Make a filename column in df for matching
    df["file_id"] = df["path"].apply(lambda x: "/".join(x.split("/")[-2:]))  # assumes you stored full paths in prepare()

    df_val = df[df["file_id"].isin(val_lookup.keys())]
    df_val = df_val.sort_values("file_id")  # ensure same order as val_gen

    X_val = df_val.drop(["label", "file_id"], axis=1).values
    y_val = le.transform(df_val["label"].values)

    y_pred_rf = rf.predict(X_val)
    print(f"[EVAL] RF predictions done on {len(y_val)} images")

    # --- Ensemble prediction (majority vote) ---
    y_pred_final = []
    for c, r in zip(y_pred_cnn, y_pred_rf):
        y_pred_final.append(max([c, r], key=[c, r].count))
    y_pred_final = np.array(y_pred_final)

    # --- Report ---
    print("[EVAL] Ensemble Classification Report:")
    print(classification_report(y_true, y_pred_final, target_names=list(val_gen.class_indices.keys())))

    cm = confusion_matrix(y_true, y_pred_final)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix - Ensemble (CNN+RF)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# --------------------------
# Realtime demo with webcam
# --------------------------
def run_demo(cnn_model_path=CNN_MODEL_PATH, rf_model_path=RF_MODEL_PATH, label_encoder_path=LABEL_ENCODER_PATH):
    # load models
    le = joblib.load(label_encoder_path)
    classes = le.classes_
    cnn = None
    if os.path.exists(cnn_model_path):
        cnn = tf.keras.models.load_model(cnn_model_path)
        print("[DEMO] Loaded CNN")
    rf = None
    if os.path.exists(rf_model_path):
        rf = joblib.load(rf_model_path)
        print("[DEMO] Loaded RF")
    if cnn is None and rf is None:
        raise RuntimeError("No model found. Train at least one model first.")

    tts = pyttsx3.init()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        prev_text = ""
        buffer_text = ""  # accumulate detected letters into a word (basic)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(frame_rgb)
            display_text = ""
            pred_label = None
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                # prepare landmark array
                lm = []
                for p in hand.landmark:
                    lm.extend([p.x, p.y, p.z])
                # CNN prediction: we need a cropped/resized RGB image of the hand region
                # We'll crop bounding box around the hand landmarks
                h, w, _ = frame.shape
                xs = [int(p.x * w) for p in hand.landmark]
                ys = [int(p.y * h) for p in hand.landmark]
                x1, x2 = max(min(xs)-20,0), min(max(xs)+20,w)
                y1, y2 = max(min(ys)-20,0), min(max(ys)+20,h)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = cv2.resize(frame, IMG_SIZE)
                else:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop = cv2.resize(crop, IMG_SIZE)
                crop_norm = crop.astype(np.float32)/255.0
                probs_list = []
                if cnn is not None:
                    p_cnn = cnn.predict(np.expand_dims(crop_norm, axis=0), verbose=0)[0]
                    probs_list.append(p_cnn)
                if rf is not None:
                    p_rf = rf_predict_proba(rf, lm)
                    # ensure rf.prob length equals number of classes in encoder
                    # If different, attempt to map. For this implementation we assume consistent training.
                    probs_list.append(p_rf)
                avg = np.mean(probs_list, axis=0)
                pred_idx = int(np.argmax(avg))
                pred_label = le.inverse_transform([pred_idx])[0]
                conf = float(avg[pred_idx])
                display_text = f"{pred_label} ({conf:.2f})"
                # simple debounce: if same label for a few frames -> add to buffer
                if display_text and conf > 0.75:
                    if prev_text == pred_label:
                        # append only if not repeated too fast
                        if len(buffer_text)==0 or buffer_text[-1] != pred_label:
                            buffer_text += pred_label
                            tts.say(pred_label)
                            tts.runAndWait()
                    else:
                        # don't append yet, wait for stabilization
                        pass
                prev_text = pred_label
            # Draw UI
            cv2.rectangle(frame, (0,0), (400, 60), (0,0,0), -1)
            cv2.putText(frame, "GestoConnect - Press q to quit", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Detected: {display_text}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Buffer: {buffer_text}", (420,45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
            cv2.imshow("GestoConnect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                buffer_text = ""  # clear
    cap.release()
    cv2.destroyAllWindows()

# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="GestoConnect - full pipeline")
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('prepare', help='Extract mediapipe landmarks for dataset images and save CSV')
    sub.add_parser('train_cnn', help='Train CNN on image dataset')
    sub.add_parser('train_rf', help='Train Random Forest on landmark CSV')
    sub.add_parser('evaluate', help='Evaluate ensemble on test set (loads models)')
    sub.add_parser('demo', help='Run real-time demo (loads models)')

    args = parser.parse_args()
    if args.cmd == 'prepare':
        extract_landmarks_from_images(DATA_DIR, csv_out=LANDMARK_CSV)
    elif args.cmd == 'train_cnn':
        train_cnn(DATA_DIR, out_model=CNN_MODEL_PATH, label_encoder_path=LABEL_ENCODER_PATH)
    elif args.cmd == 'train_rf':
        if not os.path.exists(LANDMARK_CSV):
            print("[WARN] landmarks CSV not found. Running 'prepare' first.")
            extract_landmarks_from_images(DATA_DIR, csv_out=LANDMARK_CSV)
        train_rf(LANDMARK_CSV, out_model=RF_MODEL_PATH, label_encoder_path=LABEL_ENCODER_PATH)
    elif args.cmd == 'evaluate':
        evaluate_models(DATA_DIR, LANDMARK_CSV, CNN_MODEL_PATH, RF_MODEL_PATH, LABEL_ENCODER_PATH)
    elif args.cmd == 'demo':
        run_demo(CNN_MODEL_PATH, RF_MODEL_PATH, LABEL_ENCODER_PATH)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
