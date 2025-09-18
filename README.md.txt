🖐️ GestoConnect: Hand Gesture Recognition for ASL

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-orange)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**GestoConnect** is a deep ensemble-based system for **real-time American Sign Language (ASL) hand gesture recognition**.
It combines:

* 📷 **CNN model** trained on gesture images
* ✋ **Random Forest model** trained on **MediaPipe hand landmarks**
* 🎤 **Text-to-Speech output** for accessibility

✅ Works both offline (trained models) and real-time with webcam.

---

## 📂 Project Structure

```
GestoConnect-ASL/
│── gestoconnect.py          # Main pipeline (train, evaluate, demo)
│── requirements.txt         # Python dependencies
│── README.md                # This file
│── LICENSE                  # Open source license
│── dataset/
│    └── sample/             # Few sample images (for repo structure)
│        ├── A/
│        ├── B/
│        └── C/
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repo

```bash
git clone https://github.com/<your-username>/GestoConnect-ASL.git
cd GestoConnect-ASL
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Dataset

* This project uses the **ASL Alphabet dataset** (A–Z hand gestures).
* After download, organize it like:

```
dataset/images/
   ├── A/
   │   ├── img001.jpg
   │   ├── ...
   ├── B/
   └── C/
```

---

## 🏋️ Training

### Prepare landmarks

```bash
python gestoconnect.py prepare
```

### Train CNN

```bash
python gestoconnect.py train_cnn
```

### Train Random Forest

```bash
python gestoconnect.py train_rf
```

---

## 📈 Evaluation

```bash
python gestoconnect.py evaluate
```

Generates classification report + confusion matrix.

---

## 🎥 Real-Time Demo

```bash
python gestoconnect.py demo
```

* Opens webcam
* Detects gestures in real-time
* Displays bounding box + predicted label
* Speaks out prediction (via TTS)

---

## 🤖 Pretrained Models

Download pretrained models (optional, skip training): https://drive.google.com/drive/folders/1Av25YoEbNbfTYPHEYVMewGwJZkdDiecX?usp=sharing

* CNN Model:
* Random Forest Model
* Label Encoder

Place them in the project root:

```
cnn_model.h5
rf_model.pkl
label_encoder.pkl
```

---

## 🛠️ Tech Stack

* **TensorFlow / Keras** – CNN training
* **scikit-learn** – Random Forest & evaluation
* **MediaPipe** – Hand landmark extraction
* **OpenCV** – Image processing & webcam feed
* **pyttsx3** – Text-to-Speech

---