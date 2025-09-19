🖐️ GestoConnect: Hand Gesture Recognition for ASL

📌 Overview

GestoConnect is a deep ensemble-based system for real-time American Sign Language (ASL) hand gesture recognition.
It combines:

📷 CNN model trained on gesture images

✋ Random Forest model trained on MediaPipe hand landmarks

🎤 Text-to-Speech output for accessibility

✅ Works both offline (trained models) and real-time with webcam.

📂 Project Structure
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

⚙️ Setup Instructions

1️⃣ Clone Repo

git clone https://github.com/<your-username>/GestoConnect-ASL.git

cd GestoConnect-ASL

2️⃣ Create Virtual Environment
```
python -m venv venv

source venv/bin/activate   # Linux / Mac

venv\Scripts\activate      # Windows
```
3️⃣ Install Requirements
```
pip install --upgrade pip

pip install -r requirements.txt
```
📊 Dataset

This project uses the ASL Alphabet dataset (A–Z hand gestures).

Download full dataset here: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

After download, organize it like:
```
dataset/images/
   ├── A/
   │   ├── img001.jpg
   │   ├── ...
   ├── B/
   └── C/
```
🏋️ Training

Prepare landmarks
```
python gestoconnect.py prepare
```
Train CNN
```
python gestoconnect.py train_cnn
```
Train Random Forest
```
python gestoconnect.py train_rf
```
📈 Evaluation
```
python gestoconnect.py evaluate
```

Generates classification report + confusion matrix.

🎥 Real-Time Demo
```
python gestoconnect.py demo
```

Opens webcam

Detects gestures in real-time

Displays bounding box + predicted label

Speaks out prediction (via TTS)

🤖 Pretrained Models

Download pretrained models (optional, skip training):

CNN Model: https://drive.google.com/file/d/1n7AQMEUIeMFYpRDU1CwXeR4rDU-FLEVI/view?usp=sharing

Random Forest Model: https://drive.google.com/file/d/1pPRCNF_ShIFKMw7K1qeeapULdWubOUG_/view?usp=sharing

Label Encoder: https://drive.google.com/file/d/1meoDXuJM_RJDE-d9uJivCFgR_cBD2Iju/view?usp=sharing

Place them in the project root:

cnn_model.h5

rf_model.pkl

label_encoder.pkl

🛠️ Tech Stack

TensorFlow / Keras – CNN training

scikit-learn – Random Forest & evaluation

MediaPipe – Hand landmark extraction

OpenCV – Image processing & webcam feed

pyttsx3 – Text-to-Speech

📜 License

MIT License – free to use, modify, and distribute.
