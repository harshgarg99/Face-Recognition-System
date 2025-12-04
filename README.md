# 🧠 Face Recognition based on MTCNN and FaceNet

A beginner-friendly real-time face recognition system built using **MTCNN** for face detection and **FaceNet (InceptionResNetV2)** for face embeddings, combined with a simple **Pygame UI**.

> ⚠️ **Note:**  
> This is a learning/academic project — not a production-grade high-security system.  
> Accuracy may vary based on lighting, camera quality, and dataset size.

---

## 🚀 Features

- 🔍 Real-time face detection using **MTCNN**
- 🧬 Face recognition using **FaceNet embeddings**
- 🎯 Cosine similarity-based matching
- 🚫 Detects **unknown** faces when threshold not met
- 🎨 Graphical user interface built with **Pygame**
- 🗂 Face encodings stored using **Pickle (.pkl)**
- 🟩 Green bounding box → recognized
- 🟥 Red bounding box → unknown
- 📸 Works with any standard webcam

---

## 🧪 Tech Stack

| Category        | Technologies               |
| --------------- | -------------------------- |
| Language        | Python                     |
| Deep Learning   | TensorFlow, Keras, FaceNet |
| Face Detection  | MTCNN                      |
| Computer Vision | OpenCV                     |
| Math Utils      | NumPy, SciPy               |
| Data Storage    | Pickle                     |
| GUI             | Pygame                     |

---

## 🗂️ Project Structure

```bash
Face-Recognition-System/
│── assets/                # UI images, banner, background
│── encodings/             # Stored face encodings
│── Faces/                 # Raw face images (optional)
│── MEDIA/                 # Additional files
│── env/                   # Virtual environment (ignored)
│── architecture.py        # FaceNet model
│── train_v2.py            # Preprocessing, L2-normalizer
│── Button.py              # Custom pygame button class
│── main.py                # Main app + recognition loop
│── facenet_keras_weights.h5
│── requirements.txt
│── README.md
│── .gitignore
```

---

## 🔧 How It Works (Pipeline)

```text
Webcam Frame → MTCNN Detector → Face Crop → Resize (160x160)
       ↓
Normalize → FaceNet Encoder → 128-D Embedding
       ↓
Cosine Distance Matching → Classified as Known / Unknown
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/harshgarg99/Face-Recognition-System.git
cd Face-Recognition-System
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv env
env\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Program

```bash
python main.py
```

---

## ❗ Known Limitations

- Not resistant to **photo attacks**
- Sensitive to lighting and face angles
- Small dataset → lower accuracy
- No anti-spoofing module yet
- FaceNet model is not fine-tuned for your custom faces

---

## 🔮 Future Improvements

- ✨ Add anti-spoofing (blink detection, depth map, rPPG pulse)
- ✨ Improve recognition threshold logic
- ✨ Add “Register New Face” feature in UI
- ✨ Replace MTCNN with **RetinaFace** for higher accuracy
- ✨ GPU acceleration support
- ✨ Export logs + performance metrics

---

## ⭐ Support

If you like this project, please ⭐ star the repository to support the development!

---

## 📄 License

This project is licensed under the **MIT License**.
