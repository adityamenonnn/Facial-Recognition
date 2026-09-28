# Facial Recognition with KNN

A real-time facial recognition system using OpenCV for face detection and a K-Nearest Neighbours classifier built from scratch in NumPy.

## How It Works

### 1. Data Collection (`FacialData.py`)

Captures face samples from a webcam:
- Uses Haar Cascade (`haarcascade_frontalface_alt.xml`) to detect faces in each frame
- Crops, resizes to 100x100 pixels, and stores one sample every 10 frames
- Saves the collected face data as a `.npy` file in `face_dataset/`

```bash
python3 FacialData.py
# Enter a name when prompted, then face the camera
# Press 'q' to stop. More frames = better recognition.
```

### 2. Recognition (`FaceRecognition.py`)

Identifies faces in real-time using all stored face datasets:
- Loads all `.npy` files from `face_dataset/`, each representing one person
- For each detected face, flattens the 100x100 crop into a feature vector
- Runs KNN (k=5) using Euclidean distance against the training set
- Draws a bounding box and predicted name on the video feed

```bash
python3 FaceRecognition.py
# Press 'q' to quit
```

### KNN Classifier

The classifier is implemented from scratch without scikit-learn:
- Computes Euclidean distance between the test face vector and every training sample
- Selects the k nearest neighbours and returns the most frequent label
- Simple and effective for small datasets where each person has ~20-50 samples

## Project Structure

| File | Purpose |
|------|---------|
| `FacialData.py` | Webcam face capture and dataset creation |
| `FaceRecognition.py` | Real-time face detection and KNN recognition |
| `main.py` | PyCharm template (unused) |
| `videoread.py` | Basic webcam test utility |
| `haarcascade_frontalface_alt.xml` | Pre-trained Haar Cascade for face detection |
| `face_dataset/` | Stored face data as `.npy` files (one per person) |

## Requirements

- Python 3
- OpenCV (`pip install opencv-python`)
- NumPy
