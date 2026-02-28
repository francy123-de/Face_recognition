# 🎓 Face Recognition Attendance System

Professional AI-powered attendance system using DeepFace and Facenet512 for 97-99% accuracy, based in bsc & bit members.

## ✨ Features

- **High Accuracy**: 97-99% face recognition accuracy using Facenet512
- **Automatic Attendance**: Marks students present when recognized
- **Multiple Interfaces**:
  - Desktop application with webcam
  - Web-based interface (Flask)
- **Attendance Records**: Saves to CSV and JSON formats
- **Real-time Recognition**: Live face detection and recognition
- **Student Management**: Easy to add new students

## 📋 Requirements

- Python 3.10+
- Webcam
- Windows/Linux/Mac

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Your dataset structure should be:
```
dataset_cropped/
├── Student1/
│   ├── 00.jpeg
│   ├── 01.jpeg
│   └── ...
├── Student2/
│   └── ...
```

## 📖 Usage Guide

### Step 1: Train the Model

```bash
python deepface_recognition.py
```

This will:
- Load all student images from `dataset_cropped/`
- Extract 512-D face embeddings using Facenet512
- Train an SVM classifier
- Save the model to `model/deepface_recognition.pkl`

Expected output: **100% test accuracy**

### Step 2: Run Attendance System

#### Option A: Desktop Application

```bash
python attendance_system.py
```

**Controls:**
- `q` or `ESC` - Quit and show report
- `r` - Show current attendance report
- `SPACE` - Pause/Resume
- `s` - Save screenshot

#### Option B: Web Application

```bash
python web_attendance_app.py
```

Then open your browser to: **http://localhost:5000**

Features:
- Live video feed with face recognition
- Real-time attendance statistics
- Present/Absent student lists
- Download attendance CSV

### Step 3: Add New Students

```bash
python add_new_student.py
```

Follow the prompts to:
1. Enter student name
2. Capture 20 images from webcam
3. Retrain the model

## 📁 Project Structure

```
Face_Detection/
├── attendance_system.py          # Desktop attendance app
├── web_attendance_app.py          # Web-based attendance app
├── deepface_recognition.py        # Model training script
├── add_new_student.py             # Add new students
├── clean_dataset.py               # Clean and resize images
├── face_cropper.py                # Extract faces from images
├── deepface_webcam.py             # Webcam recognition (debug)
├── students.csv                   # Student list
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── dataset/                       # Original images
├── dataset_cropped/               # Processed face images
├── model/                         # Trained models
│   └── deepface_recognition.pkl
├── attendance_records/            # Attendance files
│   ├── attendance_YYYY-MM-DD.csv
│   └── attendance_YYYY-MM-DD.json
└── templates/                     # Web app templates
    └── index.html
```

## 🎯 Attendance Records

Attendance is saved in two formats:

### CSV Format
```csv
Student Name,Date,Time,Confidence,Status
John Doe,2026-02-07,16:11:31,85.5%,Present
```

### JSON Format
```json
{
  "date": "2026-02-07",
  "records": {
    "John Doe": {
      "timestamp": "2026-02-07 16:11:31",
      "confidence": "85.5%",
      "status": "Present"
    }
  },
  "marked": ["John Doe"],
  "total_present": 1
}
```

## 🔧 Troubleshooting

### Model Not Found Error
```bash
# Train the model first
python deepface_recognition.py
```

### Low Recognition Accuracy
1. Ensure good lighting when capturing images
2. Capture images from multiple angles
3. Add more images per student (20+ recommended)
4. Retrain the model after adding images

### Webcam Not Opening
- Check if webcam is connected
- Close other applications using the webcam
- Try changing camera index in code (0, 1, 2, etc.)

### Flask Not Installed
```bash
pip install flask
```

## 📊 Model Performance

- **Model**: Facenet512 (DeepFace)
- **Embedding Size**: 512 dimensions
- **Classifier**: SVM with RBF kernel
- **Test Accuracy**: 100%
- **Cross-Validation**: 100% (5-fold)
- **Recognition Threshold**: 35% confidence

## 🎨 Customization

### Change Recognition Threshold

Edit in `attendance_system.py` or `web_attendance_app.py`:

```python
def recognize_face(self, face_embedding, confidence_threshold=0.35):
    # Lower = more lenient, Higher = more strict
```

### Change Recognition Interval

Edit in `attendance_system.py`:

```python
recognition_interval = 2.0  # seconds between scans
```

### Change Model

Edit in `deepface_recognition.py`:

```python
# Available models: VGG-Face, Facenet, Facenet512, OpenFace, 
# DeepFace, DeepID, ArcFace, Dlib, SFace
self.model_name = 'Facenet512'  # Change here
```

## 📝 Notes

- Each student is marked present only once per day
- Attendance files are created daily
- The system requires good lighting for best results
- Minimum 5-10 images per student recommended
- Face should be clearly visible and front-facing

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created with ❤️ using DeepFace, OpenCV, and Python

---

**Happy Coding! 🚀**
