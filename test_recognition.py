#!/usr/bin/env python3
"""
Quick Recognition Test - See what the model is detecting
"""

import cv2
import pickle
import numpy as np
from deepface import DeepFace
import os

print("🔍 Testing Face Recognition Model")
print("=" * 60)

# Load model
MODEL_PATH = "model/deepface_recognition.pkl"

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found! Please train the model first.")
    print("Run: python deepface_recognition.py")
    exit(1)

print("📦 Loading model...")
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

model_name = model_data['model_name']
classifier = model_data['classifier']
label_encoder = model_data['label_encoder']

print(f"✅ Model loaded: {model_name}")
print(f"✅ Trained students: {len(label_encoder.classes_)}")
print(f"📋 Students in model:")
for i, name in enumerate(label_encoder.classes_, 1):
    print(f"   {i}. {name}")

print("\n" + "=" * 60)
print("🎥 Starting camera test...")
print("Controls:")
print("  SPACE - Test recognition on current frame")
print("  'q' or ESC - Quit")
print("=" * 60)

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Warm up
for _ in range(10):
    cap.read()

print("✅ Camera ready!\n")

def extract_and_recognize(face_img):
    """Extract embedding and recognize"""
    try:
        # Resize
        face_resized = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_CUBIC)
        
        # Enhance
        lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        face_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Save temp
        temp_path = "temp_test.jpg"
        cv2.imwrite(temp_path, face_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        # Extract embedding
        print("   🔄 Extracting embedding...")
        embedding_objs = DeepFace.represent(
            img_path=temp_path,
            model_name=model_name,
            enforce_detection=False,
            detector_backend='skip',
            align=True,
            normalization='Facenet2018'
        )
        
        os.remove(temp_path)
        
        if len(embedding_objs) == 0:
            print("   ❌ Failed to extract embedding")
            return None, 0.0
        
        embedding = np.array(embedding_objs[0]['embedding']).reshape(1, -1)
        print(f"   ✅ Embedding extracted (dim: {embedding.shape[1]})")
        
        # Recognize
        probabilities = classifier.predict_proba(embedding)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        # Show top 3 predictions
        sorted_indices = np.argsort(probabilities)[::-1]
        print("\n   📊 Top 3 Predictions:")
        for i, idx in enumerate(sorted_indices[:3], 1):
            name = label_encoder.inverse_transform([idx])[0]
            conf = probabilities[idx]
            print(f"      {i}. {name}: {conf*100:.2f}%")
        
        # Calculate gap
        top_1_conf = probabilities[sorted_indices[0]]
        top_2_conf = probabilities[sorted_indices[1]] if len(sorted_indices) > 1 else 0
        gap = top_1_conf - top_2_conf
        
        print(f"\n   🎯 Best Match: {predicted_label}")
        print(f"   📈 Confidence: {confidence*100:.2f}%")
        print(f"   📊 Gap to 2nd: {gap*100:.2f}%")
        
        # Decision
        if confidence > 0.7:
            print(f"   ✅ ACCEPTED (High confidence)")
        elif confidence > 0.5 and gap > 0.1:
            print(f"   ✅ ACCEPTED (Good confidence + gap)")
        elif confidence > 0.35 and gap > 0.15:
            print(f"   ✅ ACCEPTED (Clear winner)")
        elif confidence > 0.25 and gap > 0.08:
            print(f"   ⚠️  ACCEPTABLE (Low but has gap)")
        else:
            print(f"   ❌ REJECTED (Too uncertain)")
        
        return predicted_label, confidence
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        
        # Draw faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to test", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE = Test | Q = Quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Recognition Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and len(faces) > 0:
            print("\n" + "="*60)
            print("🧪 TESTING RECOGNITION...")
            print("="*60)
            
            # Get largest face
            largest = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest
            
            # Extract with padding
            padding = int(w * 0.2)
            y1 = max(0, y - padding)
            y2 = min(frame.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(frame.shape[1], x + w + padding)
            
            face_crop = frame[y1:y2, x1:x2]
            
            # Test recognition
            name, conf = extract_and_recognize(face_crop)
            
            print("="*60 + "\n")
        
        elif key == ord('q') or key == 27:
            break

except KeyboardInterrupt:
    print("\n⏹️  Stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test complete!")
