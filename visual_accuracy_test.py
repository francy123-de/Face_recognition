#!/usr/bin/env python3
"""
Visual Model Accuracy Test
Shows images and predictions side-by-side
"""

import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace
import random

print("👁️  VISUAL ACCURACY TEST")
print("=" * 60)

# Load model
MODEL_PATH = "model/deepface_recognition.pkl"

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found!")
    exit(1)

with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

model_name = model_data['model_name']
classifier = model_data['classifier']
label_encoder = model_data['label_encoder']

print(f"✅ Model loaded: {model_name}")
print(f"✅ Students: {', '.join(label_encoder.classes_)}")

def extract_embedding(image_path):
    """Extract embedding"""
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            enforce_detection=False,
            detector_backend='skip',
            align=True,
            normalization='Facenet2018'
        )
        if len(embedding_objs) > 0:
            return np.array(embedding_objs[0]['embedding'])
    except:
        pass
    return None

def recognize_face(embedding):
    """Recognize face"""
    if embedding is None:
        return "Unknown", 0.0, 0.0
    
    embedding = embedding.reshape(1, -1)
    probabilities = classifier.predict_proba(embedding)[0]
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    sorted_indices = np.argsort(probabilities)[::-1]
    top_1_conf = probabilities[sorted_indices[0]]
    top_2_conf = probabilities[sorted_indices[1]] if len(sorted_indices) > 1 else 0
    gap = top_1_conf - top_2_conf
    
    return predicted_label, confidence, gap

print("\n🎬 Starting visual test...")
print("Controls:")
print("  SPACE - Next image")
print("  'q' or ESC - Quit")
print("=" * 60)

# Collect all test images
all_images = []
for student_name in label_encoder.classes_:
    dataset_path = f"dataset/{student_name}"
    if os.path.exists(dataset_path):
        images = [f for f in os.listdir(dataset_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            all_images.append((student_name, os.path.join(dataset_path, img)))

if len(all_images) == 0:
    print("❌ No images found in dataset folders!")
    exit(1)

# Shuffle for random testing
random.shuffle(all_images)

print(f"✅ Found {len(all_images)} images to test\n")

# Statistics
stats = {
    'total': 0,
    'correct': 0,
    'incorrect': 0
}

try:
    for true_name, img_path in all_images:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Extract and recognize
        embedding = extract_embedding(img_path)
        predicted, confidence, gap = recognize_face(embedding)
        
        # Check if correct
        is_correct = (predicted == true_name)
        stats['total'] += 1
        if is_correct:
            stats['correct'] += 1
        else:
            stats['incorrect'] += 1
        
        # Calculate current accuracy
        current_accuracy = (stats['correct'] / stats['total']) * 100
        
        # Create display
        display = img.copy()
        h, w = display.shape[:2]
        
        # Resize if too large
        max_size = 800
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)
            display = cv2.resize(display, None, fx=scale, fy=scale)
            h, w = display.shape[:2]
        
        # Add black bar at bottom for text
        bar_height = 180
        display = cv2.copyMakeBorder(display, 0, bar_height, 0, 0, 
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Determine color based on correctness
        if is_correct:
            color = (0, 255, 0)  # Green
            status = "✓ CORRECT"
        else:
            color = (0, 0, 255)  # Red
            status = "✗ WRONG"
        
        # Add text
        y_offset = h + 30
        cv2.putText(display, f"TRUE: {true_name}", (10, y_offset),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        y_offset += 35
        cv2.putText(display, f"PREDICTED: {predicted}", (10, y_offset),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
        y_offset += 35
        cv2.putText(display, f"Confidence: {confidence*100:.1f}%", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(display, f"Gap: {gap*100:.1f}%", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        y_offset += 35
        cv2.putText(display, f"{status} | Accuracy: {current_accuracy:.1f}% ({stats['correct']}/{stats['total']})",
                   (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        
        # Show
        cv2.imshow('Model Accuracy Test', display)
        
        # Wait for key
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break

except KeyboardInterrupt:
    print("\n⏹️  Stopped by user")

finally:
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    if stats['total'] > 0:
        accuracy = (stats['correct'] / stats['total']) * 100
        print(f"\n✨ Overall Accuracy: {accuracy:.2f}%")
        print(f"   Correct: {stats['correct']}")
        print(f"   Incorrect: {stats['incorrect']}")
        print(f"   Total: {stats['total']}")
        
        if accuracy >= 90:
            print("\n🎉 EXCELLENT! Your model is very accurate!")
        elif accuracy >= 75:
            print("\n👍 GOOD! Your model is working well!")
        elif accuracy >= 60:
            print("\n⚠️  FAIR. Consider adding more training images.")
        else:
            print("\n❌ POOR. Please retrain with better quality images.")
    
    print("\n✅ Test complete!")
