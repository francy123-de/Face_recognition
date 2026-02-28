#!/usr/bin/env python3
"""
Model Accuracy Testing Tool
Tests how well the model recognizes each trained student
"""

import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace
import json
from collections import defaultdict
import random

print("🎯 MODEL ACCURACY TESTING TOOL")
print("=" * 70)

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
known_embeddings = np.array(model_data['known_embeddings'])
known_labels = np.array(model_data['known_labels'])

print(f"✅ Model: {model_name}")
print(f"✅ Students: {len(label_encoder.classes_)}")
print(f"✅ Total training samples: {len(known_embeddings)}")

# Count samples per student
samples_per_student = defaultdict(int)
for label in known_labels:
    samples_per_student[label] += 1

print("\n📊 Training Data Distribution:")
for student in sorted(label_encoder.classes_):
    count = samples_per_student[student]
    print(f"   {student}: {count} images")

print("\n" + "=" * 70)
print("🧪 TESTING MODEL ACCURACY")
print("=" * 70)

def extract_embedding(image_path):
    """Extract embedding from image"""
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
        return None
    except Exception as e:
        print(f"   ⚠️  Error extracting embedding: {e}")
        return None

def recognize_face(embedding, threshold=0.25):
    """Recognize face from embedding"""
    if embedding is None:
        return "Unknown", 0.0, 0.0
    
    embedding = embedding.reshape(1, -1)
    probabilities = classifier.predict_proba(embedding)[0]
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    # Calculate gap
    sorted_indices = np.argsort(probabilities)[::-1]
    top_1_conf = probabilities[sorted_indices[0]]
    top_2_conf = probabilities[sorted_indices[1]] if len(sorted_indices) > 1 else 0
    gap = top_1_conf - top_2_conf
    
    return predicted_label, confidence, gap

# Test each student
print("\n🔍 Testing each student with their training images...\n")

overall_results = {
    'total_tests': 0,
    'correct': 0,
    'incorrect': 0,
    'per_student': {}
}

for student_name in sorted(label_encoder.classes_):
    print(f"📸 Testing: {student_name}")
    print("-" * 70)
    
    dataset_path = f"dataset/{student_name}"
    
    if not os.path.exists(dataset_path):
        print(f"   ⚠️  Dataset folder not found: {dataset_path}")
        continue
    
    # Get all images
    images = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(images) == 0:
        print(f"   ⚠️  No images found in {dataset_path}")
        continue
    
    # Test random sample (max 10 images to save time)
    test_images = random.sample(images, min(10, len(images)))
    
    student_results = {
        'total': len(test_images),
        'correct': 0,
        'incorrect': 0,
        'confidences': [],
        'gaps': [],
        'predictions': []
    }
    
    for img_name in test_images:
        img_path = os.path.join(dataset_path, img_name)
        
        # Extract embedding
        embedding = extract_embedding(img_path)
        
        if embedding is None:
            print(f"   ❌ {img_name}: Failed to extract embedding")
            continue
        
        # Recognize
        predicted, confidence, gap = recognize_face(embedding)
        
        # Check if correct
        is_correct = (predicted == student_name)
        
        if is_correct:
            student_results['correct'] += 1
            overall_results['correct'] += 1
            status = "✅"
        else:
            student_results['incorrect'] += 1
            overall_results['incorrect'] += 1
            status = "❌"
        
        student_results['confidences'].append(confidence)
        student_results['gaps'].append(gap)
        student_results['predictions'].append(predicted)
        overall_results['total_tests'] += 1
        
        print(f"   {status} {img_name}: {predicted} (conf: {confidence*100:.1f}%, gap: {gap*100:.1f}%)")
    
    # Calculate student statistics
    if student_results['total'] > 0:
        accuracy = (student_results['correct'] / student_results['total']) * 100
        avg_confidence = np.mean(student_results['confidences']) * 100 if student_results['confidences'] else 0
        avg_gap = np.mean(student_results['gaps']) * 100 if student_results['gaps'] else 0
        
        print(f"\n   📊 {student_name} Results:")
        print(f"      Accuracy: {accuracy:.1f}% ({student_results['correct']}/{student_results['total']})")
        print(f"      Avg Confidence: {avg_confidence:.1f}%")
        print(f"      Avg Gap: {avg_gap:.1f}%")
        
        overall_results['per_student'][student_name] = {
            'accuracy': accuracy,
            'correct': student_results['correct'],
            'total': student_results['total'],
            'avg_confidence': avg_confidence,
            'avg_gap': avg_gap
        }
    
    print()

# Overall statistics
print("=" * 70)
print("📊 OVERALL MODEL ACCURACY")
print("=" * 70)

if overall_results['total_tests'] > 0:
    overall_accuracy = (overall_results['correct'] / overall_results['total_tests']) * 100
    
    print(f"\n✨ Total Accuracy: {overall_accuracy:.2f}%")
    print(f"   Correct: {overall_results['correct']}")
    print(f"   Incorrect: {overall_results['incorrect']}")
    print(f"   Total Tests: {overall_results['total_tests']}")
    
    # Per-student summary
    print("\n📋 Per-Student Accuracy:")
    print("-" * 70)
    print(f"{'Student':<20} {'Accuracy':<12} {'Tests':<10} {'Avg Conf':<12} {'Avg Gap'}")
    print("-" * 70)
    
    for student, results in sorted(overall_results['per_student'].items()):
        print(f"{student:<20} {results['accuracy']:>6.1f}%     {results['correct']}/{results['total']:<6} {results['avg_confidence']:>6.1f}%      {results['avg_gap']:>6.1f}%")
    
    # Performance rating
    print("\n" + "=" * 70)
    print("🎯 MODEL PERFORMANCE RATING")
    print("=" * 70)
    
    if overall_accuracy >= 95:
        rating = "EXCELLENT ⭐⭐⭐⭐⭐"
        comment = "Your model is performing exceptionally well!"
    elif overall_accuracy >= 85:
        rating = "VERY GOOD ⭐⭐⭐⭐"
        comment = "Your model is performing very well!"
    elif overall_accuracy >= 75:
        rating = "GOOD ⭐⭐⭐"
        comment = "Your model is performing well, but could be improved."
    elif overall_accuracy >= 60:
        rating = "FAIR ⭐⭐"
        comment = "Your model needs improvement. Consider retraining with more images."
    else:
        rating = "POOR ⭐"
        comment = "Your model needs significant improvement. Retrain with better quality images."
    
    print(f"\nRating: {rating}")
    print(f"Comment: {comment}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 70)
    
    if overall_accuracy < 85:
        print("1. ⚠️  Add more training images (aim for 30+ per person)")
        print("2. ⚠️  Ensure good lighting in training images")
        print("3. ⚠️  Include various angles and expressions")
        print("4. ⚠️  Remove blurry or low-quality images")
        print("5. ⚠️  Retrain the model after adding more images")
    else:
        print("1. ✅ Your model is well-trained!")
        print("2. ✅ Maintain good lighting during recognition")
        print("3. ✅ Ensure faces are clearly visible")
        print("4. ✅ Keep adding images to improve over time")
    
    # Identify weak performers
    weak_students = [s for s, r in overall_results['per_student'].items() if r['accuracy'] < 70]
    if weak_students:
        print(f"\n⚠️  Students needing more training data:")
        for student in weak_students:
            results = overall_results['per_student'][student]
            print(f"   - {student}: {results['accuracy']:.1f}% accuracy (add more images!)")
    
    # Save results to file
    results_file = "model_accuracy_report.json"
    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {results_file}")

else:
    print("\n❌ No tests were performed. Check your dataset folders.")

print("\n" + "=" * 70)
print("✅ ACCURACY TEST COMPLETE!")
print("=" * 70)

# Additional insights
print("\n📈 UNDERSTANDING THE METRICS:")
print("-" * 70)
print("Accuracy:    % of correct predictions (higher is better)")
print("Confidence:  How sure the model is (higher is better)")
print("Gap:         Difference between 1st and 2nd choice (higher is better)")
print("\nIdeal values:")
print("  Accuracy:   > 90%")
print("  Confidence: > 60%")
print("  Gap:        > 15%")

print("\n💡 TIP: Run this test after retraining to track improvements!")
