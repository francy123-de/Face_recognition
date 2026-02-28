import cv2
import os
from datetime import datetime

def capture_student_images(student_name, num_images=20):
    """Capture images of a new student from webcam"""
    
    # Create student folder
    dataset_path = "dataset_cropped"
    student_folder = os.path.join(dataset_path, student_name)
    os.makedirs(student_folder, exist_ok=True)
    
    print(f"📸 Capturing images for: {student_name}")
    print(f"Target: {num_images} images")
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("  - Look at the camera")
    print("  - Move your head slightly (left, right, up, down)")
    print("  - Try different expressions")
    print("  - Press SPACE to capture")
    print("  - Press 'q' to finish")
    print("="*60 + "\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    captured_count = 0
    
    while captured_count < num_images:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display progress
        progress_text = f"Captured: {captured_count}/{num_images}"
        cv2.putText(frame, progress_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, "SPACE: Capture | Q: Quit", (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow(f'Capturing {student_name} - Press SPACE', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # SPACE
            if len(faces) > 0:
                # Get the largest face
                faces_sorted = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                x, y, w, h = faces_sorted[0]
                
                # Extract face with padding
                padding = 30
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                
                face_crop = frame[y1:y2, x1:x2]
                
                # Resize to 160x160 (standard size)
                face_resized = cv2.resize(face_crop, (160, 160))
                
                # Save image
                filename = f"{captured_count:02d}.jpeg"
                filepath = os.path.join(student_folder, filename)
                cv2.imwrite(filepath, face_resized)
                
                captured_count += 1
                print(f"✓ Captured {captured_count}/{num_images}: {filename}")
                
                # Visual feedback
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)
                cv2.imshow(f'Capturing {student_name} - Press SPACE', frame)
                cv2.waitKey(200)  # Show green flash
            else:
                print("✗ No face detected! Try again.")
        
        elif key == ord('q'):
            print("\n⚠️  Capture cancelled by user")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    if captured_count >= num_images:
        print(f"\n✅ Successfully captured {captured_count} images!")
        print(f"📁 Saved to: {student_folder}")
        print("\n🔄 Now run 'python deepface_recognition.py' to retrain the model")
        return True
    else:
        print(f"\n⚠️  Only captured {captured_count}/{num_images} images")
        return False

def main():
    print("🎯 Add New Student to Face Recognition System")
    print("="*60)
    
    student_name = input("Enter student name: ").strip()
    
    if not student_name:
        print("❌ Invalid name!")
        return
    
    # Check if student already exists
    dataset_path = "dataset_cropped"
    student_folder = os.path.join(dataset_path, student_name)
    
    if os.path.exists(student_folder):
        response = input(f"⚠️  {student_name} already exists. Add more images? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
        
        # Count existing images
        existing_images = len([f for f in os.listdir(student_folder) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"📊 {student_name} currently has {existing_images} images")
    
    num_images = input("How many images to capture? (default: 20): ").strip()
    num_images = int(num_images) if num_images.isdigit() else 20
    
    # Capture images
    success = capture_student_images(student_name, num_images)
    
    if success:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Run: python deepface_recognition.py")
        print("2. Run: python deepface_webcam.py")
        print("="*60)

if __name__ == "__main__":
    main()