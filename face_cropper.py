import cv2
import os
import numpy as np
from PIL import Image

class FaceCropper:
    def __init__(self):
        # Load OpenCV's pre-trained face detection classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✓ Face detection classifier loaded")
    
    def detect_and_crop_face(self, image_path, output_size=(160, 160), margin=0.2):
        """Detect and crop face from image"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None, "Cannot read image"
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None, "No face detected"
            
            # Use the largest face if multiple faces detected
            if len(faces) > 1:
                # Sort by area (width * height) and take the largest
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            
            x, y, w, h = faces[0]
            
            # Add margin around face
            margin_x = int(w * margin)
            margin_y = int(h * margin)
            
            # Calculate crop coordinates with margin
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img.shape[1], x + w + margin_x)
            y2 = min(img.shape[0], y + h + margin_y)
            
            # Crop face
            face_crop = img[y1:y2, x1:x2]
            
            # Resize to output size
            face_resized = cv2.resize(face_crop, output_size)
            
            return face_resized, "Success"
            
        except Exception as e:
            return None, str(e)
    
    def process_dataset(self, input_dir="dataset", output_dir="dataset_cropped"):
        """Process entire dataset and crop faces"""
        
        if not os.path.exists(input_dir):
            print(f"Input directory '{input_dir}' not found!")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔍 Processing dataset from '{input_dir}' to '{output_dir}'...")
        
        # Get all student folders
        student_folders = [f for f in os.listdir(input_dir) 
                          if os.path.isdir(os.path.join(input_dir, f)) and not f.startswith('_')]
        
        total_processed = 0
        total_success = 0
        total_failed = 0
        
        for student_name in student_folders:
            print(f"\n📁 Processing {student_name}...")
            
            input_student_path = os.path.join(input_dir, student_name)
            output_student_path = os.path.join(output_dir, student_name)
            
            # Create output student directory
            os.makedirs(output_student_path, exist_ok=True)
            
            # Get all image files
            image_files = [f for f in os.listdir(input_student_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            student_success = 0
            student_failed = 0
            
            for image_file in image_files:
                input_path = os.path.join(input_student_path, image_file)
                output_path = os.path.join(output_student_path, image_file)
                
                # Crop face
                cropped_face, message = self.detect_and_crop_face(input_path)
                
                if cropped_face is not None:
                    # Save cropped face
                    cv2.imwrite(output_path, cropped_face)
                    print(f"  ✓ {image_file}")
                    student_success += 1
                    total_success += 1
                else:
                    print(f"  ✗ {image_file} - {message}")
                    student_failed += 1
                    total_failed += 1
                
                total_processed += 1
            
            print(f"  📊 {student_name}: {student_success}/{len(image_files)} successful")
        
        # Summary
        print(f"\n📈 FACE CROPPING SUMMARY:")
        print(f"Total images processed: {total_processed}")
        print(f"Successfully cropped: {total_success}")
        print(f"Failed to crop: {total_failed}")
        print(f"Success rate: {(total_success/total_processed*100):.1f}%" if total_processed > 0 else "No images processed")
        
        if total_success > 0:
            print(f"\n✅ Cropped faces saved to '{output_dir}'")
        
        return total_success, total_failed

def main():
    print("✂️  Face Cropping Script")
    print("=" * 50)
    
    # Initialize face cropper
    cropper = FaceCropper()
    
    # Process dataset
    success_count, failed_count = cropper.process_dataset()
    
    if success_count > 0:
        print(f"\n🎉 Face cropping completed!")
        print(f"Cropped {success_count} faces successfully.")
        print("\nThe cropped faces are now ready for better face recognition training.")
    else:
        print(f"\n❌ No faces were successfully cropped.")
        print("Please check your images and ensure they contain clear faces.")

if __name__ == "__main__":
    main()