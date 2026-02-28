import os
import cv2
import numpy as np
from PIL import Image
import shutil

def check_image_validity(image_path):
    """Check if image file is valid and readable"""
    try:
        # Try to open with PIL
        with Image.open(image_path) as img:
            img.verify()
        
        # Try to read with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return False, "Cannot read with OpenCV"
        
        # Check if image has reasonable dimensions
        height, width = img.shape[:2]
        if height < 50 or width < 50:
            return False, f"Image too small: {width}x{height}"
        
        return True, "Valid"
    
    except Exception as e:
        return False, str(e)

def clean_dataset(dataset_path="dataset"):
    """Clean the dataset by removing invalid images and organizing structure"""
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' not found!")
        return
    
    print(f"🧹 Cleaning dataset in '{dataset_path}'...")
    
    total_images = 0
    valid_images = 0
    invalid_images = 0
    removed_files = []
    
    # Get all student folders
    student_folders = [f for f in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, f))]
    
    print(f"Found {len(student_folders)} student folders: {student_folders}")
    
    for student_name in student_folders:
        student_path = os.path.join(dataset_path, student_name)
        print(f"\n📁 Processing {student_name}...")
        
        # Get all image files in student folder
        image_files = [f for f in os.listdir(student_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        student_total = len(image_files)
        student_valid = 0
        
        for image_file in image_files:
            image_path = os.path.join(student_path, image_file)
            total_images += 1
            
            # Check if image is valid
            is_valid, message = check_image_validity(image_path)
            
            if is_valid:
                valid_images += 1
                student_valid += 1
                print(f"  ✓ {image_file}")
            else:
                invalid_images += 1
                removed_files.append(image_path)
                print(f"  ✗ {image_file} - {message}")
                
                # Move invalid file to a backup folder
                backup_folder = os.path.join(dataset_path, "_invalid_images")
                os.makedirs(backup_folder, exist_ok=True)
                
                backup_path = os.path.join(backup_folder, f"{student_name}_{image_file}")
                try:
                    shutil.move(image_path, backup_path)
                    print(f"    → Moved to {backup_path}")
                except Exception as e:
                    print(f"    → Error moving file: {e}")
        
        print(f"  📊 {student_name}: {student_valid}/{student_total} valid images")
    
    # Summary
    print(f"\n📈 CLEANING SUMMARY:")
    print(f"Total images processed: {total_images}")
    print(f"Valid images: {valid_images}")
    print(f"Invalid images: {invalid_images}")
    print(f"Success rate: {(valid_images/total_images*100):.1f}%" if total_images > 0 else "No images found")
    
    if removed_files:
        print(f"\n🗑️  Invalid files moved to '{dataset_path}/_invalid_images/':")
        for file in removed_files:
            print(f"  - {file}")
    
    return valid_images, invalid_images

def resize_images(dataset_path="dataset", target_size=(160, 160)):
    """Resize all images to target size for consistency"""
    print(f"\n🔄 Resizing images to {target_size}...")
    
    student_folders = [f for f in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, f)) and not f.startswith('_')]
    
    resized_count = 0
    
    for student_name in student_folders:
        student_path = os.path.join(dataset_path, student_name)
        image_files = [f for f in os.listdir(student_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for image_file in image_files:
            image_path = os.path.join(student_path, image_file)
            
            try:
                # Load and resize image
                img = Image.open(image_path)
                if img.size != target_size:
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    img_resized.save(image_path)
                    resized_count += 1
                    print(f"  ✓ Resized {student_name}/{image_file}")
            
            except Exception as e:
                print(f"  ✗ Error resizing {student_name}/{image_file}: {e}")
    
    print(f"📏 Resized {resized_count} images")

def main():
    print("🚀 Starting dataset cleaning process...")
    
    # Clean invalid images
    valid_count, invalid_count = clean_dataset()
    
    if valid_count > 0:
        # Resize images for consistency
        resize_images()
        
        print(f"\n✅ Dataset cleaning completed!")
        print(f"Your dataset is now clean and ready for training.")
    else:
        print(f"\n❌ No valid images found in dataset!")

if __name__ == "__main__":
    main()