import cv2
import numpy as np
from deepface import DeepFace
import pickle
import os
from datetime import datetime
from collections import deque
import time
import pandas as pd
import json
with open("students.json", "r") as f:
    STUDENTS = json.load(f)
import json
from datetime import datetime
import os

ATTENDANCE_FILE = "attendance_history.json"

def save_attendance(name, reg_no, course):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    record = {
        "name": name,
        "reg_no": reg_no,
        "course": course,
        "time": time_now
    }

    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}

    if today not in data:
        data[today] = []

    # Zuia mtu kuandikishwa mara mbili siku moja
    for r in data[today]:
        if r["reg_no"] == reg_no:
            return

    data[today].append(record)

    with open(ATTENDANCE_FILE, "w") as f:
        json.dump(data, f, indent=4)


class AttendanceSystem:
    def __init__(self, model_path="model/deepface_recognition.pkl"):
        print("🚀 Initializing Attendance System...")
        
        # Load trained model
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Please run 'python deepface_recognition.py' first!")
            exit(1)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_name = model_data['model_name']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.known_embeddings = np.array(model_data['known_embeddings'])
        self.known_labels = np.array(model_data['known_labels'])
        
        # Load student database
        self.student_database = self.load_student_database()
        
        print(f"✓ Model loaded successfully!")
        print(f"✓ Using {self.model_name} model")
        print(f"✓ Tracking {len(self.label_encoder.classes_)} students:")
        print(f"  {', '.join(self.label_encoder.classes_)}")
    
    def load_student_database(self):
        """Load student details from database"""
        db_path = "students_database.json"
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                data = json.load(f)
                print(f"✓ Student database loaded")
                return data.get('students', {})
        else:
            print("⚠️  Student database not found, using basic info")
            return {}
    
    def get_student_info(self, student_name):
        """Get full student information"""
        if student_name in self.student_database:
            return self.student_database[student_name]
        else:
            # Return basic info if not in database
            return {
                'full_name': student_name,
                'registration_number': 'N/A',
                'course': 'N/A',
                'year': 'N/A',
                'email': 'N/A'
            }
        
        # Attendance tracking
        self.attendance_records = {}
        self.marked_today = set()
        self.last_seen = {}
        
        # Smoothing buffer
        self.prediction_buffer = {}
        self.buffer_size = 3
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create attendance folder
        os.makedirs("attendance_records", exist_ok=True)
        
        # Load today's attendance if exists
        self.load_todays_attendance()
    
    def load_todays_attendance(self):
        """Load attendance records for today"""
        today = datetime.now().strftime("%Y-%m-%d")
        attendance_file = f"attendance_records/attendance_{today}.json"
        
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                data = json.load(f)
                self.attendance_records = data.get('records', {})
                self.marked_today = set(data.get('marked', []))
            print(f"✓ Loaded today's attendance: {len(self.marked_today)} students marked")
        else:
            print("✓ Starting fresh attendance for today")
    
    def save_attendance(self):
        """Save attendance records"""
        today = datetime.now().strftime("%Y-%m-%d")
        attendance_file = f"attendance_records/attendance_{today}.json"
        
        data = {
            'date': today,
            'records': self.attendance_records,
            'marked': list(self.marked_today),
            'total_present': len(self.marked_today)
        }
        
        with open(attendance_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def mark_attendance(self, student_name, confidence):
        """Mark attendance for a student"""
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if already marked today
        if student_name in self.marked_today:
            # Update last seen time
            self.last_seen[student_name] = now
            return False  # Already marked
        
        # Get student info
        student_info = self.get_student_info(student_name)
        
        # Mark attendance
        self.attendance_records[student_name] = {
            'full_name': student_info['full_name'],
            'registration_number': student_info['registration_number'],
            'course': student_info['course'],
            'year': student_info['year'],
            'timestamp': timestamp,
            'confidence': f"{confidence*100:.1f}%",
            'status': 'Present'
        }
        
        self.marked_today.add(student_name)
        self.last_seen[student_name] = now
        
        # Save to file
        self.save_attendance()
        
        # Also append to CSV
        self.save_to_csv(student_name, timestamp, confidence, student_info)
        
        return True  # Newly marked
    
    def save_to_csv(self, student_name, timestamp, confidence, student_info):
        """Save attendance to CSV file"""
        today = datetime.now().strftime("%Y-%m-%d")
        csv_file = f"attendance_records/attendance_{today}.csv"
        
        # Create or append to CSV
        data = {
            'Student Name': [student_info['full_name']],
            'Registration Number': [student_info['registration_number']],
            'Course': [student_info['course']],
            'Year': [student_info['year']],
            'Date': [today],
            'Time': [timestamp.split()[1]],
            'Confidence': [f"{confidence*100:.1f}%"],
            'Status': ['Present']
        }
        
        df = pd.DataFrame(data)
        
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)
    
    def extract_embedding(self, frame, face_location):
        """Extract embedding from face region"""
        try:
            x, y, w, h = face_location
            
            # Extract face with padding
            padding_x = int(w * 0.3)
            padding_y = int(h * 0.3)
            
            y1 = max(0, y - padding_y)
            y2 = min(frame.shape[0], y + h + padding_y)
            x1 = max(0, x - padding_x)
            x2 = min(frame.shape[1], x + w + padding_x)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None
            
            # Resize and enhance
            face_img = cv2.resize(face_img, (160, 160))
            
            # Histogram equalization
            if len(face_img.shape) == 3:
                yuv = cv2.cvtColor(face_img, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                face_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            # Save temporarily
            temp_path = "temp_face_attendance.jpg"
            cv2.imwrite(temp_path, face_img)
            
            # Extract embedding
            embedding_objs = DeepFace.represent(
                img_path=temp_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='skip',
                align=True
            )
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if len(embedding_objs) > 0:
                return np.array(embedding_objs[0]['embedding'])
            
            return None
            
        except Exception as e:
            return None
    
    def recognize_face(self, face_embedding, confidence_threshold=0.35):
        """Recognize face from embedding"""
        if face_embedding is None:
            return "Unknown", 0.0
        
        face_embedding = face_embedding.reshape(1, -1)
        probabilities = self.classifier.predict_proba(face_embedding)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Check confidence gap
        top_2_indices = np.argsort(probabilities)[-2:][::-1]
        top_1_conf = probabilities[top_2_indices[0]]
        top_2_conf = probabilities[top_2_indices[1]] if len(top_2_indices) > 1 else 0
        confidence_gap = top_1_conf - top_2_conf
        
        if confidence < confidence_threshold or confidence_gap < 0.05:
            return "Unknown", confidence
        
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        return predicted_label, confidence
    
    def smooth_prediction(self, face_id, name, confidence):
        """Smooth predictions"""
        if face_id not in self.prediction_buffer:
            self.prediction_buffer[face_id] = deque(maxlen=self.buffer_size)
        
        self.prediction_buffer[face_id].append((name, confidence))
        
        names = [pred[0] for pred in self.prediction_buffer[face_id]]
        confidences = [pred[1] for pred in self.prediction_buffer[face_id]]
        
        from collections import Counter
        name_counts = Counter(names)
        smoothed_name = name_counts.most_common(1)[0][0]
        smoothed_confidence = np.mean(confidences)
        
        return smoothed_name, smoothed_confidence
    
    def generate_attendance_report(self):
        """Generate attendance report"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        print("\n" + "="*80)
        print(f"📊 ATTENDANCE REPORT - {today}")
        print("="*80)
        
        if not self.marked_today:
            print("No students marked present yet.")
        else:
            print(f"\n✅ Total Present: {len(self.marked_today)}/{len(self.label_encoder.classes_)}\n")
            
            # Header
            print(f"{'Name':<20} {'Reg No':<12} {'Course':<25} {'Time':<10} {'Confidence':<12}")
            print("-" * 80)
            
            for student in sorted(self.marked_today):
                record = self.attendance_records[student]
                full_name = record.get('full_name', student)
                reg_no = record.get('registration_number', 'N/A')
                course = record.get('course', 'N/A')
                time = record['timestamp'].split()[1]
                conf = record['confidence']
                
                print(f"{full_name:<20} {reg_no:<12} {course:<25} {time:<10} {conf:<12}")
            
            # Show absent students
            absent = set(self.label_encoder.classes_) - self.marked_today
            if absent:
                print(f"\n❌ Absent: {len(absent)}\n")
                for student in sorted(absent):
                    student_info = self.get_student_info(student)
                    print(f"  ✗ {student_info['full_name']:<20} ({student_info['registration_number']}) - {student_info['course']}")
        
        print("="*80)
    
    def run(self):
        """Run attendance system"""
        
        # Open webcam
        print("\n📹 Opening webcam...")
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("❌ Cannot open webcam!")
            return
        
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Webcam opened successfully!")
        print("\n" + "="*70)
        print("CONTROLS:")
        print("  'q' or 'ESC' - Quit and show report")
        print("  'r' - Show current report")
        print("  'SPACE' - Pause/Resume")
        print("  's' - Save screenshot")
        print("="*70 + "\n")
        
        process_this_frame = True
        frame_count = 0
        paused = False
        last_recognition_time = 0
        recognition_interval = 2.0
        last_results = {}
        
        # FPS tracking
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        while True:
            if not paused:
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
                )
                
                # Recognition
                current_time = time.time()
                should_recognize = (current_time - last_recognition_time) >= recognition_interval
                
                if should_recognize and len(faces) > 0:
                    last_recognition_time = current_time
                    
                    for i, (x, y, w, h) in enumerate(faces):
                        embedding = self.extract_embedding(frame, (x, y, w, h))
                        
                        if embedding is not None:
                            name, confidence = self.recognize_face(embedding)
                            name, confidence = self.smooth_prediction(i, name, confidence)
                            
                            last_results[i] = (name, confidence)
                            
                            # Mark attendance if recognized
                            if name != "Unknown" and confidence > 0.35:
                                newly_marked = self.mark_attendance(name, confidence)
                                if newly_marked:
                                    print(f"✅ ATTENDANCE MARKED: {name} ({confidence*100:.1f}%)")
                
                # Draw results
                for i, (x, y, w, h) in enumerate(faces):
                    if i in last_results:
                        name, confidence = last_results[i]
                    else:
                        name, confidence = "Processing...", 0.0
                    
                    # Check if marked
                    is_marked = name in self.marked_today
                    
                    # Get student info
                    if name != "Unknown" and name != "Processing...":
                        student_info = self.get_student_info(name)
                        display_name = student_info['full_name']
                        reg_no = student_info['registration_number']
                        course = student_info['course']
                    else:
                        display_name = name
                        reg_no = ""
                        course = ""
                    
                    # Color coding
                    if is_marked:
                        color = (0, 255, 0)  # Green - Marked
                        status = "✓ PRESENT"
                    elif name == "Unknown":
                        color = (0, 0, 255)  # Red - Unknown
                        status = ""
                    elif confidence > 0.4:
                        color = (0, 255, 255)  # Yellow - Recognized but not marked yet
                        status = "Marking..."
                    else:
                        color = (0, 165, 255)  # Orange - Low confidence
                        status = ""
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Draw info box
                    box_height = 80 if reg_no else 50
                    cv2.rectangle(frame, (x, y-box_height), (x+w, y), color, -1)
                    
                    # Draw text
                    y_offset = y - box_height + 18
                    cv2.putText(frame, display_name, (x+5, y_offset), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    
                    if reg_no:
                        y_offset += 20
                        cv2.putText(frame, f"Reg: {reg_no}", (x+5, y_offset), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
                        y_offset += 18
                        cv2.putText(frame, course[:20], (x+5, y_offset), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
                    
                    y_offset += 18
                    cv2.putText(frame, f"{confidence*100:.1f}% {status}", (x+5, y_offset), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
                
                # FPS calculation
                fps_frame_count += 1
                if fps_frame_count >= 30:
                    fps_end_time = time.time()
                    current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    fps_frame_count = 0
                
                # Info overlay
                cv2.rectangle(frame, (0, 0), (640, 120), (0, 0, 0), -1)
                
                today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"Date: {today}", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(frame, f"Present: {len(self.marked_today)}/{len(self.label_encoder.classes_)}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(frame, f"Faces: {len(faces)} | FPS: {current_fps:.1f}", 
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Next scan timer
                time_until_next = max(0, recognition_interval - (current_time - last_recognition_time))
                cv2.putText(frame, f"Next scan: {time_until_next:.1f}s", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Controls
                cv2.putText(frame, "Q:Quit | R:Report | SPACE:Pause | S:Save", 
                           (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                frame_count += 1
            
            else:
                cv2.putText(frame, "PAUSED - Press SPACE to resume", 
                           (frame.shape[1]//2 - 180, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow('Attendance System - Face Recognition', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                self.generate_attendance_report()
            elif key == ord(' '):
                paused = not paused
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"attendance_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        # Cleanup
        video_capture.release()
        cv2.destroyAllWindows()
        
        # Final report
        self.generate_attendance_report()
        
        print(f"\n📁 Attendance saved to: attendance_records/")
        print(f"📊 Total frames processed: {frame_count}")

def main():
    print("🎯 Attendance System with Face Recognition")
    print("="*70)
    print("Automatic attendance marking for recognized students")
    print("="*70)
    
    system = AttendanceSystem()
    system.run()

if __name__ == "__main__":
    main()

    