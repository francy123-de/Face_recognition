from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import pickle
import os
from datetime import datetime
import json
import pandas as pd
import threading
import queue
import time

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model/deepface_recognition.pkl"

# Global variables for threaded recognition
recognition_queue = queue.Queue(maxsize=2)  # Keep 2 frames for smoother processing
recognition_results = {}
recognition_lock = threading.Lock()
camera_active = True

class WebAttendanceSystem:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first!")
        
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_name = model_data['model_name']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.known_embeddings = np.array(model_data['known_embeddings'])
        self.known_labels = np.array(model_data['known_labels'])
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load student database
        self.student_database = self.load_student_database()
        
        # Attendance tracking
        self.attendance_records = {}
        self.marked_today = set()
        self.load_todays_attendance()
        
        print(f"✓ Model loaded: {self.model_name}")
        print(f"✓ Student database loaded: {len(self.student_database)} students")
        print(f"✓ Tracking {len(self.label_encoder.classes_)} students")
    
    def load_student_database(self):
        """Load student details from database"""
        db_path = "students_database.json"
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                data = json.load(f)
                return data.get('students', {})
        else:
            print("⚠️  Student database not found, using basic info")
            return {}
    
    def get_student_info(self, student_name):
        """Get full student information"""
        if student_name in self.student_database:
            return self.student_database[student_name]
        else:
            return {
                'full_name': student_name,
                'registration_number': 'N/A',
                'course': 'N/A',
                'year': 'N/A',
                'email': 'N/A'
            }
    
    def load_todays_attendance(self):
        """Load today's attendance"""
        today = datetime.now().strftime("%Y-%m-%d")
        attendance_file = f"attendance_records/attendance_{today}.json"
        
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                data = json.load(f)
                self.attendance_records = data.get('records', {})
                self.marked_today = set(data.get('marked', []))
    
    def save_attendance(self):
        """Save attendance"""
        today = datetime.now().strftime("%Y-%m-%d")
        os.makedirs("attendance_records", exist_ok=True)
        attendance_file = f"attendance_records/attendance_{today}.json"
        
        data = {
            'date': today,
            'records': self.attendance_records,
            'marked': list(self.marked_today),
            'total_present': len(self.marked_today)
        }
        
        with open(attendance_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save to CSV with full info
        csv_file = f"attendance_records/attendance_{today}.csv"
        if self.attendance_records:
            df_data = []
            for student, record in self.attendance_records.items():
                df_data.append({
                    'Full Name': record.get('full_name', student),
                    'Registration Number': record.get('registration_number', 'N/A'),
                    'Course': record.get('course', 'N/A'),
                    'Year': record.get('year', 'N/A'),
                    'Date': today,
                    'Time': record['timestamp'].split()[1],
                    'Confidence': record['confidence'],
                    'Status': record['status']
                })
            df = pd.DataFrame(df_data)
            df.to_csv(csv_file, index=False)
    
    def mark_attendance(self, student_name, confidence):
        """Mark attendance with full student info"""
        if student_name in self.marked_today:
            return False
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        student_info = self.get_student_info(student_name)
        
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
        self.save_attendance()
        
        return True
    
    def extract_embedding(self, face_img):
        """Extract face embedding - HIGH ACCURACY OPTIMIZED"""
        try:
            # Input validation
            if face_img is None or face_img.size == 0:
                return None
            
            # Ensure minimum face size for good recognition
            if face_img.shape[0] < 60 or face_img.shape[1] < 60:
                print("⚠️  Face too small for accurate recognition")
                return None
        
            
            # Resize to optimal size for DeepFace (160x160 for best accuracy)
            target_size = (160, 160)  # Standard size for Facenet512
            face_resized = cv2.resize(face_img, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Advanced image enhancement for better recognition
            # Step 1: Convert to LAB color space
            lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Step 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Step 3: Merge and convert back
            enhanced = cv2.merge([l, a, b])
            face_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Step 4: Denoise for cleaner image
            face_enhanced = cv2.fastNlMeansDenoisingColored(face_enhanced, None, 10, 10, 7, 21)
            
            # Step 5: Sharpen for better features
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            face_enhanced = cv2.filter2D(face_enhanced, -1, kernel)
            
            # Create unique temporary file
            temp_path = f"temp_recognition_{os.getpid()}_{int(time.time()*1000000)}.jpg"
            
            # Save with maximum quality
            cv2.imwrite(temp_path, face_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            # Extract embedding using DeepFace with optimal settings
            embedding_objs = DeepFace.represent(
                img_path=temp_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='skip',
                align=True,
                normalization='Facenet2018'  # Better normalization for Facenet512
            )
            
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
            
            # Return embedding if successful
            if len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]['embedding'])
                print(f"✅ Embedding extracted successfully (dim: {len(embedding)})")
                return embedding
            
            return None
            
        except Exception as e:
            # Clean up on any error
            temp_path = f"temp_recognition_{os.getpid()}_{int(time.time()*1000000)}.jpg"
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
            
            print(f"❌ Embedding extraction error: {e}")
            return None
    
    def recognize_face(self, face_embedding, confidence_threshold=0.25):
        """Professional face recognition - OPTIMIZED FOR TRAINED PEOPLE"""
        if face_embedding is None:
            return "Unknown", 0.0
        
        try:
            # Reshape for classifier
            face_embedding = face_embedding.reshape(1, -1)
            
            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(face_embedding)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Get top 2 predictions for comparison
            sorted_indices = np.argsort(probabilities)[::-1]
            top_1_conf = probabilities[sorted_indices[0]]
            top_2_conf = probabilities[sorted_indices[1]] if len(sorted_indices) > 1 else 0
            confidence_gap = top_1_conf - top_2_conf
            
            # Print debug info
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            print(f"🔍 Recognition: {predicted_label} | Confidence: {confidence*100:.1f}% | Gap: {confidence_gap*100:.1f}%")
            
            # RELAXED THRESHOLDS FOR BETTER RECOGNITION
            # Strategy: Accept if confidence is reasonable OR gap is good
            
            if confidence > 0.7:
                # Very high confidence - definitely accept
                print(f"✅ HIGH CONFIDENCE: {predicted_label}")
                return predicted_label, confidence
            
            elif confidence > 0.5 and confidence_gap > 0.1:
                # Good confidence with decent gap
                print(f"✅ GOOD MATCH: {predicted_label}")
                return predicted_label, confidence
            
            elif confidence > 0.35 and confidence_gap > 0.15:
                # Moderate confidence but strong gap (clearly best match)
                print(f"✅ CLEAR WINNER: {predicted_label}")
                return predicted_label, confidence
            
            elif confidence > confidence_threshold and confidence_gap > 0.08:
                # Low confidence but still has a gap (better than nothing)
                print(f"⚠️  ACCEPTABLE: {predicted_label}")
                return predicted_label, confidence
            
            else:
                # Too uncertain
                print(f"❌ REJECTED: {predicted_label} (conf: {confidence*100:.1f}%, gap: {confidence_gap*100:.1f}%)")
                return "Unknown", confidence
                
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            import traceback
            traceback.print_exc()
            return "Unknown", 0.0
    
    def get_attendance_summary(self):
        """Get attendance summary with full student details"""
        total_students = len(self.label_encoder.classes_)
        present = len(self.marked_today)
        absent = total_students - present
        
        # Get full info for present students
        present_list_with_info = []
        for name in sorted(list(self.marked_today)):
            student_info = self.get_student_info(name)
            present_list_with_info.append({
                'folder_name': name,
                'full_name': student_info['full_name'],
                'registration_number': student_info['registration_number'],
                'course': student_info['course']
            })
        
        # Get full info for absent students
        absent_list_with_info = []
        for name in sorted(list(set(self.label_encoder.classes_) - self.marked_today)):
            student_info = self.get_student_info(name)
            absent_list_with_info.append({
                'folder_name': name,
                'full_name': student_info['full_name'],
                'registration_number': student_info['registration_number'],
                'course': student_info['course']
            })
        
        return {
            'total': total_students,
            'present': present,
            'absent': absent,
            'present_list': [s['folder_name'] for s in present_list_with_info],
            'present_list_full': present_list_with_info,
            'absent_list': [s['folder_name'] for s in absent_list_with_info],
            'absent_list_full': absent_list_with_info
        }

# Initialize system
try:
    attendance_system = WebAttendanceSystem()
except Exception as e:
    print(f"Error initializing system: {e}")
    attendance_system = None

# Background recognition thread
def recognition_worker():
    """Professional background thread for face recognition - NO BLOCKING!"""
    global recognition_results
    print("🔄 Recognition worker started - Professional mode")
    
    while True:
        try:
            # Get frame from queue with timeout
            frame_data = recognition_queue.get(timeout=2)
            
            if frame_data is None:  # Shutdown signal
                print("🛑 Recognition worker stopping...")
                break
            
            face_crop, frame_id = frame_data
            
            # Validate face crop
            if face_crop is None or face_crop.size == 0:
                continue
            
            print(f"🔍 Processing face recognition for frame {frame_id}")
            
            # Extract embedding with error handling
            try:
                embedding = attendance_system.extract_embedding(face_crop)
                if embedding is not None:
                    name, confidence = attendance_system.recognize_face(embedding)
                    
                    # Store result with thread safety
                    with recognition_lock:
                        recognition_results[frame_id] = {
                            'name': name,
                            'confidence': confidence,
                            'timestamp': time.time()
                        }
                    
                    # Auto-mark attendance for recognized faces
                    if name != "Unknown" and confidence > 0.25:
                        success = attendance_system.mark_attendance(name, confidence)
                        if success:
                            print(f"✅ ATTENDANCE MARKED: {name} ({confidence*100:.1f}%)")
                        else:
                            print(f"ℹ️  Already present: {name}")
                    elif name != "Unknown":
                        print(f"⚠️  Recognized but low confidence: {name} ({confidence*100:.1f}%)")
                    else:
                        print(f"❓ Unknown person detected")
                else:
                    print("❌ Failed to extract face embedding")
                    
            except Exception as e:
                print(f"❌ Recognition error: {e}")
                # Continue processing other frames
                continue
                
        except queue.Empty:
            # No frames to process, continue waiting
            continue
        except Exception as e:
            print(f"❌ Worker thread error: {e}")
            continue

# Start recognition thread
if attendance_system is not None:
    recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
    recognition_thread.start()

def generate_frames():
    """Generate video frames with smooth threaded recognition - NO FREEZING!"""
    if attendance_system is None:
        return
    
    global recognition_results, camera_active
    
    # Professional camera setup with DirectShow backend for Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow for better Windows performance
    if not cap.isOpened():
        print("❌ Cannot open camera with DirectShow, trying default...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
    
    # Optimize camera settings for performance and stability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to prevent lag
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPEG for better performance
    
    # Warm up camera (critical for stability!)
    print("🔥 Warming up camera...")
    for i in range(15):
        ret, _ = cap.read()
        if not ret:
            print(f"⚠️  Warmup frame {i+1} failed")
        time.sleep(0.05)
    
    frame_count = 0
    last_recognition_time = 0
    current_result = {'name': 'Scanning...', 'confidence': 0.0}
    recognition_cooldown = 1.5  # Recognition every 1.5 seconds (faster response)
    
    print("🎥 Camera initialized! Starting smooth video stream...")
    
    last_frame_time = time.time()
    fps_counter = 0
    fps_display = 0
    
    try:
        while camera_active:
            # Read frame with timeout protection
            success, frame = cap.read()
            if not success:
                print("⚠️  Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Calculate FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - last_frame_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                last_frame_time = current_time
            
            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)
            
            # FAST face detection (optimized for speed) - only every 3rd frame
            faces = []
            if frame_count % 3 == 0:  # Detect faces every 3 frames for better performance
                small_frame = cv2.resize(frame, (320, 240))  # Smaller for speed
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization for better detection
                gray = cv2.equalizeHist(gray)
                
                detected_faces = attendance_system.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.15,  # Better accuracy
                    minNeighbors=4,    # Balanced
                    minSize=(30, 30),  # Reasonable minimum
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Scale faces back to original frame size
                faces = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in detected_faces]
            
            # Keep last detected faces for smooth display
            if not hasattr(generate_frames, 'last_faces'):
                generate_frames.last_faces = []
            if len(faces) > 0:
                generate_frames.last_faces = faces
            else:
                faces = generate_frames.last_faces
            
            # Send face for recognition (NON-BLOCKING)
            if len(faces) > 0 and (current_time - last_recognition_time) >= recognition_cooldown:
                # Get the largest, most centered face
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2
                
                def face_score(face):
                    x, y, w, h = face
                    face_center_x = x + w/2
                    face_center_y = y + h/2
                    distance = ((face_center_x - frame_center_x)**2 + (face_center_y - frame_center_y)**2)**0.5
                    size = w * h
                    return size / (distance + 1)  # Prefer larger faces closer to center
                
                best_face = max(faces, key=face_score)
                x, y, w, h = best_face
                
                # Extract face with padding
                padding = max(15, int(w * 0.2))
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                
                face_crop = frame[y1:y2, x1:x2].copy()
                
                # Validate face crop
                if face_crop.size > 0 and face_crop.shape[0] > 50 and face_crop.shape[1] > 50:
                    # Send to background thread (non-blocking!)
                    try:
                        # Clear old items if queue is full
                        while recognition_queue.full():
                            try:
                                recognition_queue.get_nowait()
                            except:
                                break
                        
                        recognition_queue.put_nowait((face_crop, frame_count))
                        last_recognition_time = current_time
                        print(f"📸 Face sent for recognition (Frame {frame_count})")
                    except Exception as e:
                        print(f"⚠️  Queue error: {e}")
            
            # Get latest recognition result (non-blocking!)
            try:
                with recognition_lock:
                    if recognition_results:
                        # Get most recent result
                        latest_frame = max(recognition_results.keys())
                        result_age = frame_count - latest_frame
                        
                        # Use result if it's recent (within 10 seconds)
                        if result_age <= 300:  # 300 frames = ~10 seconds at 30fps
                            current_result = recognition_results[latest_frame]
                        
                        # Clean old results (keep only last 5)
                        if len(recognition_results) > 5:
                            sorted_frames = sorted(recognition_results.keys())
                            for old_frame in sorted_frames[:-5]:
                                del recognition_results[old_frame]
            except Exception as e:
                print(f"⚠️  Result retrieval error: {e}")
            
            # DRAW EVERYTHING (this is fast and smooth)
            for (x, y, w, h) in faces:
                name = current_result['name']
                confidence = current_result['confidence']
                
                # Get student information
                if name not in ["Unknown", "Scanning..."]:
                    student_info = attendance_system.get_student_info(name)
                    display_name = student_info['full_name']
                    reg_no = student_info['registration_number']
                    is_marked = name in attendance_system.marked_today
                else:
                    display_name = name
                    reg_no = ""
                    is_marked = False
                
                # Professional color coding
                if is_marked:
                    color = (0, 200, 0)      # Green - Already marked
                    status_color = (0, 255, 0)
                elif name not in ["Unknown", "Scanning..."]:
                    color = (0, 255, 255)    # Yellow - Recognized
                    status_color = (255, 255, 0)
                else:
                    color = (255, 255, 255)  # White - Detecting
                    status_color = (255, 255, 255)
                
                # Draw face rectangle with thickness based on confidence
                thickness = 4 if confidence > 0.5 else 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                
                # Draw information box
                if name not in ["Unknown", "Scanning..."]:
                    # Calculate box size
                    box_height = 80 if reg_no and reg_no != "N/A" else 50
                    box_y = max(0, y - box_height)
                    
                    # Draw semi-transparent background
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, box_y), (x+w, y), color, -1)
                    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                    
                    # Draw text with better positioning
                    text_y = box_y + 20
                    cv2.putText(frame, display_name[:18], (x+8, text_y), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(frame, display_name[:18], (x+8, text_y), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    
                    if reg_no and reg_no != "N/A":
                        text_y += 25
                        cv2.putText(frame, f"ID: {reg_no}", (x+8, text_y), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"ID: {reg_no}", (x+8, text_y), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                    
                    text_y += 25
                    status_text = "✓ PRESENT" if is_marked else f"Conf: {confidence*100:.0f}%"
                    cv2.putText(frame, status_text, (x+8, text_y), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, status_text, (x+8, text_y), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, status_color, 1)
                else:
                    # Simple label for unknown/scanning
                    cv2.putText(frame, display_name, (x+5, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                    cv2.putText(frame, display_name, (x+5, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Professional status overlay
            summary = attendance_system.get_attendance_summary()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (450, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Status text
            cv2.putText(frame, f"Attendance: {summary['present']}/{summary['total']}", 
                       (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), 
                       (15, 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # FPS and recognition status
            cv2.putText(frame, f"FPS: {fps_display}", 
                       (15, 85), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
            
            if len(faces) > 0:
                status_text = f"{len(faces)} face(s)"
                cv2.putText(frame, status_text, (250, 60), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
            
            frame_count += 1
            
            # Encode frame for streaming (optimized quality/speed balance)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Slightly lower for better performance
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if not ret:
                print("⚠️  Frame encoding failed")
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield frame with proper headers
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + 
                   frame_bytes + b'\r\n')
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.001)
    
    except GeneratorExit:
        print("🛑 Client disconnected")
    except Exception as e:
        print(f"❌ Camera error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera_active = False
        cap.release()
        cv2.destroyAllWindows()
        print("📹 Camera released and cleaned up")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance_summary')
def attendance_summary():
    """Get attendance summary as JSON"""
    if attendance_system is None:
        return jsonify({'error': 'System not initialized'})
    
    summary = attendance_system.get_attendance_summary()
    summary['records'] = attendance_system.attendance_records
    
    return jsonify(summary)

@app.route('/download_csv')
def download_csv():
    """Download today's attendance CSV"""
    today = datetime.now().strftime("%Y-%m-%d")
    csv_file = f"attendance_records/attendance_{today}.csv"
    
    if os.path.exists(csv_file):
        from flask import send_file
        return send_file(csv_file, as_attachment=True)
    else:
        return "No attendance records for today", 404

@app.route('/attendance_history')
def attendance_history():
    """Get list of all attendance dates"""
    if not os.path.exists("attendance_records"):
        return jsonify({'dates': []})
    
    files = os.listdir("attendance_records")
    json_files = [f for f in files if f.endswith('.json')]
    
    dates = []
    for file in json_files:
        # Extract date from filename: attendance_YYYY-MM-DD.json
        date_str = file.replace('attendance_', '').replace('.json', '')
        
        # Load file to get summary
        file_path = os.path.join("attendance_records", file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                dates.append({
                    'date': date_str,
                    'total_present': data.get('total_present', 0),
                    'records': data.get('records', {})
                })
        except:
            pass
    
    # Sort by date (newest first)
    dates.sort(key=lambda x: x['date'], reverse=True)
    
    return jsonify({'dates': dates})

@app.route('/attendance_by_date/<date>')
def attendance_by_date(date):
    """Get attendance for specific date"""
    file_path = f"attendance_records/attendance_{date}.json"
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'No records for this date'}), 404
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Add student full info
        if attendance_system:
            for student_name, record in data.get('records', {}).items():
                student_info = attendance_system.get_student_info(student_name)
                record['full_name'] = student_info['full_name']
                record['registration_number'] = student_info['registration_number']
                record['course'] = student_info['course']
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/student_list')
def student_list():
    """Get list of all students"""
    if attendance_system is None:
        return jsonify({'error': 'System not initialized'})
    
    students = []
    for folder_name in sorted(attendance_system.label_encoder.classes_):
        student_info = attendance_system.get_student_info(folder_name)
        students.append({
            'folder_name': folder_name,
            'full_name': student_info['full_name'],
            'registration_number': student_info['registration_number'],
            'course': student_info['course'],
            'year': student_info['year']
        })
    
    return jsonify({'students': students})

@app.route('/student_history/<student_name>')
def student_history(student_name):
    """Get attendance history for a specific student"""
    if not os.path.exists("attendance_records"):
        return jsonify({'error': 'No attendance records found'}), 404
    
    # Get student info
    student_info = attendance_system.get_student_info(student_name) if attendance_system else {}
    
    # Get all attendance files
    files = os.listdir("attendance_records")
    json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
    
    history = []
    total_days = len(json_files)
    present_days = 0
    absent_days = 0
    
    for file in json_files:
        date_str = file.replace('attendance_', '').replace('.json', '')
        file_path = os.path.join("attendance_records", file)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                records = data.get('records', {})
                
                if student_name in records:
                    # Student was present
                    record = records[student_name]
                    history.append({
                        'date': date_str,
                        'status': 'Present',
                        'time': record.get('timestamp', 'N/A').split()[1] if 'timestamp' in record else 'N/A',
                        'confidence': record.get('confidence', 'N/A')
                    })
                    present_days += 1
                else:
                    # Student was absent
                    history.append({
                        'date': date_str,
                        'status': 'Absent',
                        'time': '-',
                        'confidence': '-'
                    })
                    absent_days += 1
        except:
            pass
    
    # Calculate attendance percentage
    attendance_percentage = (present_days / total_days * 100) if total_days > 0 else 0
    
    return jsonify({
        'student_name': student_name,
        'full_name': student_info.get('full_name', student_name),
        'registration_number': student_info.get('registration_number', 'N/A'),
        'course': student_info.get('course', 'N/A'),
        'year': student_info.get('year', 'N/A'),
        'total_days': total_days,
        'present_days': present_days,
        'absent_days': absent_days,
        'attendance_percentage': round(attendance_percentage, 1),
        'history': history
    })

if __name__ == '__main__':
    if attendance_system is None:
        print("❌ Failed to initialize attendance system!")
        print("Please run 'python deepface_recognition.py' first to train the model.")
    else:
        print("\n🚀 Starting Web Attendance System...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("Press Ctrl+C to stop\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
