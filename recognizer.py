# server/recognizer.py
import cv2
import face_recognition
import numpy as np
import threading
import time
from datetime import datetime
from .db import init_db, load_all_students_embeddings, add_attendance
from .config import CAMERA_SOURCE, SIMILARITY_THRESHOLD, FRAMES_REQUIRED, ADMIN_PASSPHRASE
from .anti_spoof import is_live

class RecognizerThread(threading.Thread):
    def __init__(self, session_id: str, mqtt_publish=None):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.mqtt_publish = mqtt_publish or (lambda x: None)
        self.running = False
        self.state = "STOPPED"
        self.last_seen = {}  # roll -> timestamp
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.known_rolls = []
        self.frame_count = 0
        self.recognition_buffer = {}  # roll -> consecutive frames count
        self.occupancy_state = "EMPTY"
        self.last_motion_time = 0
        self.cap = None
        
    def load_students(self):
        """Load student data from database"""
        if not ADMIN_PASSPHRASE:
            print("⚠️  No admin passphrase set. Cannot load encrypted embeddings.")
            return False
            
        data = load_all_students_embeddings(ADMIN_PASSPHRASE)
        if not data:
            print("⚠️  No enrolled students found in database.")
            return False
            
        self.known_encodings = [d["embedding"] for d in data]
        self.known_names = [d["name"] for d in data]
        self.known_ids = [d["id"] for d in data]
        self.known_rolls = [d["roll"] for d in data]
        
        print(f"✅ Loaded {len(self.known_encodings)} enrolled students.")
        return True
    
    def start_camera(self):
        """Initialize camera"""
        try:
            cam_source = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
        except:
            cam_source = 0
            
        self.cap = cv2.VideoCapture(cam_source)
        if not self.cap.isOpened():
            print("❌ Could not open camera.")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        print("📷 Camera initialized successfully.")
        return True
    
    def detect_motion(self, frame):
        """Simple motion detection to determine occupancy"""
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize background if not exists
        if not hasattr(self, 'background'):
            self.background = gray
            return False
        
        # Compute difference and threshold
        diff = cv2.absdiff(self.background, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours and check for significant motion
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area threshold
                motion_detected = True
                break
        
        # Update background slowly
        self.background = cv2.addWeighted(self.background, 0.95, gray, 0.05, 0)
        
        if motion_detected:
            self.last_motion_time = time.time()
        
        return motion_detected
    
    def update_occupancy_state(self):
        """Update room occupancy state based on recent activity"""
        current_time = time.time()
        
        # Consider room occupied if:
        # 1. Face was recognized recently (within 30 seconds)
        # 2. Motion was detected recently (within 15 seconds)
        
        recent_face = any(current_time - ts < 30 for ts in self.last_seen.values())
        recent_motion = current_time - self.last_motion_time < 15
        
        old_state = self.occupancy_state
        
        if recent_face or recent_motion:
            self.occupancy_state = "OCCUPIED"
        else:
            self.occupancy_state = "EMPTY"
        
        # Send command to control devices if state changed
        if old_state != self.occupancy_state:
            print(f"🏠 Occupancy changed: {old_state} -> {self.occupancy_state}")
            
            if self.occupancy_state == "OCCUPIED":
                self.mqtt_publish({"mode": "AUTO", "lights": "ON", "fans": "ON"})
            else:
                self.mqtt_publish({"mode": "AUTO", "lights": "OFF", "fans": "OFF"})
    
    def process_face_recognition(self, frame):
        """Process frame for face recognition"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]  # BGR to RGB
        
        # Find faces and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        current_time = time.time()
        recognized_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            # Anti-spoofing check
            face_crop = frame[top:bottom, left:right]
            if face_crop.size > 0:
                live, score = is_live(face_crop)
                if not live:
                    recognized_faces.append({
                        'name': 'SPOOF DETECTED',
                        'location': (top, right, bottom, left),
                        'color': (0, 0, 255),  # Red
                        'confidence': 0.0
                    })
                    continue
            
            # Find best match
            if len(self.known_encodings) > 0:
                distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]
                
                if best_distance <= SIMILARITY_THRESHOLD:
                    name = self.known_names[best_match_index]
                    roll = self.known_rolls[best_match_index]
                    student_id = self.known_ids[best_match_index]
                    confidence = max(0.0, 1.0 - best_distance)
                    
                    # Use frame counting for stability
                    if roll not in self.recognition_buffer:
                        self.recognition_buffer[roll] = 0
                    
                    self.recognition_buffer[roll] += 1
                    
                    # Require multiple consecutive frames for recognition
                    if self.recognition_buffer[roll] >= FRAMES_REQUIRED:
                        # Check if we should log attendance (not seen recently)
                        if roll not in self.last_seen or current_time - self.last_seen[roll] > 30:
                            try:
                                add_attendance(
                                    student_id=student_id,
                                    roll=roll,
                                    name=name,
                                    confidence=confidence,
                                    session_id=self.session_id
                                )
                                print(f"✅ Attendance logged: {name} ({roll}) - Confidence: {confidence:.2f}")
                            except Exception as e:
                                print(f"❌ Failed to log attendance: {e}")
                        
                        self.last_seen[roll] = current_time
                        self.recognition_buffer[roll] = 0  # Reset buffer
                        
                        recognized_faces.append({
                            'name': f"{name} ({roll})",
                            'location': (top, right, bottom, left),
                            'color': (0, 255, 0),  # Green
                            'confidence': confidence
                        })
                    else:
                        # Still recognizing, show in progress
                        progress = self.recognition_buffer[roll] / FRAMES_REQUIRED
                        recognized_faces.append({
                            'name': f"Recognizing... {progress:.0%}",
                            'location': (top, right, bottom, left),
                            'color': (0, 255, 255),  # Yellow
                            'confidence': confidence
                        })
                else:
                    # Unknown person
                    recognized_faces.append({
                        'name': 'Unknown',
                        'location': (top, right, bottom, left),
                        'color': (0, 0, 255),  # Red
                        'confidence': 0.0
                    })
            else:
                # No known faces to compare
                recognized_faces.append({
                    'name': 'No Reference Data',
                    'location': (top, right, bottom, left),
                    'color': (128, 128, 128),  # Gray
                    'confidence': 0.0
                })
        
        # Clear buffers for faces not seen in this frame
        current_rolls = []
        for face in recognized_faces:
            if '(' in face['name'] and ')' in face['name']:
                roll = face['name'].split('(')[1].split(')')[0]
                current_rolls.append(roll)
        
        # Decay buffers for faces not currently visible
        for roll in list(self.recognition_buffer.keys()):
            if roll not in current_rolls:
                self.recognition_buffer[roll] = max(0, self.recognition_buffer[roll] - 1)
                if self.recognition_buffer[roll] == 0:
                    del self.recognition_buffer[roll]
        
        return recognized_faces
    
    def run(self):
        """Main recognition loop"""
        print("🚀 Starting recognition thread...")
        
        if not self.load_students():
            self.state = "ERROR"
            return
        
        if not self.start_camera():
            self.state = "ERROR"
            return
        
        self.running = True
        self.state = "RUNNING"
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Motion detection for occupancy
                self.detect_motion(frame)
                
                # Face recognition (process every 3rd frame for performance)
                if self.frame_count % 3 == 0:
                    recognized_faces = self.process_face_recognition(frame)
                    
                    # Draw rectangles and labels
                    for face in recognized_faces:
                        top, right, bottom, left = face['location']
                        color = face['color']
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        # Draw label
                        label = face['name']
                        if face['confidence'] > 0:
                            label += f" ({face['confidence']:.2f})"
                        
                        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                        cv2.putText(frame, label, (left + 6, bottom - 6), 
                                  cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                
                # Update occupancy state
                self.update_occupancy_state()
                
                # Add status overlay
                status_text = f"State: {self.occupancy_state} | Students: {len(self.known_encodings)}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (255, 255, 255), 2)
                
                # Show frame (optional - comment out for headless operation)
                cv2.imshow('Smart Classroom - Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"❌ Error in recognition loop: {e}")
            self.state = "ERROR"
        finally:
            self.cleanup()
    
    def stop(self):
        """Stop the recognition thread"""
        print("🛑 Stopping recognition thread...")
        self.running = False
        self.state = "STOPPED"
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Recognition thread cleaned up.")
    
    def snapshot(self):
        """Get current status snapshot"""
        return {
            'state': self.state,
            'occupancy': self.occupancy_state,
            'frame_count': self.frame_count,
            'students_loaded': len(self.known_encodings),
            'last_motion': self.last_motion_time,
            'recognition_buffer': dict(self.recognition_buffer)
        }

# Standalone function for command line usage
def run_recognizer(cam_index: int = 0):
    """Run recognizer in standalone mode"""
    init_db()
    
    recognizer = RecognizerThread(
        session_id=f"standalone-{int(time.time())}",
        mqtt_publish=lambda x: print(f"MQTT would send: {x}")
    )
    
    try:
        recognizer.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        recognizer.stop()

if __name__ == "__main__":
    run_recognizer()