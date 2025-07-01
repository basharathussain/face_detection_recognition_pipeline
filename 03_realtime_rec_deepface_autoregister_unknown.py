import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from deepface import DeepFace
import os
import time
from datetime import datetime
import sys
import hashlib

# Configuration
SIMILARITY_THRESHOLD = 0.6
MAX_DISTANCE = 1.0
GENDER_CONFIDENCE_THRESHOLD = 0.75
UNKNOWN_FACE_TIMEOUT = 10  # seconds
CAPTURE_VARIANTS = 3
FACE_SIMILARITY_THRESHOLD = 0.8  # For grouping unknown faces

def initialize_camera():
    """Robust camera initialization with multiple fallbacks"""
    print("Initializing camera...")
    
    # Method 1: Try default backend first
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Camera initialized with default backend")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            else:
                cap.release()
    except Exception as e:
        print(f"Default backend failed: {e}")
    
    # Method 2: Try DirectShow (Windows specific)
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Camera initialized with DirectShow backend")
                return cap
            else:
                cap.release()
    except Exception as e:
        print(f"DirectShow backend failed: {e}")
    
    # Method 3: Try different camera indices
    for camera_index in range(1, 4):
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Camera initialized with index {camera_index}")
                    return cap
                else:
                    cap.release()
        except Exception as e:
            continue
    
    raise RuntimeError(
        "Could not initialize camera. Please check:\n"
        "1. Camera is connected and not used by another application\n"
        "2. Camera permissions are granted\n"
        "3. Try running as administrator"
    )

def safe_audio_setup():
    """Setup audio with error handling"""
    try:
        import pyaudio
        import webrtcvad
        import queue
        
        # Audio configuration
        AUDIO_FRAME_DURATION = 30  # ms
        AUDIO_SAMPLE_RATE = 16000
        AUDIO_SENSITIVITY = 1
        
        audio_queue = queue.Queue()
        vad = webrtcvad.Vad(AUDIO_SENSITIVITY)
        audio = pyaudio.PyAudio()
        
        def audio_callback(in_data, frame_count, time_info, status):
            audio_queue.put(in_data)
            return (None, pyaudio.paContinue)
        
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_DURATION / 1000),
            stream_callback=audio_callback
        )
        
        print("✓ Audio system initialized")
        return audio_queue, vad, stream, audio
        
    except Exception as e:
        print(f"⚠ Audio setup failed: {e}")
        print("Continuing without audio detection...")
        return None, None, None, None

# Initialize models with error handling
def initialize_models():
    """Initialize face recognition models"""
    try:
        print("Loading face recognition models...")
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("✓ InsightFace model loaded")
        return face_app
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

# Face Database
def load_face_database():
    """Load or create face database"""
    try:
        face_db = np.load('face_db.npy', allow_pickle=True).item()
        print(f"✓ Loaded face database with {len(face_db)} profiles")
    except FileNotFoundError:
        face_db = {}
        np.save('face_db.npy', face_db)
        print("✓ Created new face database")
    return face_db

def find_best_match(embedding, face_db):
    """Find best matching face in database"""
    best_match, min_distance = None, MAX_DISTANCE
    for name, db_embedding in face_db.items():
        try:
            dot_product = np.dot(embedding, db_embedding)
            norm_product = np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
            if norm_product == 0:
                continue
            similarity = dot_product / norm_product
            distance = 1 - similarity
            
            if similarity > SIMILARITY_THRESHOLD and distance < min_distance:
                min_distance = distance
                best_match = name
        except Exception:
            continue
    return best_match, min_distance

def generate_face_id(embedding):
    """Generate consistent face ID from embedding"""
    # Convert embedding to string and hash it consistently
    embedding_str = np.array2string(embedding, precision=6)
    return hashlib.md5(embedding_str.encode()).hexdigest()[:12]

def find_similar_unknown_face(embedding, unknown_faces):
    """Find if this embedding matches any existing unknown face"""
    for face_id, face_data in unknown_faces.items():
        # Compare with the most recent embedding
        if face_data['embeddings']:
            recent_embedding = face_data['embeddings'][-1]
            dot_product = np.dot(embedding, recent_embedding)
            norm_product = np.linalg.norm(embedding) * np.linalg.norm(recent_embedding)
            if norm_product > 0:
                similarity = dot_product / norm_product
                if similarity > FACE_SIMILARITY_THRESHOLD:
                    return face_id
    return None

def check_speaking(audio_queue, vad):
    """Check if person is speaking"""
    if audio_queue is None or vad is None:
        return False
        
    speech_frames = 0
    total_frames = 0
    AUDIO_SAMPLE_RATE = 16000
    
    for _ in range(5):
        try:
            audio_frame = audio_queue.get_nowait()
            if len(audio_frame) >= 2:
                if vad.is_speech(audio_frame, AUDIO_SAMPLE_RATE):
                    speech_frames += 1
                total_frames += 1
        except:
            pass
    return (speech_frames / total_frames) > 0.6 if total_frames > 0 else False

def analyze_gender_emotion(face_img):
    """Analyze gender, emotion and age"""
    try:
        if face_img.size == 0:
            return None
            
        analysis = DeepFace.analyze(
            face_img, 
            actions=['gender', 'emotion', 'age'],
            enforce_detection=False,
            silent=True
        )
        result = analysis[0] if isinstance(analysis, list) else analysis
        
        # Determine gender
        if result['gender']['Man'] > GENDER_CONFIDENCE_THRESHOLD:
            result['gender'] = 'Male'
        elif result['gender']['Woman'] > GENDER_CONFIDENCE_THRESHOLD:
            result['gender'] = 'Female'
        else:
            result['gender'] = 'Unknown'
            
        return result
    except Exception as e:
        return None

def save_frame(frame, folder, filename):
    """Save frame to folder"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), frame)

def create_new_profile(face_id, frames, face_db, unknown_faces):
    """Create new profile for unknown face"""
    print(f"\n🔔 New unknown face detected for {UNKNOWN_FACE_TIMEOUT} seconds!")
    print("Enter person's name (Firstname_Lastname) or press Enter to skip: ", end='')
    
    # Use a non-blocking input approach
    try:
        name = input().strip()
        if not name:
            print("Skipped profile creation")
            return face_db
    except KeyboardInterrupt:
        return face_db
    
    profile_folder = os.path.join('employee_photos', name)
    os.makedirs(profile_folder, exist_ok=True)
    
    # Save frames
    for i, frame in enumerate(frames[:CAPTURE_VARIANTS]):
        save_frame(frame, profile_folder, f"variant_{i+1}.jpg")
    
    # Save embedding
    avg_embedding = np.mean(unknown_faces[face_id]['embeddings'], axis=0)
    face_db[name] = avg_embedding
    np.save('face_db.npy', face_db)
    
    print(f"✓ New profile created for {name} with {len(frames)} images")
    return face_db

def draw_face_info(frame, face, bbox, name, distance, audio_queue, vad, unknown_faces):
    """Draw face information on frame"""
    color = (0, 255, 0) if name else (0, 0, 255)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    
    label = name if name else "Unknown"
    confidence = 1 - distance
    y_offset = 30
    
    # Name and confidence
    cv2.putText(frame, f"{label} ({confidence:.2f})", 
               (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # If unknown face, show timer
    if not name:
        # Find this face in unknown_faces
        face_id = find_similar_unknown_face(face.embedding, unknown_faces)
        if face_id and face_id in unknown_faces:
            elapsed = time.time() - unknown_faces[face_id]['first_seen']
            remaining = max(0, UNKNOWN_FACE_TIMEOUT - elapsed)
            cv2.putText(frame, f"Timer: {remaining:.1f}s", 
                       (bbox[0], bbox[1]-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Extract face region safely
    h, w = frame.shape[:2]
    y1, y2 = max(0, bbox[1]), min(h, bbox[3])
    x1, x2 = max(0, bbox[0]), min(w, bbox[2])
    
    face_img = frame[y1:y2, x1:x2]
    
    if face_img.size > 0:
        analysis = analyze_gender_emotion(face_img)
        
        if analysis:
            # Age and gender
            if analysis['gender'] != 'Unknown':
                cv2.putText(frame, f"{analysis['age']:.0f} {analysis['gender']}", 
                           (bbox[0], bbox[3]+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y_offset += 25
            
            # Emotion
            emotion = analysis['dominant_emotion']
            cv2.putText(frame, f"Emotion: {emotion}", 
                       (bbox[0], bbox[3]+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_offset += 25
    
    # Speaking detection
    speaking = check_speaking(audio_queue, vad)
    cv2.putText(frame, f"Speaking: {'YES' if speaking else 'no'}", 
               (bbox[0], bbox[3]+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
               (0, 255, 0) if speaking else (0, 0, 255), 1)

def main():
    """Main function"""
    print("🚀 Starting Enhanced Face Recognition System")
    
    # Create directories
    os.makedirs('saved_frames', exist_ok=True)
    os.makedirs('employee_photos', exist_ok=True)
    
    # Initialize components
    try:
        cap = initialize_camera()
        face_app = initialize_models()
        face_db = load_face_database()
        audio_queue, vad, stream, audio = safe_audio_setup()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    unknown_faces = {}
    cv2.namedWindow("Enhanced Recognition", cv2.WINDOW_NORMAL)
    print("✓ System ready! Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
                
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            try:
                faces = face_app.get(frame)
            except Exception as e:
                print(f"Face detection error: {e}")
                continue
                
            current_time = time.time()
            frame_count += 1
            
            # Print face count every 30 frames to reduce spam
            if frame_count % 30 == 0:
                print(f"Faces detected: {len(faces)}, Unknown faces tracked: {len(unknown_faces)}")

            # Process each face
            for face in faces:
                bbox = face.bbox.astype(int)
                name, distance = find_best_match(face.embedding, face_db)
                
                # Handle unknown faces
                if name is None:
                    # Check if this face is similar to any existing unknown face
                    face_id = find_similar_unknown_face(face.embedding, unknown_faces)
                    
                    if face_id is None:
                        # This is a completely new unknown face
                        face_id = generate_face_id(face.embedding)
                        unknown_faces[face_id] = {
                            'first_seen': current_time,
                            'embeddings': [face.embedding],
                            'frames': [frame.copy()],
                            'last_seen': current_time
                        }
                        print(f"📷 New unknown face detected: {face_id[:8]}")
                    else:
                        # This is a similar face we've seen before
                        unknown_faces[face_id]['embeddings'].append(face.embedding)
                        unknown_faces[face_id]['last_seen'] = current_time
                        if len(unknown_faces[face_id]['frames']) < CAPTURE_VARIANTS:
                            unknown_faces[face_id]['frames'].append(frame.copy())
                        
                        # Check if it's time to create profile
                        elapsed_time = current_time - unknown_faces[face_id]['first_seen']
                        if elapsed_time > UNKNOWN_FACE_TIMEOUT:
                            print(f"⏰ Creating profile for face {face_id[:8]} (seen for {elapsed_time:.1f}s)")
                            face_db = create_new_profile(face_id, unknown_faces[face_id]['frames'], 
                                                       face_db, unknown_faces)
                            if face_id in unknown_faces:
                                del unknown_faces[face_id]
                
                # Draw face information
                draw_face_info(frame, face, bbox, name, distance, audio_queue, vad, unknown_faces)
            
            # Clean up old unknown faces that haven't been seen recently
            faces_to_remove = []
            for face_id, face_data in unknown_faces.items():
                time_since_last_seen = current_time - face_data['last_seen']
                if time_since_last_seen > 5:  # Remove if not seen for 5 seconds
                    faces_to_remove.append(face_id)
                    print(f"🗑️ Removing stale unknown face: {face_id[:8]}")
            
            for face_id in faces_to_remove:
                del unknown_faces[face_id]
            
            # Display instructions and status
            cv2.putText(frame, "Press 's' to save, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Unknown faces: {len(unknown_faces)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Enhanced Recognition", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"saved_frames/frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved: {filename}")
                
    except KeyboardInterrupt:
        print("\n⏹ Stopping...")
    
    finally:
        # Cleanup
        if stream:
            stream.stop_stream()
            stream.close()
        if audio:
            audio.terminate()
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")

if __name__ == "__main__":
    main()