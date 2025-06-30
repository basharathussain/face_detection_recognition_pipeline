import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from deepface import DeepFace
import pyaudio
import webrtcvad
import queue
import os
from datetime import datetime

# Configuration
SIMILARITY_THRESHOLD = 0.6
MAX_DISTANCE = 1.0
AUDIO_FRAME_DURATION = 30  # ms
AUDIO_SAMPLE_RATE = 16000
GENDER_CONFIDENCE_THRESHOLD = 0.75  # Require higher confidence for gender
AUDIO_SENSITIVITY = 1  # 1-3 (higher = more sensitive speaking detection)

# Initialize InsightFace
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Audio Setup with increased sensitivity
audio_queue = queue.Queue()
vad = webrtcvad.Vad(AUDIO_SENSITIVITY)  # More sensitive voice detection

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=AUDIO_SAMPLE_RATE,
    input=True,
    frames_per_buffer=int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_DURATION / 1000),
    stream_callback=audio_callback,
    input_device_index=None  # Auto-select best microphone
)

# Face Database
try:
    face_db = np.load('face_db.npy', allow_pickle=True).item()
except FileNotFoundError:
    print("Error: face_db.npy not found. Please run face_database.py first")
    exit(1)

def find_best_match(embedding):
    best_match = None
    min_distance = MAX_DISTANCE
    
    for name, db_embedding in face_db.items():
        dot_product = np.dot(embedding, db_embedding)
        norm_product = np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
        similarity = dot_product / norm_product
        distance = 1 - similarity
        
        if similarity > SIMILARITY_THRESHOLD and distance < min_distance:
            min_distance = distance
            best_match = name
            
    return best_match, min_distance

def check_speaking():
    """More accurate speaking detection with multiple audio frames"""
    speech_frames = 0
    total_frames = 0
    
    # Check multiple frames for more reliable detection
    for _ in range(5):  # Check 5 consecutive frames
        try:
            audio_frame = audio_queue.get_nowait()
            if len(audio_frame) >= 2:
                if vad.is_speech(audio_frame, AUDIO_SAMPLE_RATE):
                    speech_frames += 1
                total_frames += 1
        except queue.Empty:
            pass
    
    # Consider speaking if majority of frames contain speech
    return (speech_frames / total_frames) > 0.6 if total_frames > 0 else False

def analyze_gender_emotion(face_img):
    """More accurate gender analysis with confidence threshold"""
    try:
        analysis = DeepFace.analyze(
            face_img, 
            actions=['gender', 'emotion', 'age'],
            enforce_detection=False,
            silent=True
        )
        result = analysis[0] if isinstance(analysis, list) else analysis
        
        # Only return gender if confidence is high enough
        if result['gender']['Man'] > GENDER_CONFIDENCE_THRESHOLD:
            result['gender'] = 'Male'
        elif result['gender']['Woman'] > GENDER_CONFIDENCE_THRESHOLD:
            result['gender'] = 'Female'
        else:
            result['gender'] = 'Unknown'
            
        return result
    except Exception as e:
        print(f"Analysis error: {e}")
        return None

def save_frame(frame):
    if not os.path.exists('saved_frames'):
        os.makedirs('saved_frames')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"saved_frames/frame_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Frame saved as {filename}")

def draw_face_info(frame, face, bbox, name, distance):
    color = (0, 255, 0) if name else (0, 0, 255)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    
    label = name if name else "Unknown"
    confidence = 1 - distance
    y_offset = 30
    
    cv2.putText(frame, f"{label} ({confidence:.2f})", 
               (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    face_img = frame[max(0, bbox[1]):min(frame.shape[0], bbox[3]), 
                    max(0, bbox[0]):min(frame.shape[1], bbox[2])]
    
    if face_img.size > 0:
        analysis = analyze_gender_emotion(face_img)
    else:
        analysis = None
    
    if analysis:
        # Display gender only if confident
        if analysis['gender'] != 'Unknown':
            cv2.putText(frame, f"{analysis['age']} {analysis['gender']}", 
                       (bbox[0], bbox[3]+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_offset += 25
        
        # Emotion
        emotion = analysis['dominant_emotion']
        cv2.putText(frame, f"Emotion: {emotion}", 
                   (bbox[0], bbox[3]+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y_offset += 25
        
        # Speaking detection
        speaking = check_speaking()
        cv2.putText(frame, f"Speaking: {'YES' if speaking else 'no'}", 
                   (bbox[0], bbox[3]+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if speaking else (0, 0, 255), 1)

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Enhanced Recognition", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        faces = face_app.get(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            name, distance = find_best_match(face.embedding)
            draw_face_info(frame, face, bbox, name, distance)
        
        cv2.putText(frame, "Press 's' to save, 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Enhanced Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_frame(frame)
            
    stream.stop_stream()
    stream.close()
    audio.terminate()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()