import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from datetime import datetime

# Load face database
try:
    face_db = np.load('face_db.npy', allow_pickle=True).item()
except FileNotFoundError:
    print("Error: face_db.npy not found. Please run face_database.py first")
    exit(1)

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Matching parameters
SIMILARITY_THRESHOLD = 0.6
MAX_DISTANCE = 1.0

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

def save_frame(frame):
    if not os.path.exists('saved_frames'):
        os.makedirs('saved_frames')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"saved_frames/frame_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Frame saved as {filename}")

def main():
    cap = cv2.VideoCapture(0)
    window_name = "Employee Recognition (Drag to resize)"
    
    # Create resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # Initial size
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror frame
        frame = cv2.flip(frame, 1)
        original_h, original_w = frame.shape[:2]
        
        # Face detection
        faces = app.get(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            name, distance = find_best_match(face.embedding)
            
            color = (0, 255, 0) if name else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            label = name if name else "Unknown"
            confidence = 1 - distance
            cv2.putText(frame, f"{label} ({confidence:.2f})", 
                       (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Get current window size
        win_w = cv2.getWindowImageRect(window_name)[2]
        win_h = cv2.getWindowImageRect(window_name)[3]
        
        # Calculate aspect ratio preserving dimensions
        aspect_ratio = original_w / original_h
        if win_w / win_h > aspect_ratio:
            new_h = win_h
            new_w = int(new_h * aspect_ratio)
        else:
            new_w = win_w
            new_h = int(new_w / aspect_ratio)
        
        # Resize frame to fit window while maintaining aspect ratio
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        # Create black background canvas
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        
        # Center the resized frame on canvas
        x_offset = (win_w - new_w) // 2
        y_offset = (win_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
        
        # Display instructions
        cv2.putText(canvas, "Press 's' to save, 'q' to quit | Drag window to resize",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(window_name, canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_frame(frame)  # Save original frame, not resized one
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()