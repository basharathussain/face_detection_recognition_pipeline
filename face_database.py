import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from tqdm import tqdm

# Initialize InsightFace with enhanced settings
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def register_employees(folder_path):
    face_db = {}
    
    # Process each employee folder
    for employee_folder in tqdm(os.listdir(folder_path)):
        employee_path = os.path.join(folder_path, employee_folder)
        if not os.path.isdir(employee_path):
            continue
            
        embeddings = []
        
        # Process each image for this employee
        for img_file in os.listdir(employee_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(employee_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Get face embeddings (multiple faces per image possible)
            faces = app.get(img)
            for face in faces:
                embeddings.append(face.embedding)
        
        # Store average embedding for better accuracy
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            face_db[employee_folder] = avg_embedding
    
    # Save database
    np.save('face_db.npy', face_db)
    print(f"\nDatabase created with {len(face_db)} employees")

if __name__ == "__main__":
    register_employees('employee_photos')