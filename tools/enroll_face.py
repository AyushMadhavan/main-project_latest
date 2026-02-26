import cv2
import numpy as np
import argparse
import os
import sys
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.vector_db import VectorDB
from insightface.app import FaceAnalysis

def main():
    parser = argparse.ArgumentParser(description="Enroll a face from webcam")
    parser.add_argument("--name", type=str, required=True, help="Name of the person to enroll")
    parser.add_argument("--source", type=int, default=0, help="Webcam source ID")
    args = parser.parse_args()

    # Initialize DB
    print("Initializing Database...")
    db = VectorDB(uri="./milvus_demo_local.json")

    # Initialize Face Detection
    print("Initializing AI Models...")
    # Use CPU/GPU based on availability (try GPU first)
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection', 'recognition', 'landmark_3d_68'])
    app.prepare(ctx_id=0, det_thresh=0.5)

    # Open Webcam
    print(f"Opening Webcam {args.source}...")
    # Use DSHOW on Windows
    if os.name == 'nt':
        cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    new_records = []
    print(f"Enrollment Session for: {args.name}")
    print("Press 'SPACE' to capture a photo.")
    print("Press 'q' to finish and save.")

    # Initialize MediaPipe Face Mesh
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Draw generic UI
        display = frame.copy()
        cv2.putText(display, f"Enroll: {args.name}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Captured: {len(new_records)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, "SPACE: Capture | Q: Save & Quit", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.imshow("Enrollment", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Capture!
            print("Capturing...")
            faces = app.get(frame)
            
            if len(faces) == 0:
                print("No face detected! Try again.")
                continue
            
            if len(faces) > 1:
                print("Multiple faces detected! Please ensure only one person is in view.")
                continue
                
            # Got one face
            face = faces[0]
            embedding = face.embedding
            
            # Extract 3D Landmarks if available
            landmarks_3d = None
            if hasattr(face, 'landmark_3d_68'):
                landmarks_3d = face.landmark_3d_68

            # Extract 468-point Mesh (MediaPipe)
            # We crop the face to ensure MediaPipe finds the same face (though max_num_faces=1 helps)
            # Actually, running on full frame is safer for relative coords if we want robust logic,
            # but usually cropping with padding is better for small faces.
            # Let's run on full frame for simplicity since only 1 face is allowed.
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            landmarks_468 = None
            if results.multi_face_landmarks:
                # Convert normalized landmarks to pixel coordinates (or keep normalized?)
                # InsightFace stores pixel coords. Let's store pixel coords for consistency.
                h, w, c = frame.shape
                raw_lmks = results.multi_face_landmarks[0].landmark
                landmarks_468 = [[lm.x * w, lm.y * h, lm.z * w] for lm in raw_lmks] # z is relative to width in MP

            # Add to buffer
            record = {
                "vector": embedding,
                "name": args.name,
                "id": len(db.data) + len(new_records) + 1,
                "landmark_3d_68": landmarks_3d,
                "landmark_3d_468": landmarks_468
            }
            new_records.append(record)
            print(f"Captured image #{len(new_records)} for {args.name}")
            time.sleep(0.5) # Debounce 

    if new_records:
        print(f"Saving {len(new_records)} records to database...")
        db.insert_embeddings(new_records)
        print("Done!")
    else:
        print("No images captured.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
