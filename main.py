import cv2
import yaml
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

# Components
from src.ingestion.stream_loader import StreamLoader
from src.reconstruction.inpainter import FaceInpainter
from src.recognition.tracker import IOUTracker
from database.vector_db import VectorDB
from dashboard.app import start_dashboard, update_frame, update_logs

# InsightFace
from insightface.app import FaceAnalysis

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def load_config(path: str = "settings.yaml") -> Dict:
    with open(path, 'r') as f:
        content = f.read()
        
    # Expand environment variables in the YAML content
    # matches ${VAR} or $VAR
    expanded_content = os.path.expandvars(content)
    
    return yaml.safe_load(expanded_content)

def main():
    config = load_config()
    
    # 1. Initialize Components
    logger.info("Initializing components...")
    
    # Dashboard
    start_dashboard(config=config.get('system', {}), port=5000)

    # Database
    db = VectorDB(
        uri=config['database']['uri'], 
        collection_name=config['database']['collection_name'],
        dim=config['database']['dim']
    )
    
    # 3. Stream & Tracker Initialization (Multi-Camera)
    cameras_config = config.get('cameras', [])
    streams = {}
    trackers = {}
    
    # Fallback if no cameras defined but stream key exists (legacy)
    if not cameras_config and 'stream' in config:
        cameras_config = [{
            'id': 'cam_default',
            'name': 'Default Camera',
            'source': config['stream']['source'],
            'active': True
        }]

    for cam in cameras_config:
        if not cam.get('active', True):
            continue
            
        cid = cam['id']
        logger.info(f"Initializing {cam['name']} ({cid})...")
        
        # Loader
        loader = StreamLoader(
            source=cam['source'],
            queue_size=config.get('stream', {}).get('queue_size', 5),
            width=config.get('stream', {}).get('width', 1280),
            height=config.get('stream', {}).get('height', 720)
        )
        loader.start()
        streams[cid] = {'loader': loader, 'config': cam}
        
        # Tracker
        trackers[cid] = {'tracker': IOUTracker(), 'identities': {}}

    if not streams:
        logger.error("No active cameras found!")
        return

    # Thread Pool
    executor = ThreadPoolExecutor(max_workers=4) # Increased for multi-cam
    
    # State
    fps_counter = 0
    start_time = time.time()
    
    try:
        while True:
            frames_to_stitch = []
            
            # Process each camera
            for cid, data in streams.items():
                loader = data['loader']
                cam_conf = data['config']
                
                ret, timestamp, frame = loader.read()
                if not ret:
                    # Create black placeholder
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "No Signal", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    # --- Processing Pipeline ---
                    
                    # 1. Detection
                    faces = app.get(frame)
                    
                    # 2. Prepare for Tracker
                    detections = []
                    face_map = {}
                    for i, face in enumerate(faces):
                        bbox = face.bbox.astype(int)
                        detections.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                        face_map[i] = face
                        
                    # 3. Update Camera-Specific Tracker
                    tracker_data = trackers[cid]
                    tracked_objects = tracker_data['tracker'].update(detections)
                    
                    for track_id, bbox in tracked_objects:
                        matched_face = None
                        for i, face in face_map.items():
                            f_bbox = face.bbox.astype(int)
                            if np.array_equal(f_bbox, bbox):
                                matched_face = face
                                break
                        
                        if matched_face is None: continue

                        # Identity
                        name = config['recognition']['unknown_label']
                        score = 0.0
                        local_identities = tracker_data['identities']
                        
                        if track_id in local_identities and local_identities[track_id]['score'] > 0.6:
                            name = local_identities[track_id]['name']
                            score = local_identities[track_id]['score']
                        else:
                            embedding = matched_face.embedding
                            matches = db.search_embedding(embedding, threshold=config['recognition']['similarity_threshold'])
                            if matches:
                                name = matches[0]['name']
                                score = matches[0]['score']
                                local_identities[track_id] = {'name': name, 'score': score}
                                
                            # Log (with location info!)
                            update_logs(name, score, location=cam_conf.get('name', cid))

                        # Visualization
                        color = (0, 255, 0) if name != config['recognition']['unknown_label'] else (0, 0, 255)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(frame, f"{name} ({score:.2f})", (bbox[0], bbox[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # 3D
                        if hasattr(matched_face, 'landmark_3d_68') and matched_face.landmark_3d_68 is not None:
                            for pt in matched_face.landmark_3d_68.astype(int):
                                cv2.circle(frame, (pt[0], pt[1]), 1, (0, 255, 255), -1)

                # Overlay Camera Name
                cv2.putText(frame, cam_conf['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Resize for Grid (fix to 640x480 for uniformity)
                frame_resized = cv2.resize(frame, (640, 480))
                frames_to_stitch.append(frame_resized)

            # Stitching (Simple logic: linear horizontal, wrapping if > 2)
            # For 2 cams: HConcat. For 4: 2x2.
            count = len(frames_to_stitch)
            if count == 1:
                final_view = frames_to_stitch[0]
            elif count == 2:
                final_view = np.hstack(frames_to_stitch)
            else:
                # Basic 2-column layout
                rows = []
                for i in range(0, count, 2):
                    chunk = frames_to_stitch[i:i+2]
                    if len(chunk) == 1: # Pad with black if odd
                        chunk.append(np.zeros((480, 640, 3), dtype=np.uint8))
                    rows.append(np.hstack(chunk))
                final_view = np.vstack(rows)

            # 5. Update and FPS
            update_frame(final_view)
            
            fps_counter += 1
            if time.time() - start_time > 1.0:
                logger.info(f"FPS: {fps_counter}")
                fps_counter = 0
                start_time = time.time()

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        for data in streams.values():
            data['loader'].stop()
        executor.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
