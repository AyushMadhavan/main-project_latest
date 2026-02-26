import os
import sys
import time
import logging
import json
import numpy as np
import cv2
import yaml
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from dotenv import load_dotenv
import requests

from insightface.app import FaceAnalysis

from backend.engine.location import get_live_location
from backend.engine.forensics import save_forensics
from backend.database.vector_db import VectorDB
from backend.ingestion.stream_loader import StreamLoader
from backend.recognition.tracker import IOUTracker
from backend.reconstruction.inpainter import FaceInpainter
from backend.api.server import start_dashboard, update_frame, update_logs, set_face_app, set_vector_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

load_dotenv()


def load_config(path: str = "settings.yaml") -> Dict:
    with open(path, 'r') as f:
        content = f.read()
    expanded_content = os.path.expandvars(content)
    return yaml.safe_load(expanded_content)


def send_to_hq(url, data):
    """Send detection data to HQ Server asynchronously."""
    if not url:
        return
    try:
        if not url.endswith('/api/ingest'):
            url = url.rstrip('/') + '/api/ingest'
        requests.post(url, json=data, timeout=2)
    except Exception as e:
        logger.debug(f"Failed to send to HQ: {e}")


def main():
    # Ensure CWD is the application directory
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
        # When called from backend/engine/pipeline.py, go up two levels to project root
        application_path = os.path.dirname(os.path.dirname(application_path))

    os.chdir(application_path)
    logger.info(f"Running in: {application_path}")

    config = load_config()

    # 0. Auto-Detect Location
    detected_gps = get_live_location(config)
    if detected_gps:
        config.setdefault('system', {})
        config['system']['gps'] = detected_gps
        for cam in config.get('cameras', []):
            if isinstance(cam.get('source'), int):
                cam['gps'] = detected_gps

    # 1. Dashboard
    start_dashboard(config=config.get('system', {}), port=5000)

    # 2. Database
    db = VectorDB(
        uri=config['database']['uri'],
        collection_name=config['database']['collection_name'],
        dim=config['database']['dim']
    )

    # 3. Models
    logger.info("Loading models...")
    inpainter = None
    if config['inpainting']['enabled']:
        logger.info("Loading Inpainting model...")
        inpainter = FaceInpainter(
            model_path=config['inpainting']['model_path'],
            mask_threshold=config['inpainting']['mask_threshold']
        )
    else:
        logger.info("Inpainting disabled.")

    face_app = FaceAnalysis(
        name=config['detection']['model_name'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition', 'landmark_3d_68']
    )
    face_app.prepare(ctx_id=config['detection']['ctx_id'], det_thresh=config['detection']['det_thresh'])

    # Share model and DB with enrollment routes
    set_face_app(face_app)
    set_vector_db(db)

    # 4. Streams & Trackers
    cameras_config = config.get('cameras', [])
    streams = {}
    trackers = {}

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
        loader = StreamLoader(
            source=cam['source'],
            queue_size=config.get('stream', {}).get('queue_size', 5),
            width=config.get('stream', {}).get('width', 1280),
            height=config.get('stream', {}).get('height', 720)
        )
        loader.start()
        streams[cid] = {'loader': loader, 'config': cam}
        trackers[cid] = {'tracker': IOUTracker(), 'identities': {}}

    if not streams:
        logger.error("No active cameras found!")
        return

    executor = ThreadPoolExecutor(max_workers=4)
    fps_counter = 0
    start_time = time.time()
    captures_dir = os.path.join("frontend", "static", "captures")

    # Pre-fill grid buffer with black frames
    grid_buffer = {}
    last_frame_time = {}
    for cid in streams:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Initializing...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        grid_buffer[cid] = blank
        last_frame_time[cid] = time.time()

    try:
        while True:
            any_new_frame = False

            for cid, data in streams.items():
                loader = data['loader']
                cam_conf = data['config']

                ret, timestamp, frame = loader.read()
                if not ret:
                    if time.time() - last_frame_time[cid] > 2.0:
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, cam_conf['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(blank, "Source Input Lost", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        grid_buffer[cid] = blank
                    continue

                any_new_frame = True
                last_frame_time[cid] = time.time()

                # Detection
                try:
                    faces = face_app.get(frame)
                except Exception as e:
                    logger.error(f"Detection error: {e}")
                    faces = []

                # Prepare detections for tracker
                detections = []
                face_map = {}
                for i, face in enumerate(faces):
                    bbox = face.bbox.astype(int)
                    detections.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                    face_map[i] = face

                # Update tracker
                tracker_data = trackers[cid]
                tracked_objects = tracker_data['tracker'].update(detections)

                for track_id, bbox in tracked_objects:
                    # Match tracked box to face
                    matched_face = None
                    for i, face in face_map.items():
                        if np.array_equal(face.bbox.astype(int), bbox):
                            matched_face = face
                            break
                    if matched_face is None:
                        continue

                    # Identity
                    name = config['recognition']['unknown_label']
                    score = 0.0
                    local_identities = tracker_data['identities']
                    matches = []

                    if track_id in local_identities and local_identities[track_id]['score'] > 0.6:
                        name = local_identities[track_id]['name']
                        score = local_identities[track_id]['score']
                    else:
                        matches = db.search_embedding(
                            matched_face.embedding,
                            threshold=config['recognition']['similarity_threshold']
                        )
                        if matches:
                            name = matches[0]['name']
                            score = matches[0]['score']
                            local_identities[track_id] = {'name': name, 'score': score}

                        if track_id not in local_identities:
                            local_identities[track_id] = {'name': name, 'score': score}

                        # Forensics (once per track)
                        capture_paths = {}
                        if 'forensics_saved' not in local_identities[track_id]:
                            try:
                                capture_paths = save_forensics(
                                    frame, bbox, matched_face, matches, name, captures_dir
                                )
                                local_identities[track_id]['captures'] = capture_paths
                                local_identities[track_id]['forensics_saved'] = True
                            except Exception as e:
                                logger.error(f"Forensics capture failed: {e}")

                        if not capture_paths and 'captures' in local_identities.get(track_id, {}):
                            capture_paths = local_identities[track_id]['captures']

                        # 3D mesh for logs
                        mesh_data = None
                        if hasattr(matched_face, 'landmark_3d_68') and matched_face.landmark_3d_68 is not None:
                            mesh_data = matched_face.landmark_3d_68.tolist()

                        # Log (throttled: max once per 10s per person)
                        last_log = local_identities[track_id].get('last_log', 0)
                        if time.time() - last_log > 10.0:
                            update_logs(name, score,
                                        location=cam_conf.get('name', cid),
                                        gps=cam_conf.get('gps'),
                                        mesh=mesh_data,
                                        captures=capture_paths)
                            local_identities[track_id]['last_log'] = time.time()

                            # Send to HQ
                            hq_url = config.get('system', {}).get('central_server')
                            if hq_url:
                                payload = {
                                    'name': name,
                                    'score': float(score),
                                    'timestamp': time.time(),
                                    'location': cam_conf.get('name', cid),
                                    'gps': cam_conf.get('gps'),
                                    'device_id': config.get('system', {}).get('device_id', 'Unknown'),
                                    'mesh': mesh_data,
                                    'captures': capture_paths
                                }
                                executor.submit(send_to_hq, hq_url, payload)

                    # Visualization
                    color = (0, 255, 0) if name != config['recognition']['unknown_label'] else (0, 0, 255)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, f"{name} ({score:.2f})", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if hasattr(matched_face, 'landmark_3d_68') and matched_face.landmark_3d_68 is not None:
                        for pt in matched_face.landmark_3d_68.astype(int):
                            cv2.circle(frame, (pt[0], pt[1]), 1, (0, 255, 255), -1)

                cv2.putText(frame, cam_conf['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                grid_buffer[cid] = cv2.resize(frame, (640, 480))

            # Stitch frames for dashboard
            if not any_new_frame:
                time.sleep(0.01)

            frames_to_stitch = list(grid_buffer.values())
            count = len(frames_to_stitch)
            if count == 1:
                final_view = frames_to_stitch[0]
            elif count == 2:
                final_view = np.hstack(frames_to_stitch)
            else:
                rows = []
                for i in range(0, count, 2):
                    chunk = frames_to_stitch[i:i + 2]
                    if len(chunk) == 1:
                        chunk.append(np.zeros((480, 640, 3), dtype=np.uint8))
                    rows.append(np.hstack(chunk))
                final_view = np.vstack(rows)

            if final_view is not None and final_view.size > 0:
                update_frame(final_view)
            else:
                logger.warning("Main Loop: Final view is empty or None")

            fps_counter += 1
            if time.time() - start_time > 1.0:
                logger.info(f"FPS: {fps_counter}")
                fps_counter = 0
                start_time = time.time()

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        for stream_data in streams.values():
            stream_data['loader'].stop()
        executor.shutdown()
        cv2.destroyAllWindows()
