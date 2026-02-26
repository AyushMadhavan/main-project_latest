import os
import time
import json
import logging
import numpy as np
import cv2

logger = logging.getLogger("Forensics")

# MediaPipe FaceMesh (initialized once at import time)
import mediapipe as mp
_mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


def save_forensics(frame, bbox, matched_face, matches, name, captures_dir):
    """
    Save forensic captures for a detected face.

    Args:
        frame:        Full camera frame (numpy array).
        bbox:         Bounding box [x1, y1, x2, y2].
        matched_face: InsightFace face object.
        matches:      DB search result list.
        name:         Recognized person name.
        captures_dir: Directory to save captures into.

    Returns:
        dict: Paths to saved capture files.
    """
    capture_paths = {}
    os.makedirs(captures_dir, exist_ok=True)

    ts_str = f"{int(time.time())}"

    # 1. Original Face Crop
    face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if face_crop.size == 0:
        return capture_paths

    orig_name = f"{ts_str}_orig.jpg"
    cv2.imwrite(os.path.join(captures_dir, orig_name), face_crop)
    capture_paths['original'] = f"static/captures/{orig_name}"

    # 2. Mesh Visualization (468-point MediaPipe or 68-point fallback)
    mesh_viz = face_crop.copy()
    mesh_data_468 = None

    try:
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        results_mp = _mp_face_mesh.process(rgb_crop)

        if results_mp.multi_face_landmarks:
            h_c, w_c, _ = face_crop.shape
            raw_lmks = results_mp.multi_face_landmarks[0].landmark
            mesh_data_468 = [[lm.x * w_c, lm.y * h_c, lm.z * w_c] for lm in raw_lmks]

            for pt in mesh_data_468:
                cv2.circle(mesh_viz, (int(pt[0]), int(pt[1])), 1, (0, 165, 255), -1)
        else:
            raise ValueError("No MP Mesh found")

    except Exception:
        # Fallback to 68 points (InsightFace)
        if hasattr(matched_face, 'landmark_3d_68') and matched_face.landmark_3d_68 is not None:
            lmks = matched_face.landmark_3d_68.astype(int)
            lmks[:, 0] -= bbox[0]
            lmks[:, 1] -= bbox[1]
            for pt in lmks:
                cv2.circle(mesh_viz, (pt[0], pt[1]), 2, (0, 255, 255), -1)

    mesh_name = f"{ts_str}_mesh.jpg"
    cv2.imwrite(os.path.join(captures_dir, mesh_name), mesh_viz)
    capture_paths['mesh_img'] = f"static/captures/{mesh_name}"

    # 3. Save .obj and .json for 3D viewers
    obj_points = mesh_data_468
    if not obj_points and hasattr(matched_face, 'landmark_3d_68') and matched_face.landmark_3d_68 is not None:
        lmks_obj = matched_face.landmark_3d_68.copy()
        lmks_obj[:, 0] -= bbox[0]
        lmks_obj[:, 1] -= bbox[1]
        obj_points = lmks_obj.tolist()

    if obj_points:
        obj_name = f"{ts_str}_mesh.obj"
        with open(os.path.join(captures_dir, obj_name), 'w') as f:
            f.write(f"# Face Mesh {len(obj_points)} points - {name}\n")
            pts_np = np.array(obj_points)
            centroid = np.mean(pts_np, axis=0)
            centered_pts = pts_np - centroid
            max_dist = np.max(np.abs(centered_pts))
            if max_dist > 0:
                centered_pts /= max_dist

            f.write("# Face Mesh Points as Geometry\n")
            s = 0.007
            v_idx = 1
            for p in centered_pts:
                x, y, z = p[0], -p[1], p[2]
                f.write(f"v {x:.6f} {y+s:.6f} {z:.6f}\n")
                f.write(f"v {x+s:.6f} {y-s:.6f} {z+s:.6f}\n")
                f.write(f"v {x-s:.6f} {y-s:.6f} {z+s:.6f}\n")
                f.write(f"v {x:.6f} {y-s:.6f} {z-s:.6f}\n")
                f.write(f"f {v_idx} {v_idx+1} {v_idx+2}\n")
                f.write(f"f {v_idx} {v_idx+2} {v_idx+3}\n")
                f.write(f"f {v_idx} {v_idx+1} {v_idx+3}\n")
                f.write(f"f {v_idx+1} {v_idx+2} {v_idx+3}\n")
                v_idx += 4

        capture_paths['mesh_obj'] = f"static/captures/{obj_name}"

        json_name = f"{ts_str}_mesh.json"
        json_data = {
            "timestamp": time.time(),
            "person": name,
            "points_count": len(obj_points),
            "landmarks": [
                {"id": i, "x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
                for i, v in enumerate(obj_points)
            ]
        }
        with open(os.path.join(captures_dir, json_name), 'w') as f:
            json.dump(json_data, f, indent=2)
        capture_paths['mesh_json'] = f"static/captures/{json_name}"

    # 4. Enrolled/Reference Mesh
    if matches:
        ref_mesh_points = None
        if 'landmark_3d_468' in matches[0] and matches[0]['landmark_3d_468']:
            ref_mesh_points = np.array(matches[0]['landmark_3d_468'])
        elif 'landmark_3d_68' in matches[0] and matches[0]['landmark_3d_68']:
            ref_mesh_points = np.array(matches[0]['landmark_3d_68'])

        if ref_mesh_points is not None and len(ref_mesh_points) > 0:
            ref_viz = np.zeros((200, 200, 3), dtype=np.uint8)
            min_xy = np.min(ref_mesh_points, axis=0)
            max_xy = np.max(ref_mesh_points, axis=0)
            center = (min_xy + max_xy) / 2
            scale = 140.0 / (np.max(max_xy - min_xy) + 1e-6)
            ref_centered = (ref_mesh_points - center) * scale + [100, 100, 0]
            for pt in ref_centered.astype(int):
                cv2.circle(ref_viz, (pt[0], pt[1]), 1, (0, 255, 0), -1)
            cv2.putText(ref_viz, "Enrolled Mesh", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            ref_name = f"{ts_str}_ref_mesh.jpg"
            cv2.imwrite(os.path.join(captures_dir, ref_name), ref_viz)
            capture_paths['ref_mesh'] = f"static/captures/{ref_name}"

    logger.info(f"Saved forensics for {name}")
    return capture_paths
