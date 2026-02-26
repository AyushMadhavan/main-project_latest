import os
import logging
import threading
import time
import json
import cv2
import numpy as np
import requests
from collections import deque
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

logger = logging.getLogger("Dashboard")

# Resolve frontend folder relative to project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_TEMPLATE_FOLDER = os.path.join(_PROJECT_ROOT, "frontend", "templates")
_STATIC_FOLDER = os.path.join(_PROJECT_ROOT, "frontend", "static")

app = Flask(__name__, template_folder=_TEMPLATE_FOLDER, static_folder=_STATIC_FOLDER)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Temporary hardcoded users
USERS = {
    'admin': {'id': 1, 'password': 'admin123'},
    'operator': {'id': 2, 'password': 'operator123'}
}


class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username


@login_manager.user_loader
def load_user(user_id):
    for username, data in USERS.items():
        if str(data['id']) == str(user_id):
            return User(data['id'], username)
    return None


# ── Shared state ────────────────────────────────────────────────────────────

frame_buffer = None
buffer_lock = threading.Lock()

logs_buffer = deque(maxlen=50)
logs_lock = threading.Lock()

analytics_data = {
    "detections_over_time": {},
    "top_criminals": {},
    "total_detections": 0
}
analytics_lock = threading.Lock()

system_config = {}

HISTORY_FILE = "detection_history.json"


# ── Public API used by pipeline ──────────────────────────────────────────────

def update_frame(frame):
    """Update the shared video frame buffer."""
    global frame_buffer
    with buffer_lock:
        frame_buffer = frame.copy()


def update_logs(name, score, location="Unknown", gps=None, mesh=None, captures=None):
    """Add a detection log entry and update analytics."""
    timestamp = time.time()

    with logs_lock:
        logs_buffer.appendleft({
            "timestamp": timestamp,
            "name": name,
            "score": float(score),
            "location": location,
            "captures": captures
        })

    with analytics_lock:
        analytics_data["total_detections"] += 1
        if name != "Unknown":
            analytics_data["top_criminals"][name] = analytics_data["top_criminals"].get(name, 0) + 1

            final_gps = gps if gps else system_config.get("gps", {"lat": 0, "lng": 0})
            entry = {
                "timestamp": timestamp,
                "name": name,
                "score": float(score),
                "location": location,
                "device_id": system_config.get("device_id", "Unknown"),
                "gps": final_gps,
                "mesh": mesh,
                "captures": captures
            }
            _save_history_entry(entry)

            hq_url = system_config.get("central_server")
            if hq_url:
                def push_to_hq(payload, url):
                    try:
                        requests.post(f"{url}/api/ingest", json=payload, timeout=2)
                    except Exception:
                        pass
                threading.Thread(target=push_to_hq, args=(entry, hq_url)).start()

        t_str = time.strftime("%H:%M")
        analytics_data["detections_over_time"][t_str] = analytics_data["detections_over_time"].get(t_str, 0) + 1


def _save_history_entry(entry):
    try:
        with open(HISTORY_FILE, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to save history: {e}")


def _generate_stream():
    """MJPEG video stream generator."""
    while True:
        try:
            with buffer_lock:
                if frame_buffer is None:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Waiting for Camera...", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    _, encoded = cv2.imencode(".jpg", placeholder)
                else:
                    _, encoded = cv2.imencode(".jpg", frame_buffer)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')
            time.sleep(0.04)
        except Exception as e:
            logger.error(f"Stream error: {e}")
            time.sleep(0.1)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USERS and USERS[username]['password'] == password:
            login_user(User(USERS[username]['id'], username), remember=request.form.get('remember'))
            return redirect(url_for('index'))
        return redirect(url_for('login', error=1))
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route("/")
@login_required
def index():
    return render_template("index.html", username=current_user.username)


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(_generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/logs")
@login_required
def get_logs():
    with logs_lock:
        return jsonify(list(logs_buffer))


@app.route("/api/analytics")
@login_required
def get_analytics():
    with analytics_lock:
        sorted_criminals = dict(
            sorted(analytics_data["top_criminals"].items(), key=lambda x: x[1], reverse=True)[:5]
        )
        return jsonify({
            "system": system_config,
            "stats": {
                "total": analytics_data["total_detections"],
                "top_criminals": sorted_criminals,
                "timeline": analytics_data["detections_over_time"]
            }
        })


@app.route("/api/history/<name>")
def get_history(name):
    try:
        track = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get('name') == name:
                            gps = entry.get('gps')
                            if gps and isinstance(gps, dict) and 'lat' in gps and 'lng' in gps:
                                if gps['lat'] != 0 or gps['lng'] != 0:
                                    track.append({
                                        'lat': gps['lat'],
                                        'lng': gps['lng'],
                                        'timestamp': entry.get('timestamp')
                                    })
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON line in history.")
        track.sort(key=lambda x: x['timestamp'])
        return jsonify(track)
    except Exception as e:
        logger.error(f"Error in get_history for {name}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/trail/<name>")
def get_trail(name):
    trail = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry['name'] == name:
                            trail.append(entry)
                    except Exception:
                        pass
        except Exception:
            pass
    trail.sort(key=lambda x: x['timestamp'])
    return jsonify(trail)


# ── Enrollment ───────────────────────────────────────────────────────────────

# Shared face analysis model (set by pipeline at startup)
_face_app = None
_face_app_lock = threading.Lock()


def set_face_app(face_app_instance):
    """Called by pipeline to share the InsightFace model."""
    global _face_app
    _face_app = face_app_instance


def _get_or_create_face_app():
    """Get shared model or create one for enrollment-only usage."""
    global _face_app
    if _face_app is not None:
        return _face_app

    with _face_app_lock:
        if _face_app is not None:
            return _face_app
        from insightface.app import FaceAnalysis
        logger.info("Enrollment: Loading InsightFace model (standalone)...")
        fa = FaceAnalysis(
            name="buffalo_l",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition', 'landmark_3d_68']
        )
        fa.prepare(ctx_id=0, det_thresh=0.5)
        _face_app = fa
        return _face_app


# VectorDB instance (set by pipeline at startup)
_vector_db = None


def set_vector_db(db_instance):
    """Called by pipeline to share the VectorDB."""
    global _vector_db
    _vector_db = db_instance


def _get_or_create_db():
    """Get shared DB or create one for enrollment-only usage."""
    global _vector_db
    if _vector_db is not None:
        return _vector_db
    from backend.database.vector_db import VectorDB
    _vector_db = VectorDB(uri="./milvus_demo_local.json")
    return _vector_db


@app.route("/enroll")
@login_required
def enroll_page():
    """Serve the enrollment page (admin-only, requires re-auth via JS)."""
    return render_template("enroll.html", username=current_user.username)


@app.route("/api/enroll/verify", methods=["POST"])
@login_required
def enroll_verify():
    """Re-authenticate the current user for enrollment access."""
    data = request.get_json()
    password = data.get("password", "")
    username = current_user.username

    if username not in USERS:
        return jsonify({"ok": False, "error": "User not found"}), 403

    if username != "admin":
        return jsonify({"ok": False, "error": "Only admin can enroll faces"}), 403

    if USERS[username]["password"] != password:
        return jsonify({"ok": False, "error": "Incorrect password"}), 401

    # Set a session flag so the enroll page knows auth passed
    session["enroll_verified"] = True
    return jsonify({"ok": True})


@app.route("/api/enroll/submit", methods=["POST"])
@login_required
def enroll_submit():
    """Accept images + name, run InsightFace, save embeddings to VectorDB."""
    if current_user.username != "admin":
        return jsonify({"ok": False, "error": "Only admin can enroll faces"}), 403

    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Name is required"}), 400

    files = request.files.getlist("images")
    if not files:
        return jsonify({"ok": False, "error": "No images provided"}), 400

    face_app = _get_or_create_face_app()
    db = _get_or_create_db()

    # MediaPipe for 468-point mesh
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    )

    results = []
    records = []

    for f in files:
        fname = f.filename or "unknown"
        try:
            # Read image from upload
            file_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                results.append({"file": fname, "ok": False, "error": "Invalid image"})
                continue

            faces = face_app.get(img)
            if not faces:
                results.append({"file": fname, "ok": False, "error": "No face detected"})
                continue

            if len(faces) > 1:
                results.append({"file": fname, "ok": False, "error": "Multiple faces detected — use a single-face image"})
                continue

            face = faces[0]
            embedding = face.embedding

            # 68-point landmarks
            landmarks_3d = None
            if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                landmarks_3d = face.landmark_3d_68

            # 468-point MediaPipe mesh
            landmarks_468 = None
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_result = mp_face_mesh.process(rgb)
                if mp_result.multi_face_landmarks:
                    h, w, _ = img.shape
                    raw = mp_result.multi_face_landmarks[0].landmark
                    landmarks_468 = [[lm.x * w, lm.y * h, lm.z * w] for lm in raw]
            except Exception:
                pass

            record = {
                "vector": embedding,
                "name": name,
                "id": len(db.data) + len(records) + 1,
                "landmark_3d_68": landmarks_3d,
                "landmark_3d_468": landmarks_468
            }
            records.append(record)
            results.append({"file": fname, "ok": True})

        except Exception as e:
            results.append({"file": fname, "ok": False, "error": str(e)})

    mp_face_mesh.close()

    enrolled_count = 0
    if records:
        db.insert_embeddings(records)
        enrolled_count = len(records)
        logger.info(f"Enrolled {enrolled_count} face(s) for '{name}'")

    return jsonify({
        "ok": True,
        "enrolled": enrolled_count,
        "total_submitted": len(files),
        "details": results
    })


# ── Server start ─────────────────────────────────────────────────────────────

def start_dashboard(config=None, host='0.0.0.0', port=5000):
    """Start Flask dashboard in a background daemon thread."""
    global system_config
    if config:
        system_config = config

    def run():
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    logger.info(f"Dashboard started at http://{host}:{port}")
