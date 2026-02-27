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
from backend.database.auth_db import init_db, get_user_by_id, get_user_by_username, verify_password

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

# Initialize SQLite user database
init_db()


class User(UserMixin):
    def __init__(self, id, username, role='operator'):
        self.id = id
        self.username = username
        self.role = role


@login_manager.user_loader
def load_user(user_id):
    row = get_user_by_id(user_id)
    if row:
        return User(row['id'], row['username'], row['role'])
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
        if verify_password(username, password):
            user_row = get_user_by_username(username)
            login_user(User(user_row['id'], user_row['username'], user_row['role']),
                       remember=request.form.get('remember'))
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
        # Filter out unknown/unidentified faces
        filtered_logs = [log for log in logs_buffer if log.get('name') != 'Unknown']
        return jsonify(filtered_logs)


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

    if getattr(current_user, 'role', None) != 'admin':
        return jsonify({"ok": False, "error": "Only admin can enroll faces"}), 403

    if not verify_password(username, password):
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
        
    raw_emails = request.form.get("alert_emails", "").strip()
    alert_emails = []
    if raw_emails:
        # Split by comma, strip whitespace, remove empty, take up to 3
        alert_emails = [e.strip() for e in raw_emails.split(",") if e.strip()][:3]

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
            if alert_emails:
                record["alert_emails"] = alert_emails

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


# ── Forensics Gallery ────────────────────────────────────────────────────────

@app.route("/api/forensics")
@login_required
def get_forensics():
    """Return all detection history entries for the gallery."""
    entries = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    pass
    # Sort newest first
    entries.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    return jsonify(entries)


@app.route("/api/forensics/captures")
@login_required
def get_captures():
    """Return capture images grouped by person."""
    captures_dir = os.path.join(_STATIC_FOLDER, "captures")
    if not os.path.exists(captures_dir):
        return jsonify({})

    # Read history to map capture files to names
    person_captures = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    name = entry.get('name', 'Unknown')
                    caps = entry.get('captures', {})
                    if caps:
                        person_captures.setdefault(name, []).append({
                            'timestamp': entry.get('timestamp', 0),
                            'score': entry.get('score', 0),
                            'location': entry.get('location', 'Unknown'),
                            'captures': caps
                        })
                except json.JSONDecodeError:
                    pass

    return jsonify(person_captures)


@app.route("/api/forensics/report/<name>")
@login_required
def generate_report(name):
    """Generate a PDF report for a specific person."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from io import BytesIO
    import datetime

    entries = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('name') == name:
                        entries.append(entry)
                except json.JSONDecodeError:
                    pass

    entries.sort(key=lambda x: x.get('timestamp', 0))

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
                                 fontSize=22, textColor=colors.black, spaceAfter=6*mm)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                    fontSize=11, textColor=colors.gray, spaceAfter=10*mm)
    heading_style = ParagraphStyle('SectionHead', parent=styles['Heading2'],
                                   fontSize=14, textColor=colors.black, spaceBefore=8*mm, spaceAfter=4*mm)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                fontSize=10, textColor=colors.black)

    story = []

    # Header
    story.append(Paragraph(f"Forensic Report: {name}", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        f"Total Detections: {len(entries)} · EagleEye Surveillance System",
        subtitle_style
    ))

    # Summary table
    if entries:
        story.append(Paragraph("Detection Summary", heading_style))

        first_ts = datetime.datetime.fromtimestamp(entries[0].get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
        last_ts = datetime.datetime.fromtimestamp(entries[-1].get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
        avg_score = sum(e.get('score', 0) for e in entries) / len(entries) if entries else 0
        locations = list(set(e.get('location', 'Unknown') for e in entries))

        summary_data = [
            ['Metric', 'Value'],
            ['First Seen', first_ts],
            ['Last Seen', last_ts],
            ['Total Detections', str(len(entries))],
            ['Average Confidence', f"{avg_score:.1%}"],
            ['Locations', ', '.join(locations)],
        ]

        t = Table(summary_data, colWidths=[50*mm, 100*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t)

    # Detection timeline
    story.append(Paragraph("Detection Timeline", heading_style))

    for i, entry in enumerate(entries[:50]):  # Limit to 50 entries
        ts = datetime.datetime.fromtimestamp(entry.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
        score = entry.get('score', 0)
        location = entry.get('location', 'Unknown')
        gps = entry.get('gps', {})
        gps_str = ""
        if gps and isinstance(gps, dict):
            gps_str = f" ({gps.get('lat', 0):.4f}, {gps.get('lng', 0):.4f})"

        story.append(Paragraph(
            f"<b>#{i+1}</b> · {ts} · <b>{score:.1%}</b> · {location}{gps_str}",
            body_style
        ))

        # Try to include face capture image
        caps = entry.get('captures', {})
        orig = caps.get('original', '')
        if orig:
            img_path = os.path.join(_PROJECT_ROOT, "frontend", orig)
            if os.path.exists(img_path):
                try:
                    img = RLImage(img_path, width=30*mm, height=30*mm)
                    story.append(img)
                except Exception:
                    pass

        story.append(Spacer(1, 3*mm))

    doc.build(story)
    buffer.seek(0)

    return Response(
        buffer.read(),
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment; filename=report_{name}.pdf'}
    )




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
