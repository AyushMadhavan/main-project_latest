from flask import Flask, jsonify, request, render_template
import logging
import time
from collections import deque
import json
import os

# Create templates dir for HQ
if not os.path.exists("tools/hq_templates"):
    os.makedirs("tools/hq_templates")

# HQ HTML
# HQ HTML
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>POLICE HQ - EAGLE EYE GLOBAL</title>
    <!-- Leaflet -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { background: #111; color: #eee; font-family: sans-serif; margin: 0; display: flex; height: 100vh; }
        #sidebar { width: 350px; background: #222; border-right: 1px solid #444; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; }
        #map { flex: 1; }
        .alert { background: #333; padding: 10px; margin-bottom: 10px; border-left: 4px solid red; animation: flash 1s; cursor: pointer; transition: background 0.2s; }
        .alert:hover { background: #444; }
        .alert.match { border-left-color: #0f0; }
        @keyframes flash { 0% { opacity: 0; } 100% { opacity: 1; } }
        h1 { font-size: 1.2rem; border-bottom: 2px solid #555; padding-bottom: 10px; margin-top: 0; }
        .controls { margin-bottom: 20px; text-align: right; }
        button { background: #555; color: white; border: none; padding: 5px 10px; cursor: pointer; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>HQ GLOBAL FEED</h1>
        <div style="font-size: 0.8rem; color: #aaa; margin-bottom: 15px;">Click any alert to track suspect movement.</div>
        <div id="feed">Waiting for signals...</div>
    </div>
    <div id="map"></div>

    <script>
        var map = L.map('map').setView([12.9716, 77.5946], 12);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap &copy; CARTO'
        }).addTo(map);

        var markers = {}; // id -> marker
        var pathLayer = L.layerGroup().addTo(map);

        function trackPerson(name) {
            console.log("Tracking:", name);
            if(name === 'Unknown') return;
            
            fetch('/api/history/' + encodeURIComponent(name))
                .then(r => r.json())
                .then(route => {
                    pathLayer.clearLayers();
                    
                    if (!route || route.length === 0) {
                        alert('No GPS history found for ' + name);
                        return;
                    }
                    
                    const latlngs = route.map(p => [p.lat, p.lng]);
                    
                    // Draw Line
                    L.polyline(latlngs, {color: 'cyan', weight: 4, opacity: 0.8, dashArray: '10, 10'}).addTo(pathLayer);
                    
                    // Markers
                    if (latlngs.length > 0) {
                        L.circleMarker(latlngs[0], {color: 'green', radius: 6}).addTo(pathLayer)
                            .bindPopup(`<b>Start</b><br>${new Date(route[0].timestamp * 1000).toLocaleTimeString()}`);
                        
                        L.circleMarker(latlngs[latlngs.length - 1], {color: 'red', radius: 6}).addTo(pathLayer)
                            .bindPopup(`<b>End</b><br>${new Date(route[route.length-1].timestamp * 1000).toLocaleTimeString()}`);
                            
                        map.fitBounds(L.polyline(latlngs).getBounds(), {padding: [50, 50]});
                    }
                });
        }

        function update() {
            fetch('/api/global_stats')
                .then(r => r.json())
                .then(data => {
                    const feed = document.getElementById('feed');
                    feed.innerHTML = "";
                    
                    data.recent_logs.forEach(log => {
                        const div = document.createElement('div');
                        div.className = "alert " + (log.name == 'Unknown' ? '' : 'match');
                        div.onclick = () => trackPerson(log.name);
                        div.innerHTML = `
                            <strong>${log.name}</strong> <span style="font-size:0.8em; float:right">${(log.score*100).toFixed(0)}%</span><br>
                            <small>${new Date(log.timestamp*1000).toLocaleTimeString()}</small><br>
                            <em>${log.location} (${log.device_id})</em>
                        `;
                        feed.appendChild(div);
                    });

                    // Update Map Live Markers
                    data.locations.forEach(loc => {
                        const key = loc.device_id;
                        if(loc.gps.lat !== 0) {
                            if(!markers[key]) {
                                markers[key] = L.marker([loc.gps.lat, loc.gps.lng]).addTo(map)
                                    .bindPopup(`<b>${loc.location}</b><br>ID: ${loc.device_id}<br>Status: ONLINE`);
                            } else {
                                markers[key].setLatLng([loc.gps.lat, loc.gps.lng]);
                            }
                        }
                    });
                });
        }
        setInterval(update, 2000);
        update();
    </script>
</body>
</html>
"""

with open("tools/hq_templates/index.html", "w") as f:
    f.write(HTML_TEMPLATE)

app = Flask(__name__, template_folder="hq_templates")
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# In-Memory Store & Persistence
HQ_HISTORY_FILE = "hq_history.json"
logs = deque(maxlen=200) # Increased size
active_locations = {} # device_id -> {location_name, gps, last_seen}

# Load history on startup
if os.path.exists(HQ_HISTORY_FILE):
    try:
        with open(HQ_HISTORY_FILE, 'r') as f:
            saved_data = json.load(f)
            # Restore logs (last 200)
            for entry in saved_data:
                logs.appendleft(entry)
    except Exception as e:
        print(f"Failed to load history: {e}")

def save_entry(entry):
    """Append entry to history file."""
    try:
        # Simple append-only or load-save. ideally append-only JSONL but user used JSON list.
        # For simplicity in this demo, we will use JSONL (Line Delimited) for performance/stability
        # But to be compatible with dashboard logic, let's just append to the in-memory and periodically save?
        # Or simple append to a file.
        
        # Let's use JSONL for the file to avoid reading the whole thing every write
        with open(HQ_HISTORY_FILE, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Save failed: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ingest", methods=["POST"])
def ingest():
    data = request.json
    # data: {name, score, timestamp, location, device_id, gps: {lat, lng}}
    
    logs.appendleft(data)
    save_entry(data)
    
    if 'device_id' in data:
        active_locations[data['device_id']] = {
            "device_id": data['device_id'],
            "location": data.get('location', 'Unknown'),
            "gps": data.get('gps', {'lat':0, 'lng':0}),
            "last_seen": time.time()
        }
    
    print(f"[HQ] Receive from {data.get('location')}: {data.get('name')}")
    return jsonify({"status": "ok"})

@app.route("/api/global_stats")
def stats():
    return jsonify({
        "recent_logs": list(logs),
        "locations": list(active_locations.values())
    })

@app.route('/api/history/<name>')
def get_history(name):
    """Get movement history for a specific person from HQ history."""
    try:
        track = []
        if os.path.exists(HQ_HISTORY_FILE):
            with open(HQ_HISTORY_FILE, 'r') as f:
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
                    except:
                         pass
        
        track.sort(key=lambda x: x['timestamp'])
        return jsonify(track)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("STARTING POLICE HQ SERVER ON PORT 8000...")
    app.run(host="0.0.0.0", port=8000)
