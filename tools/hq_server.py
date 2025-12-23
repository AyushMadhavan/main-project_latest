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
        #sidebar { width: 300px; background: #222; border-right: 1px solid #444; padding: 20px; overflow-y: auto; }
        #map { flex: 1; }
        .alert { background: #333; padding: 10px; margin-bottom: 10px; border-left: 4px solid red; animation: flash 1s; }
        .alert.match { border-left-color: #0f0; }
        @keyframes flash { 0% { opacity: 0; } 100% { opacity: 1; } }
        h1 { font-size: 1.2rem; border-bottom: 2px solid #555; padding-bottom: 10px; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>HQ GLOBAL FEED</h1>
        <div id="feed">Waiting for signals...</div>
    </div>
    <div id="map"></div>

    <script>
        var map = L.map('map').setView([12.9716, 77.5946], 12);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap &copy; CARTO'
        }).addTo(map);

        var markers = {}; // id -> marker

        function update() {
            fetch('/api/global_stats')
                .then(r => r.json())
                .then(data => {
                    const feed = document.getElementById('feed');
                    feed.innerHTML = "";
                    
                    data.recent_logs.forEach(log => {
                        const div = document.createElement('div');
                        div.className = "alert " + (log.name == 'Unknown' ? '' : 'match');
                        div.innerHTML = `
                            <strong>${log.name}</strong><br>
                            <small>${new Date(log.timestamp*1000).toLocaleTimeString()}</small><br>
                            <em>${log.location} (${log.device_id})</em>
                        `;
                        feed.appendChild(div);
                    });

                    // Update Map Markers
                    data.locations.forEach(loc => {
                        const key = loc.device_id;
                        if(!markers[key]) {
                            markers[key] = L.marker([loc.gps.lat, loc.gps.lng]).addTo(map)
                                .bindPopup(`<b>${loc.location}</b><br>ID: ${loc.device_id}<br>Status: ONLINE`);
                        }
                    });
                });
        }
        setInterval(update, 2000);
    </script>
</body>
</html>
"""

with open("tools/hq_templates/index.html", "w") as f:
    f.write(HTML_TEMPLATE)

app = Flask(__name__, template_folder="hq_templates")
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# In-Memory Store
logs = deque(maxlen=100)
active_locations = {} # device_id -> {location_name, gps, last_seen}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ingest", methods=["POST"])
def ingest():
    data = request.json
    # data: {name, score, timestamp, location, device_id, gps: {lat, lng}}
    
    logs.appendleft(data)
    
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

if __name__ == "__main__":
    print("STARTING POLICE HQ SERVER ON PORT 8000...")
    app.run(host="0.0.0.0", port=8000)
