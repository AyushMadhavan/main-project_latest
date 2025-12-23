import requests
import yaml
import os

def get_ip_location():
    providers = [
        ("http://ip-api.com/json", lambda d: (d['lat'], d['lon'], d.get('city', 'Unknown'))),
        ("https://ipinfo.io/json", lambda d: (float(d['loc'].split(',')[0]), float(d['loc'].split(',')[1]), d.get('city', 'Unknown'))),
        ("https://ipapi.co/json/", lambda d: (float(d['latitude']), float(d['longitude']), d.get('city', 'Unknown')))
    ]

    for url, parser in providers:
        try:
            print(f"Trying {url}...")
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return parser(data)
        except Exception as e:
            print(f"Failed {url}: {e}")
            
    return None, None, None

def update_settings(lat, lng, city):
    path = "settings.yaml"
    with open(path, 'r') as f:
        content = f.read()
    
    # Simple replace to match the structure in settings.yaml
    # We load it safely to not break comments if possible, but PyYAML doesn't preserve comments well.
    # So we used string replacement or a comment-preserving parser if available.
    # Since we control the formatting closely, read/replace is risky if formatting changed.
    # Let's use string replacement for safety of comments.
    
    # Read file lines
    with open(path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    in_gps = False
    for line in lines:
        if "gps:" in line:
            in_gps = True
            new_lines.append(line)
            continue
            
        if in_gps:
            if "lat:" in line:
                # Keep indentation
                indent = line.split("lat:")[0]
                new_lines.append(f"{indent}lat: {lat}\n")
                continue
            if "lng:" in line:
                indent = line.split("lng:")[0]
                new_lines.append(f"{indent}lng: {lng} # {city}\n")
                in_gps = False # Done updating this block
                continue
        
        new_lines.append(line)
        
    with open(path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Updated settings.yaml with location: {city} ({lat}, {lng})")

if __name__ == "__main__":
    print("Fetching IP-based location...")
    lat, lng, city = get_ip_location()
    if lat and lng:
        update_settings(lat, lng, city)
    else:
        print("Could not detect location. Please manually edit settings.yaml.")
