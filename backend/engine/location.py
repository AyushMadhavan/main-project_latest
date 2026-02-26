import logging
import asyncio
import requests
import serial
import pynmea2

logger = logging.getLogger("Location")


def get_physical_gps(config):
    """Attempt to read from physical GPS sensor."""
    gps_conf = config.get('system', {}).get('gps_device', {})
    if not gps_conf.get('enabled', False):
        return None

    port = gps_conf.get('port', 'COM3')
    baud = gps_conf.get('baudrate', 9600)

    logger.info(f"Attempting to connect to GPS at {port}...")
    try:
        with serial.Serial(port, baud, timeout=2) as ser:
            for _ in range(20):
                line = ser.readline().decode('utf-8', errors='ignore')
                if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                    msg = pynmea2.parse(line)
                    if hasattr(msg, 'latitude') and msg.latitude != 0:
                        logger.info(f"Physical GPS Fix: {msg.latitude}, {msg.longitude}")
                        return {
                            'lat': msg.latitude,
                            'lng': msg.longitude,
                            'city': 'GPS Fixed'
                        }
    except Exception as e:
        logger.warning(f"Physical GPS failed: {e}")

    return None


async def get_windows_gps():
    """Access Windows Location API via WinSDK."""
    try:
        from winsdk.windows.devices.geolocation import Geolocator, GeolocationAccessStatus

        logger.info("Requesting Windows Location Access...")
        status = await Geolocator.request_access_async()

        if status == GeolocationAccessStatus.ALLOWED:
            logger.info("Access Granted. Acquiring signal...")
            locator = Geolocator()
            locator.desired_accuracy_in_meters = 10
            pos = await locator.get_geoposition_async()

            lat = pos.coordinate.point.position.latitude
            lng = pos.coordinate.point.position.longitude
            logger.info(f"Windows GPS Fix: {lat}, {lng}")
            return {'lat': lat, 'lng': lng, 'city': 'Windows GPS'}

        elif status == GeolocationAccessStatus.DENIED:
            logger.warning("Windows Location Access DENIED.")
        else:
            logger.warning(f"Windows Location Access Status: {status}")

    except ImportError:
        logger.warning("winsdk not installed. Skipping Windows GPS.")
    except Exception as e:
        logger.warning(f"Windows GPS Error: {e}")

    return None


def get_live_location(config):
    """Detect location with Physical GPS -> Windows GPS -> IP Failover."""

    # 1. Try Physical Serial GPS
    gps = get_physical_gps(config)
    if gps:
        return gps

    # 2. Try Windows Inbuilt GPS
    try:
        gps = asyncio.run(get_windows_gps())
        if gps:
            return gps
    except Exception as e:
        logger.warning(f"Async execution failed: {e}")

    # 3. IP-based Fallback
    providers = [
        ("http://ip-api.com/json", lambda d: (d['lat'], d['lon'], d.get('city', 'Unknown'))),
        ("https://ipinfo.io/json", lambda d: (float(d['loc'].split(',')[0]), float(d['loc'].split(',')[1]), d.get('city', 'Unknown'))),
        ("https://ipapi.co/json/", lambda d: (float(d['latitude']), float(d['longitude']), d.get('city', 'Unknown')))
    ]

    for url, parser in providers:
        try:
            logger.info(f"Detecting location via {url}...")
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                lat, lng, city = parser(data)
                logger.info(f"Location detected: {city} ({lat}, {lng})")
                return {'lat': lat, 'lng': lng, 'city': city}
        except Exception as e:
            logger.warning(f"Location fetch failed for {url}: {e}")

    return None
