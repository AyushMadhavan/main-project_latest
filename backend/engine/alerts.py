"""
Email Alert System for high-confidence face detections.

Sends email notifications via SMTP when a known culprit is detected
with a score above the configured threshold. Includes per-person cooldown
to avoid alert fatigue.
"""

import os
import time
import logging
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

logger = logging.getLogger("Alerts")


class AlertManager:
    """Manages email alerts for face detections."""

    def __init__(self, config: dict):
        """
        Args:
            config: The 'alerts' section from settings.yaml
        """
        self.enabled = config.get('enabled', False)
        self.min_score = config.get('min_score', 0.6)
        self.cooldown = config.get('cooldown_seconds', 300)

        email_conf = config.get('email', {})
        self.smtp_host = email_conf.get('smtp_host', 'smtp.gmail.com')
        self.smtp_port = email_conf.get('smtp_port', 587)
        self.sender = email_conf.get('sender', '')
        self.password = email_conf.get('password', '')
        self.recipients = email_conf.get('recipients', [])

        # Cooldown tracker: { person_name: last_alert_timestamp }
        self._last_alert = {}
        self._lock = threading.Lock()

        if self.enabled and self.sender and self.password:
            logger.info(f"Alerts enabled → {self.smtp_host}:{self.smtp_port}, "
                        f"min_score={self.min_score}, cooldown={self.cooldown}s, "
                        f"recipients={self.recipients}")
        elif self.enabled:
            logger.warning("Alerts enabled but email credentials not configured. "
                           "Set ALERT_EMAIL_PASSWORD in .env and update settings.yaml.")
            self.enabled = False

    def check_and_alert(self, name, score, location, gps=None, capture_path=None, custom_emails=None):
        """
        Check if an alert should be sent and send it in a background thread.

        Args:
            name:          Detected person's name.
            score:         Detection confidence score.
            location:      Camera name / location string.
            gps:           GPS dict with 'lat' and 'lng'.
            capture_path:  Relative path to the face capture image.
            custom_emails: List of specific emails to alert instead of globals.
        """
        if not self.enabled:
            return

        if name == "Unknown":
            return

        if score < self.min_score:
            return

        # Cooldown check
        with self._lock:
            last = self._last_alert.get(name, 0)
            if time.time() - last < self.cooldown:
                return
            self._last_alert[name] = time.time()

        # Send in background thread
        threading.Thread(
            target=self._send_email,
            args=(name, score, location, gps, capture_path, custom_emails),
            daemon=True
        ).start()

    def _send_email(self, name, score, location, gps, capture_path, custom_emails):
        """Compose and send the alert email."""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Build GPS string
            gps_str = "N/A"
            maps_link = ""
            if gps and isinstance(gps, dict):
                lat = gps.get('lat', 0)
                lng = gps.get('lng', 0)
                city = gps.get('city', '')
                gps_str = f"{lat:.4f}, {lng:.4f}"
                if city:
                    gps_str += f" ({city})"
                maps_link = f"https://www.google.com/maps?q={lat},{lng}"

            subject = f"⚠ ALERT: {name} detected at {location} (Score: {score:.0%})"

            # HTML email body
            html = f"""
            <div style="font-family: -apple-system, sans-serif; max-width: 500px; margin: 0 auto; background: #111; color: #fff; border-radius: 12px; overflow: hidden;">
                <div style="background: #fff; color: #000; padding: 16px 24px;">
                    <strong>⚠ EagleEye Alert</strong>
                </div>
                <div style="padding: 24px;">
                    <h2 style="margin: 0 0 16px 0; font-weight: 500;">Subject Detected</h2>
                    <table style="width: 100%; border-collapse: collapse; color: #ccc;">
                        <tr><td style="padding: 8px 0; color: #666;">Name</td><td style="padding: 8px 0; font-weight: 600; color: #fff;">{name}</td></tr>
                        <tr><td style="padding: 8px 0; color: #666;">Confidence</td><td style="padding: 8px 0;">{score:.1%}</td></tr>
                        <tr><td style="padding: 8px 0; color: #666;">Location</td><td style="padding: 8px 0;">{location}</td></tr>
                        <tr><td style="padding: 8px 0; color: #666;">GPS</td><td style="padding: 8px 0;">{gps_str}</td></tr>
                        <tr><td style="padding: 8px 0; color: #666;">Time</td><td style="padding: 8px 0;">{timestamp}</td></tr>
                    </table>
                    {"<p style='margin-top: 16px;'><a href='" + maps_link + "' style='color: #999;'>View on Google Maps →</a></p>" if maps_link else ""}
                </div>
                <div style="padding: 12px 24px; background: #0a0a0a; color: #444; font-size: 12px;">
                    EagleEye Surveillance System · Automated Alert
                </div>
            </div>
            """

            target_recipients = custom_emails if custom_emails else self.recipients
            if not target_recipients:
                logger.error(f"Cannot send alert for {name} — no recipients configured.")
                return

            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = ', '.join(target_recipients)
            msg['Subject'] = subject
            msg.attach(MIMEText(html, 'html'))

            # Attach face capture image if available
            if capture_path:
                abs_path = os.path.join("frontend", capture_path) if not os.path.isabs(capture_path) else capture_path
                if os.path.exists(abs_path):
                    with open(abs_path, 'rb') as img_file:
                        img_data = img_file.read()
                    img_attachment = MIMEImage(img_data)
                    img_attachment.add_header('Content-Disposition', 'attachment', filename=f'{name}_capture.jpg')
                    msg.attach(img_attachment)

            # Send
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(self.sender, target_recipients, msg.as_string())

            logger.info(f"Alert sent for '{name}' (score={score:.2f}) → {target_recipients}")

        except Exception as e:
            logger.error(f"Failed to send alert for '{name}': {e}")
