import threading
import queue
import cv2
import time
import logging
import os
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamLoader")

class StreamLoader:
    def __init__(self, source: str | int, queue_size: int = 5, width: int = 1280, height: int = 720):
        """
        Initialize the StreamLoader.
        
        Args:
            source: RTSP URL or camera index.
            queue_size: Maximum number of frames to hold in the queue.
            width: Desired frame width.
            height: Desired frame height.
        """
        self.source = source
        self.queue_size = queue_size
        self.width = width
        self.height = height
        
        # Shared queue for frames: (frame_id, frame_timestamp, frame_data)
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start the frame ingestion process."""
        self.running = True
        self.thread = threading.Thread(target=self._update, args=(self.source, self.width, self.height))
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"StreamLoader started for source: {self.source}")

    def stop(self):
        """Stop the frame ingestion process."""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("StreamLoader stopped.")

    def read(self) -> Tuple[bool, Optional[float], Optional[object]]:
        """
        Get the latest frame from the queue.
        
        Returns:
            Tuple containing:
            - success (bool): True if a frame was retrieved.
            - timestamp (float): Time the frame was captured.
            - frame (numpy.ndarray): The image data.
        """
        if not self.frame_queue.empty():
            return True, *self.frame_queue.get()
        return False, None, None

    def _update(self, source, width, height):
        """Internal thread loop to fetch frames."""
        print(f"[DEBUG] StreamLoader thread started for {source}")
        
        def open_cam(src):
            if isinstance(src, int) and os.name == 'nt':
                return cv2.VideoCapture(src, cv2.CAP_DSHOW)
            return cv2.VideoCapture(src)

        cap = open_cam(source)

        if not cap.isOpened():
            logger.error(f"Failed to open stream: {source}")
            print(f"[ERROR] Failed to open stream: {source}")
            self.running = False
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        frame_count = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] Failed read from {source}. Reconnecting...")
                cap.release()
                time.sleep(1)
                cap = open_cam(source)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                continue

            timestamp = time.time()
            if frame_count % 100 == 0:
                 pass # Too noisy, silencing for now
            frame_count += 1
            
            # Non-blocking put with frame dropping behavior
            if self.frame_queue.full():
                try:
                    _ = self.frame_queue.get_nowait()  # Drop oldest frame
                except Exception:
                    pass
            
            try:
                self.frame_queue.put((timestamp, frame), block=False)
            except Exception:
                pass

        cap.release()
        print(f"[DEBUG] StreamLoader thread stopped for {source}")
