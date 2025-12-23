import multiprocessing as mp
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
        self.frame_queue = mp.Queue(maxsize=queue_size)
        self.running = mp.Value('b', True)
        self.process: Optional[mp.Process] = None

    def start(self):
        """Start the frame ingestion process."""
        self.running.value = True
        self.process = mp.Process(target=self._update, args=(self.running, self.frame_queue, self.source, self.width, self.height))
        self.process.daemon = True
        self.process.start()
        logger.info(f"StreamLoader started for source: {self.source}")

    def stop(self):
        """Stop the frame ingestion process."""
        self.running.value = False
        if self.process:
            self.process.join()
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

    @staticmethod
    def _update(running, frame_queue, source, width, height):
        """Internal process loop to fetch frames."""
        # On Windows, using cv2.CAP_DSHOW can be more stable for webcams
        if isinstance(source, int) and os.name == 'nt':
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error(f"Failed to open stream: {source}")
            running.value = False
            return

        # Set resolution if possible (mostly for webcams, RTSP usually ignores this)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        while running.value:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(source)
                continue

            timestamp = time.time()
            
            # Non-blocking put with frame dropping behavior
            if frame_queue.full():
                try:
                    _ = frame_queue.get_nowait()  # Drop oldest frame
                except Exception:
                    pass  # Queue might have been emptied by consumer simultaneously

            try:
                frame_queue.put((timestamp, frame), block=False)
            except Exception:
                pass  # Queue full again, just skip this frame

        cap.release()
