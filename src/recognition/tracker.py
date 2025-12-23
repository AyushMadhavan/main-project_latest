import numpy as np

class IOUTracker:
    def __init__(self, max_lost=5, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = {}  # id -> {bbox: [], lost: 0}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def update(self, detections):
        """
        Update tracks with new detections.
        detections: list of [x1, y1, x2, y2]
        Returns: list of (track_id, bbox)
        """
        # detections is a list of bboxes
        # tracks is a dict
        
        updated_tracks = []
        used_det_indices = set()
        
        # Match existing tracks
        for track_id, track in self.tracks.items():
            if track['lost'] > 0:
                # If lost, checking against all detections is valid
                pass
            
            # Predict (Constant velocity or just static for IOU)
            # For IOU tracker, we assume static or small movement
            last_bbox = track['bbox']
            
            best_iou = 0
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_det_indices:
                    continue
                
                iou = self._iou(last_bbox, det)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            if best_iou > self.iou_threshold:
                # Match found
                track['bbox'] = detections[best_det_idx]
                track['lost'] = 0
                used_det_indices.add(best_det_idx)
                updated_tracks.append((track_id, track['bbox']))
            else:
                # No match
                track['lost'] += 1
        
        # Create new tracks
        for i, det in enumerate(detections):
            if i not in used_det_indices:
                self.tracks[self.next_id] = {'bbox': det, 'lost': 0}
                updated_tracks.append((self.next_id, det))
                self.next_id += 1
                
        # Prune dead tracks
        self.tracks = {tid: t for tid, t in self.tracks.items() if t['lost'] < self.max_lost}
        
        return updated_tracks

    @staticmethod
    def _iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou
