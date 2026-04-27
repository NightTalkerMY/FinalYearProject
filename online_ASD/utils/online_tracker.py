import numpy as np

def calculate_iou(boxA, boxB):
    # Ensure we only use the first 4 elements [x1, y1, x2, y2] 
    # just in case S3FD passes the confidence score [x1, y1, x2, y2, conf]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Avoid division by zero
    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0:
        return 0.0
        
    iou = interArea / denominator
    return iou

class OnlineTracker:
    # Added alpha=0.3. Lower alpha = more smoothing, less jitter.
    def __init__(self, iou_threshold=0.4, max_lost=5, alpha=0.8):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.alpha = alpha 
        self.active_tracks = {}  
        self.next_id = 0

    def update(self, detections):
        matched_track_ids = set()
        new_tracks = {}

        for det in detections:
            best_iou = 0
            best_id = -1
            
            for track_id, track_info in self.active_tracks.items():
                if track_id in matched_track_ids:
                    continue 
                    
                iou = calculate_iou(det, track_info['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id
                    
            if best_iou > self.iou_threshold:
                # --- EMA SMOOTHING FIX ---
                old_bbox = np.array(self.active_tracks[best_id]['bbox'])
                new_bbox = np.array(det)
                smoothed_bbox = self.alpha * new_bbox + (1 - self.alpha) * old_bbox
                
                new_tracks[best_id] = {'bbox': smoothed_bbox.tolist(), 'lost_count': 0}
                matched_track_ids.add(best_id)
            else:
                new_tracks[self.next_id] = {'bbox': det, 'lost_count': 0}
                self.next_id += 1

        for track_id, track_info in self.active_tracks.items():
            if track_id not in matched_track_ids:
                track_info['lost_count'] += 1
                if track_info['lost_count'] <= self.max_lost:
                    new_tracks[track_id] = track_info

        self.active_tracks = new_tracks
        return {tid: info['bbox'] for tid, info in self.active_tracks.items()}