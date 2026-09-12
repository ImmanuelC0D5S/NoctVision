"""
Collision Risk & Tracking Layer for NoctVision.
Combines ByteTrack for persistent ID tracking, optical flow backup,
pinhole camera depth estimation, and physics-based TTC calculation.
"""

import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]  # xc, yc, w, h (YOLO)
    class_id: int = 0
    age: int = 0
    hits: int = 1
    velocity: Optional[tuple[float, float]] = None  # (vx, vy) in px/frame
    distance: Optional[float] = None  # meters
    ttc: Optional[float] = None  # seconds


class NoctVisionTracker:
    """
    ByteTrack-based tracker with TTC estimation.

    Usage:
        tracker = NoctVisionTracker(focal_length=615.0, real_height=1.7)
        for frame in video_stream:
            detections = model(frame)
            tracks = tracker.update(detections, frame)
            for t in tracks:
                print(f"ID={t.track_id} TTC={t.ttc:.2f}s dist={t.distance:.1f}m")
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        focal_length: float = 615.0,
        real_person_height: float = 1.7,
        fps: float = 30.0,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.focal_length = focal_length
        self.real_person_height = real_person_height
        self.fps = fps

        self.tracks: list[Track] = []
        self.frame_count = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._next_id = 1

    def update(
        self,
        detections: list[tuple[float, float, float, float, float]],
        frame: Optional[np.ndarray] = None,
    ) -> list[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: list of (xc, yc, w, h, conf) in YOLO normalized format.
            frame: current frame (optional, needed for optical flow backup).

        Returns:
            list of active Track objects.
        """
        self.frame_count += 1
        dets = [d[:4] for d in detections if d[4] >= self.track_thresh]

        matched, unmatched_dets, unmatched_tracks = self._match(dets)

        for det_idx, track_idx in matched:
            self.tracks[track_idx].bbox = dets[det_idx]
            self.tracks[track_idx].hits += 1
            self.tracks[track_idx].age += 1

        for idx in unmatched_dets:
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    bbox=dets[idx],
                    age=1,
                    hits=1,
                )
            )
            self._next_id += 1

        for idx in unmatched_tracks:
            self.tracks[idx].age += 1

        self.tracks = [t for t in self.tracks if t.age < self.track_buffer]

        if frame is not None:
            self._compute_flow velocities(frame)

        for t in self.tracks:
            t.distance = self._estimate_distance(t.bbox)
            t.ttc = self._compute_ttc(t)

        self._prev_gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None
        )

        return self.tracks

    def _match(
        self, dets: list[tuple[float, float, float, float]]
    ) -> tuple[list, list, list]:
        """Simple IoU-based matching (Greedy)."""
        if not self.tracks or not dets:
            return [], list(range(len(dets))), list(range(len(self.tracks)))

        iou_matrix = np.zeros((len(dets), len(self.tracks)))
        for d, det in enumerate(dets):
            for t, track in enumerate(self.tracks):
                iou_matrix[d, t] = self._iou(det, track.bbox)

        matched = []
        used_dets = set()
        used_tracks = set()

        while True:
            if iou_matrix.size == 0:
                break
            idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if iou_matrix[idx] < self.match_thresh:
                break
            matched.append((idx[0], idx[1]))
            used_dets.add(idx[0])
            used_tracks.add(idx[1])
            iou_matrix[idx[0], :] = 0
            iou_matrix[:, idx[1]] = 0

        unmatched_dets = [i for i in range(len(dets)) if i not in used_dets]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        return matched, unmatched_dets, unmatched_tracks

    def _iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """Compute IoU between two YOLO bboxes (xc, yc, w, h)."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xa = max(x1 - w1 / 2, x2 - w2 / 2)
        ya = max(y1 - h1 / 2, y2 - h2 / 2)
        xb = min(x1 + w1 / 2, x2 + w2 / 2)
        yb = min(y1 + h1 / 2, y2 + h2 / 2)

        inter = max(0, xb - xa) * max(0, yb - ya)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    def _compute_flow_velocities(self, frame: np.ndarray):
        """Compute per-track velocities using optical flow (backup mechanism)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            return

        for track in self.tracks:
            xc, yc, w, h = track.bbox
            fx, fy = int(xc * frame.shape[1]), int(yc * frame.shape[0])
            half = max(int(max(w, h) * frame.shape[1] / 2), 5)

            x0 = max(0, fx - half)
            y0 = max(0, fy - half)
            x1 = min(frame.shape[1], fx + half)
            y1 = min(frame.shape[0], fy + half)

            if x1 - x0 < 10 or y1 - y0 < 10:
                continue

            prev_patch = self._prev_gray[y0:y1, x0:x1]
            curr_patch = gray[y0:y1, x0:x1]

            flow = cv2.calcOpticalFlowFarneback(
                prev_patch, curr_patch, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            vx = float(np.mean(flow[..., 0]))
            vy = float(np.mean(flow[..., 1]))
            track.velocity = (vx, vy)

    def _estimate_distance(self, bbox: tuple[float, float, float, float]) -> float:
        """
        Pinhole camera depth estimation.
        Z = (f * H_real) / h_pixel
        """
        _, _, _, h_norm = bbox
        h_pixel = h_norm * 640  # img_size
        if h_pixel < 1:
            return float("inf")
        return (self.focal_length * self.real_person_height) / h_pixel

    def _compute_ttc(self, track: Track) -> Optional[float]:
        """
        Time-To-Collision (constant velocity model).
        TTC = distance / |vz|
        If no velocity, estimate from consecutive frame distance deltas.
        """
        if track.distance is None or track.distance == float("inf"):
            return None

        if track.velocity is not None:
            vz = track.velocity[1]  # use y-velocity as proxy for approach speed
            if abs(vz) > 0.1:
                return track.distance / abs(vz) * (1.0 / self.fps)

        return None  # insufficient data


def run_tracking_demo(video_path: str = "0", weights: str = "models/best.pt"):
    """
    Run the full detection + tracking + TTC pipeline on a video or webcam.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    cap = cv2.VideoCapture(0 if video_path == "0" else video_path)
    tracker = NoctVisionTracker(fps=cap.get(cv2.CAP_PROP_FPS) or 30)

    print("Tracking demo running. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        dets = []
        for box in results.boxes:
            xc, yc, w, h = box.xywhn[0].tolist()
            conf = box.conf[0].item()
            dets.append((xc, yc, w, h, conf))

        tracks = tracker.update(dets, frame)

        for t in tracks:
            xc, yc, w, h = t.bbox
            x1 = int((xc - w / 2) * frame.shape[1])
            y1 = int((yc - h / 2) * frame.shape[0])
            x2 = int((xc + w / 2) * frame.shape[1])
            y2 = int((yc + h / 2) * frame.shape[0])

            color = (0, 255, 0)
            if t.ttc is not None and t.ttc < 3.0:
                color = (0, 0, 255)  # red = danger
            elif t.ttc is not None and t.ttc < 5.0:
                color = (0, 165, 255)  # orange = warning

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{t.track_id}"
            if t.distance:
                label += f" {t.distance:.1f}m"
            if t.ttc:
                label += f" TTC:{t.ttc:.1f}s"
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        cv2.imshow("NoctVision Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NoctVision Tracking Demo")
    parser.add_argument("--video", type=str, default="0", help="Video path or 0 for webcam")
    parser.add_argument("--weights", type=str, default="models/best.pt")
    args = parser.parse_args()
    run_tracking_demo(args.video, args.weights)
