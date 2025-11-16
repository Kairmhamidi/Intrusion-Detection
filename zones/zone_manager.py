import json
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

Point = Tuple[int, int]
Polygon = List[Point]

DEFAULT_ZONES_FILE = "restricted_zones.json"

class RestrictedZoneManager:
    def __init__(self, zones_file: str = DEFAULT_ZONES_FILE):
        self.zones_file = Path(zones_file)
        self.zones: List[Polygon] = []
        self.drawing_mode = False
        self.current_pts: Polygon = []
        self._draw_frame = None

    def load(self):
        if not self.zones_file.exists():
            print(f"[Zones] {self.zones_file} not found, starting with no zones.")
            self.zones = []
            return
        with open(self.zones_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.zones = [[(int(p[0]), int(p[1])) for p in zone] for zone in data]
        print(f"[Zones] Loaded {len(self.zones)} zones from {self.zones_file}")

    def save(self):
        data = [[[int(p[0]), int(p[1])] for p in zone] for zone in self.zones]
        with open(self.zones_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Zones] Saved {len(self.zones)} zones to {self.zones_file}")

    # --- Drawing methods ---
    def start_drawing(self, frame):
        self.drawing_mode = True
        self.current_pts = []
        self._draw_frame = frame.copy()
        print("[Zones] Entered drawing mode.")

    def finish_current_zone(self):
        if len(self.current_pts) >= 3:
            self.zones.append(self.current_pts.copy())
            print(f"[Zones] Finished zone with {len(self.current_pts)} points. Total zones: {len(self.zones)}")
            self.current_pts = []
        else:
            print("[Zones] Need at least 3 points to finish a zone.")

    def cancel_drawing(self):
        self.drawing_mode = False
        self.current_pts = []
        self._draw_frame = None
        print("[Zones] Cancelled drawing mode.")

    def reset_zones(self):
        self.zones = []
        print("[Zones] All zones cleared.")

    # --- Mouse & drawing ---
    def on_mouse(self, event, x, y, flags, param):
        if not self.drawing_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_pts.append((x, y))
            print(f"[Zones] Added point: {(x, y)}")
        elif event == cv2.EVENT_RBUTTONDOWN and self.current_pts:
            removed = self.current_pts.pop()
            print(f"[Zones] Removed last point: {removed}")

    def draw(self, frame):
        for idx, zone in enumerate(self.zones):
            pts = np.array(zone, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
            centroid = np.mean(pts, axis=0).astype(int)
            cv2.putText(frame, f"Zone {idx+1}", (int(centroid[0]), int(centroid[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Draw current points in drawing mode
        if self.drawing_mode and self._draw_frame is not None:
            overlay = frame
            for i, p in enumerate(self.current_pts):
                cv2.circle(overlay, p, 4, (0, 255, 255), -1)
                if i > 0:
                    cv2.line(overlay, self.current_pts[i-1], self.current_pts[i], (0, 255, 255), 2)
