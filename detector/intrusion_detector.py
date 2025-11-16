import time
import numpy as np
import cv2
from typing import List, Tuple, Dict
from ultralytics import YOLO
from zones.zone_manager import RestrictedZoneManager
from detector.utils import point_in_polygon, centroid_from_xyxy
import pyautogui
from PIL import Image
import os


Point = Tuple[int, int]
ALARM_SILENCE_SECONDS = 3.0
PERSON_CLASS_ID = 0

class IntrusionDetector:
    def __init__(self, model: YOLO, zone_manager: RestrictedZoneManager):
        self.model = model
        self.zone_manager = zone_manager
        self.zone_states: List[Dict] = []
        self._reset_zone_states()

    def _reset_zone_states(self):
        self.zone_states = [{"alarm_on": False, "last_inside_time": None} for _ in self.zone_manager.zones]

    def refresh_zones(self):
        self._reset_zone_states()

    def process_frame(self, frame) -> np.ndarray:
        results = self.model(frame, verbose=False)[0]

        persons_centroids: List[Point] = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy_arr = results.boxes.xyxy.cpu().numpy()
            conf_arr = results.boxes.conf.cpu().numpy()
            cls_arr = results.boxes.cls.cpu().numpy()

            for xyxy, conf, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
                if int(cls_id) != PERSON_CLASS_ID:
                    continue
                cx, cy = centroid_from_xyxy(xyxy)
                persons_centroids.append((cx, cy))
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if len(self.zone_states) != len(self.zone_manager.zones):
            self.refresh_zones()

        now = time.time()
        for idx, zone in enumerate(self.zone_manager.zones):
            inside_any = any(point_in_polygon(c, zone) for c in persons_centroids)
            state = self.zone_states[idx]
            if inside_any:
                state["alarm_on"] = True
                state["last_inside_time"] = now
            else:
                last = state["last_inside_time"] or 0.0
                if state["alarm_on"] and (now - last) >= ALARM_SILENCE_SECONDS:
                    state["alarm_on"] = False
                    state["last_inside_time"] = None
            if "last_capture" not in state:
                state["last_capture"] = 0        
            if state["alarm_on"]:
                current_time = time.time()
                pts = np.array(zone, dtype=np.int32)
                cen = np.mean(pts, axis=0).astype(int)
                cv2.putText(frame, "ALarm! Leave", (int(cen[0])-40, int(cen[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),3)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0,0,255))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                if current_time - state["last_capture"] >= 5:
                    state["last_capture"] = current_time
                    folder = "screenshots"
                    os.makedirs(folder, exist_ok=True)
                    # Unique filename using timestamp
                    filename = f"alarm_{int(current_time)}.png"
                    path = os.path.join(folder, filename)
                    screenshot = pyautogui.screenshot()

                        # Resize to 1080x720
                    screenshot = screenshot.resize((1080, 720), Image.LANCZOS)

                        # Save
                    screenshot.save(path)
                    print("Screenshot saved:", path)


        self.zone_manager.draw(frame)
        cv2.putText(frame, "Press 'd' draw zones, 'l' load, 's' save, 'r' reset zones, 'q' quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return frame
