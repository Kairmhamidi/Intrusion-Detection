import cv2
import numpy as np
from typing import List, Tuple

Point = Tuple[int, int]
Polygon = List[Point]

def point_in_polygon(pt: Point, polygon: Polygon) -> bool:
    """Return True if pt is inside polygon (inclusive)."""
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), pt, False) >= 0

def centroid_from_xyxy(xyxy: List[float]) -> Point:
    """Compute integer centroid from [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = xyxy
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy
