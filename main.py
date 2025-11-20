import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from zones.zone_manager import RestrictedZoneManager
from detector.intrusion_detector import IntrusionDetector

def parse_args():
    parser = argparse.ArgumentParser(description="instrusion detection using Yolo and opencv")
    parser.add_argument("--source", type=str, default="test.mp4", help="Path to video file (or '0' for webcam)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or gpu")
    parser.add_argument("--zones", type=str, default="restricted_zones.json", help="Zones JSON file")
    return parser.parse_args()

def main():
    args = parse_args()
    zone_mgr = RestrictedZoneManager(args.zones)
    zone_mgr.load()

    print("[Model] Loading model:", args.weights)
    model = YOLO(args.weights)
    try:
        model.to(args.device)
    except Exception as e:
        print(f" Warning:could not set the device{args.device}: {e}.")

    detector = IntrusionDetector(model, zone_mgr)

    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("sorry cant open the source", args.source)
        return


    window_name = "Intrusion Detection"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, zone_mgr.on_mouse)

    paused = False
    last_frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("end of streen or cant read the frame .")
                break
            last_frame = frame.copy()
            frame = detector.process_frame(frame)
            cv2.imshow(window_name, frame)
        else:
            display = last_frame.copy() if last_frame is not None else np.zeros((480,640,3), dtype=np.uint8)
            zone_mgr.draw(display)
            cv2.putText(display, "PAUSED - drawing mode available. Press 'd' to toggle drawing, 'n' finish zone",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("p"): paused = not paused
        elif key == ord("d"): paused = True; zone_mgr.start_drawing(last_frame if last_frame is not None else np.zeros((480,640,3), dtype=np.uint8))
        elif key == ord("n"): zone_mgr.finish_current_zone(); detector.refresh_zones()
        elif key == ord("r"): zone_mgr.reset_zones(); detector.refresh_zones()
        elif key == ord("s"): zone_mgr.save()
        elif key == ord("l"): zone_mgr.load(); detector.refresh_zones()
        elif key == 27:
            if zone_mgr.drawing_mode: zone_mgr.cancel_drawing(); paused = False
            else: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
