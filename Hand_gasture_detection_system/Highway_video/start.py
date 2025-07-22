import cv2
from scipy.spatial import distance

# === Tracker Class ===
class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        new_center_points = {}

        for rect in objects_rect:
            x, y, w, h = rect
            cx = x + w // 2
            cy = y + h // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = distance.euclidean((cx, cy), pt)
                if dist < 35:
                    new_center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                new_center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

# === Initialize tracker ===
tracker = EuclideanDistTracker()

# === Load video ===
cap = cv2.VideoCapture("cars.mp4")

# Optional: Resize output video (for speed)
frame_width = 800
frame_height = 450

# === Create background subtractor ===
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

# Frame counter
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    frame_count += 1

    # === Apply background subtraction ===
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)

    # === Find contours ===
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 600:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, w, h))

    # === Update tracker ===
    boxes_ids = tracker.update(detections)

    for box in boxes_ids:
        x, y, w, h, obj_id = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # === Overlay Info ===
    cv2.putText(frame, "Tracking Vehicles...", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)

    # === Display ===
    cv2.imshow("Mask", mask)
    cv2.imshow("Tracking", frame)

    # === Slow down playback: 100 ms delay (~10 FPS) ===
    if cv2.waitKey(100) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
