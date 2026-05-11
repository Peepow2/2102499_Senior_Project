import cv2
import math
import numpy as np
from collections import defaultdict, deque


backSub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=80, detectShadows=False)

# ------------------------------------------------------------ #

def draw_direction(raw_frame, Vector_Road):
    overlay = raw_frame.copy()
    for vec in Vector_Road:
        Vec_color = (255, 0, 255)
        if len(vec) == 2:
            p1, p2 = vec
            cv2.line(overlay, p1, p2, Vec_color, 10, cv2.LINE_AA)
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            size = 50
            spread = 0.5
            tp1 = (int(p2[0] - size * math.cos(angle - spread)),
                int(p2[1] - size * math.sin(angle - spread)))
            tp2 = (int(p2[0] - size * math.cos(angle + spread)),
                int(p2[1] - size * math.sin(angle + spread)))
            triangle_pts = np.array([p2, tp1, tp2], np.int32)
            cv2.fillPoly(overlay, [triangle_pts], Vec_color)
    alpha = 0.7
    beta = 0.5
    cv2.addWeighted(overlay, beta, raw_frame, alpha, 0, raw_frame)
    return raw_frame


def preprocessing(frame, RoI, Vector_Road):
    w, h = 960, 540
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in RoI:
        cv2.fillPoly(mask, [np.array(r)], 255)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    frame = draw_direction(result, Vector_Road)
    return frame


def get_centroid(x, y, w, h):
    return (int(x + w/2), int(y + h/2))


# ============================================================ #
# DIRECTION UTILS
# ============================================================ #

def calc_angle_deg(p1, p2):
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]
    angle = math.atan2(dy, dx) * 180 / math.pi
    return angle % 360


def angle_diff(a, b):
    return abs(a - b) % 360


def get_roi_angles(Direction):
    return [calc_angle_deg(vec[0], vec[1]) for vec in Direction]


def check_direction(move_angle, roi_angles, angle_tolerance):
    for idx, ref_angle in enumerate(roi_angles):
        if angle_diff(move_angle, ref_angle) <= angle_tolerance:
            return True, idx
    return False, -1


# ============================================================ #
# LIGHT SPOT DETECTION + GROUPING
# ============================================================ #

def detect_light_spots(frame, brightness_threshold, min_area, max_area):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN,  kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            spots.append((x + w // 2, y + h // 2, area))

    return spots, bright_mask


def group_spots_into_vehicles(spots, group_distance=80):
    if not spots:
        return []

    used = [False] * len(spots)
    vehicles = []

    for i, (cx, cy, area) in enumerate(spots):
        if used[i]:
            continue

        group = [(cx, cy, area)]
        used[i] = True

        for j, (cx2, cy2, area2) in enumerate(spots):
            if used[j]:
                continue
            if math.hypot(cx - cx2, cy - cy2) < group_distance:
                group.append((cx2, cy2, area2))
                used[j] = True

        avg_cx = int(np.mean([s[0] for s in group]))
        avg_cy = int(np.mean([s[1] for s in group]))
        pad = 25
        x_min = max(0,   min(s[0] for s in group) - pad)
        y_min = max(0,   min(s[1] for s in group) - pad)
        x_max = min(959, max(s[0] for s in group) + pad)
        y_max = min(539, max(s[1] for s in group) + pad)

        vehicles.append({
            "spots":    group,
            "centroid": (avg_cx, avg_cy),
            "bbox":     (x_min, y_min, x_max, y_max),
            "count":    len(group)
        })

    return vehicles


def classify_vehicle(spot_count):
    if spot_count == 1:
        return "motorcycle"
    elif spot_count == 2:
        return "car"
    else:
        return "other"


# ============================================================ #
# TRACKING + DIRECTION FILTER
# ============================================================ #

def match_vehicles(current_vehicles, prev_centroids, object_id_count, max_distance=100):
    """
    จับคู่ยานพาหนะกับ frame ก่อนหน้า (best-match)
    คืนค่า: new_prev_centroids, object_id_count, id_map
    """
    new_prev_centroids = {}
    id_map = {}
    assigned = set()

    for i, v in enumerate(current_vehicles):
        cx, cy = v["centroid"]
        best_id, best_dist = None, float("inf")

        for obj_id, (px, py) in prev_centroids.items():
            if obj_id in assigned:
                continue
            dist = math.hypot(cx - px, cy - py)
            if dist < max_distance and dist < best_dist:
                best_dist = dist
                best_id = obj_id

        if best_id is not None:
            new_prev_centroids[best_id] = (cx, cy)
            id_map[i] = best_id
            assigned.add(best_id)
        else:
            new_prev_centroids[object_id_count] = (cx, cy)
            id_map[i] = object_id_count
            object_id_count += 1

    return new_prev_centroids, object_id_count, id_map


# ============================================================ #
# DRAW
# ============================================================ #

def draw_vehicle(frame, vehicle, obj_id, label, move_angle, is_valid_dir):
    color = (0,0,255) if is_valid_dir else (0,255,0)
    x1, y1, x2, y2 = vehicle["bbox"]

    if not is_valid_dir:
        cv2.imwrite(f"Save file Path", frame[y1-20:y2+20,x1-20:x2+20])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    for (sx, sy, _) in vehicle["spots"]:
        cv2.circle(frame, (sx, sy), 5, (0, 255, 255), -1)

    # บรรทัดบน: ID + ประเภท + จำนวนจุดไฟ
    label_text = f"ID:{obj_id} {label}({vehicle['count']})"
    cv2.putText(frame, label_text, (x1, max(y1 - 20, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    


# ============================================================ #
# MAIN LOOP
# ============================================================ #

def detect(RoI, Direction, cap):
    prev_centroids  = {}          # {obj_id: (cx, cy)}
    object_id_count = 0
    max_distance    = 100

    # เก็บ trajectory สำหรับคำนวณทิศทาง (smoothing)
    # {obj_id: deque of (cx, cy)}
    tracks = defaultdict(lambda: deque(maxlen=20))

    counter = {"car": 0, "motorcycle": 0, "other": 0}
    counted_ids = set()

    # คำนวณมุมอ้างอิงจาก Direction vector ครั้งเดียว
    roi_angles = get_roi_angles(Direction)

    ANGLE_TOLERANCE = 60
    MIN_TRACK_LEN   = 3

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        raw_frame = cv2.resize(raw_frame, (960, 540))
        frame     = preprocessing(raw_frame, RoI, Direction)

        # ── STEP 1: ตรวจจับจุดสว่าง ──────────────────────────
        spots, bright_mask = detect_light_spots(frame, brightness_threshold=180, min_area = 200, max_area = 600)

        # ── STEP 2: จัดกลุ่มเป็นยานพาหนะ ────────────────────
        vehicles = group_spots_into_vehicles(spots, group_distance=80)

        # ── STEP 3: Tracking ──────────────────────────────────
        prev_centroids, object_id_count, id_map = match_vehicles(
            vehicles, prev_centroids, object_id_count, max_distance
        )

        # ── STEP 4: คำนวณทิศทาง + วาด + นับ ─────────────────
        for i, vehicle in enumerate(vehicles):
            obj_id     = id_map[i]
            label      = classify_vehicle(vehicle["count"])
            cx, cy     = vehicle["centroid"]

            # อัปเดต trajectory
            tracks[obj_id].append((cx, cy))
            track = tracks[obj_id]

            move_angle   = None
            is_valid_dir = True   # default: ยังไม่รู้ทิศ
            matched_roi  = -1

            if len(track) >= MIN_TRACK_LEN:
                # ใช้จุดแรกสุดใน buffer กับจุดปัจจุบัน (smooth กว่าแบบ frame-to-frame)
                x_start, y_start = track[-3]
                x_end, y_end = track[-1]
                move_angle = calc_angle_deg((x_start, y_start), (x_end, y_end))

                # ตรวจว่าเคลื่อนไหวจริง (กัน noise กรณีรถจอดนิ่ง)
                dy = cx - x_start
                dx = cy - y_start
                dist_moved = math.hypot(dx, dy)
                if 10 <= dist_moved <= 300 and (abs(dx) > 10 or abs(dy) > 10):
                    is_valid_dir, matched_roi = check_direction(
                        move_angle, roi_angles, ANGLE_TOLERANCE
                    )

            # นับเฉพาะที่เคลื่อนที่ผิดทิศทาง
            if not is_valid_dir and obj_id not in counted_ids:
                counter[label] += 1
                counted_ids.add(obj_id)

            draw_vehicle(frame, vehicle, obj_id, label, move_angle, is_valid_dir)


        cv2.imshow("Vehicle Tracking System (Night)", frame)
        #cv2.imshow("Bright Mask", bright_mask)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    
    print(counter)

    cap.release()
    cv2.destroyAllWindows()
