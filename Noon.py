import cv2
import math
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

ANGLE_TOLERANCE = 60
MIN_TRACK_LEN   = 3


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

def detect(RoI, Direction, cap):
    #model = YOLO("yolo11n.pt")
    #model = YOLO("yolo11m.pt")
    model = YOLO("yolo11l.pt")

    fps = cap.get(cv2.CAP_PROP_FPS)
    T = int(max(1, fps//30)) * 5

    # trajectory storage
    tracks = defaultdict(lambda: deque(maxlen=20))
    counter = {
        "car": 0,
        "motorcycle": 0,
        "other": 0
    }

    Roi_theta = [int(math.atan2(vec[0][1]-vec[1][1], vec[1][0]-vec[0][0]) * 180 / math.pi) % 360 for vec in Direction]
    wrong_way_ids = set()
    frame_count = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        raw_frame = cv2.resize(raw_frame, (960, 540))
        frame = preprocessing(raw_frame, RoI, Direction)
        #cv2.imshow("Wrong Way Detection", frame)
        #cv2.waitKey(10)

        results = model.track(frame, persist = True, tracker = "bytetrack.yaml", conf = 0.25, classes = [2, 3, 5, 7], device = 0) 
        #results = model.track(frame, persist = True, tracker = "botsort.yaml", conf = 0.25, classes = [2, 3, 5, 7], device = 0)

        frame_count += 1
        if frame_count % T != 0:
            continue
        frame_count = 0

        if results[0].boxes != None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            if results[0].boxes.id == None:
                ids = []
            else:
                ids = results[0].boxes.id.cpu().numpy()

        else:
            boxes = []
            classes = []
            ids = []

        current_ids = set()

        for box, obj_id, cls in zip(boxes, ids, classes):
            obj_id = int(obj_id)
            cls = int(cls)

            # สนใจเฉพาะ vehicle
            if cls == 2:    label = "car"
            elif cls == 3:  label = "motorcycle"
            else:           label = "other"

            x1,y1,x2,y2 = map(int, box)
            
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            current_ids.add(obj_id)

            if obj_id not in tracks:
                tracks[obj_id] = list()
            tracks[obj_id].append((cx, cy))

            wrong = False

            if len(tracks[obj_id]) >= MIN_TRACK_LEN:
                x_start, y_start = tracks[obj_id][-3]
                x_end, y_end = tracks[obj_id][-1]
                
                for i, polygon in enumerate(RoI):
                    In_RoI = cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), (x_start, y_start), False)
                    if In_RoI >= 0: # 1 = in, 0 = on edge
                        d_theta = Roi_theta[i]
                        break

                dy = y_start - y_end
                dx = x_end - x_start

                theta = int(math.atan2(dy, dx) * 180 / math.pi) % 360

                if abs(theta - d_theta) >= ANGLE_TOLERANCE and (abs(dx) > 10 or abs(dy) > 10):
                    wrong = True
                    if wrong and obj_id not in wrong_way_ids:
                        wrong_way_ids.add(obj_id)
                        counter[label] += 1
                        cv2.imwrite(f"Source Code WORK//Capture_vechine//ID{obj_id}_{label}.png", raw_frame[y1-20:y2+20,x1-20:x2+20])

            # -----------------------------
            # draw bbox
            # -----------------------------
            color = (0,0,255) if obj_id in wrong_way_ids else (0,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.circle(frame, (cx, cy), 2, color, 2)
            cv2.putText(frame, f"ID {obj_id} {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # cleanup tracks
        tracks = {k:v for k,v in tracks.items() if k in current_ids}

        cv2.imshow("Wrong Way Detection", frame)
        cv2.waitKey(1)
    return