import cv2
import math
import numpy as np

Roi_shapes = []
Vector_Road = []
current_points = []
current_vector = []
is_selecting_vector = False
is_done = False
img_main = None
bottom_bar = None
ROI_win_Message = "Define the ROI on the map"

def redraw():
    global img_main, bottom_bar, Roi_shapes, Vector_Road, current_points, current_vector, is_selecting_vector
    
    temp_img = np.vstack((img_main, bottom_bar)).copy()
    h_offset = img_main.shape[0]

    # 1. วาด Shape (Polygon) ที่บันทึกไปแล้ว (สีน้ำเงิน/เขียว)
    for shape in Roi_shapes:
        if len(shape) > 2:
            pts = np.array(shape, np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_img, [pts], isClosed=True, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        for pt in shape:
            cv2.rectangle(temp_img, (pt[0]-3, pt[1]-3), (pt[0]+3, pt[1]+3), (0, 255, 0), -1)

    # 2. วาดเส้น Vector (สีชมพู) ที่บันทึกไปแล้ว
    for vec in Vector_Road:
        Vec_color = (255, 0, 255)
        if len(vec) == 2:
            p1, p2 = vec
            cv2.line(temp_img, p1, p2, Vec_color, 10, cv2.LINE_AA)
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            L = 50
            spread = 30 * (math.pi / 180) 
            tp1 = (int(p2[0] - L * math.cos(angle - spread)),
                int(p2[1] - L * math.sin(angle - spread)))
            tp2 = (int(p2[0] - L * math.cos(angle + spread)),
                int(p2[1] - L * math.sin(angle + spread)))
            triangle_pts = np.array([p2, tp1, tp2], np.int32)
            cv2.fillPoly(temp_img, [triangle_pts], Vec_color)

    # 3. วาดสิ่งที่กำลังทำอยู่
    if is_selecting_vector:
        for pt in current_vector:
            cv2.circle(temp_img, pt, 5, (255, 100, 255), -1)

    else:
        if len(current_points) > 1:
            for i in range(len(current_points) - 1):
                cv2.line(temp_img, current_points[i], current_points[i+1], (0, 255, 255), 2, cv2.LINE_AA)
            if len(current_points) >= 3:
                cv2.line(temp_img, current_points[-1], current_points[0], (0, 255, 255), 2, cv2.LINE_AA)
        for pt in current_points:
            cv2.rectangle(temp_img, (pt[0]-5, pt[1]-5), (pt[0]+5, pt[1]+5), (255, 0, 0), -1)

    # 4. วาดปุ่ม Menu
    # NEXT SHAPE button
    cv2.rectangle(temp_img, (10, h_offset + 20), (150, h_offset + 70), (200, 100, 0), -1)
    cv2.putText(temp_img, " NEXT STEP", (30, h_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # FINISH button
    cv2.rectangle(temp_img, (170, h_offset + 20), (270, h_offset + 70), (0, 150, 0), -1)
    cv2.putText(temp_img, "FINISH", (195, h_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Night button
    cv2.rectangle(temp_img, (830, h_offset + 20), (950, h_offset + 70), (0, 0, 0), -1)
    cv2.putText(temp_img, "Night Mode", (840, h_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Status Message
    mode_text = "MODE: Vector (Pink)" if is_selecting_vector else "MODE: Shape (Yellow)"
    NN = "Night" if Night_Algo else "Noon"
    status = f"Shapes: {len(Roi_shapes)} | Vectors: {len(Vector_Road)} | Detect Mode: {NN}"
    cv2.putText(temp_img, status, (290, h_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(temp_img, mode_text, (290, h_offset + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.imshow(ROI_win_Message, temp_img)

def clickPosition(event, x, y, flags, param):
    global current_points, is_done, is_selecting_vector, current_vector, Night_Algo
    global Roi_shapes, Vector_Road, img_main

    h_offset = img_main.shape[0]
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if h_offset + 15 <= y <= h_offset + 65:
            if 10 <= x <= 150: # NEXT STEP
                if len(current_points) >= 3:
                    Roi_shapes.append(tuple(current_points.copy()))
                    current_points = []
                    is_selecting_vector = True # เข้าสู่โหมดเลือกจุดสีชมพู
                    current_vector = []
            
            elif 170 <= x <= 270: # FINISH
                if len(Roi_shapes) != len(Vector_Road) or len(Vector_Road) == 0:
                    is_done = False
                else:
                    if len(current_points) >= 3: 
                        Roi_shapes.append(tuple(current_points.copy()))
                    is_done = True

            elif 850 <= x <= 950: # FINISH
                Night_Algo = not Night_Algo

        # ตรวจสอบการคลิกบนภาพ
        elif y < h_offset:
            if is_selecting_vector:
                current_vector.append((x, y))
                if len(current_vector) == 2:
                    Vector_Road.append(tuple(current_vector.copy()))
                    current_vector = []
                    is_selecting_vector = False # ครบ 2 จุดแล้ว กลับไปวาด Polygon ปกติ
            else:
                current_points.append((x, y))
        redraw()
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        if is_selecting_vector:
            if len(current_vector) > 0:
                current_vector.pop()
            else:
                is_selecting_vector = False
        else:
            if len(current_points) != 0: 
                current_points.pop()
            elif len(Roi_shapes) != 0: 
                current_points = Roi_shapes.pop()
        redraw()


def ROI_Click(img_raw):
    global img_main, bottom_bar, Roi_shapes, Vector_Road, current_points, is_done, ROI_win_Message, Night_Algo
    
    fix_size = (960, 540)
    img_main = cv2.resize(img_raw, fix_size).copy()

    # Button Bar
    bar_height = 90
    bottom_bar = np.full((bar_height, img_main.shape[1], 3), 50, dtype=np.uint8)

    # Initial lists
    Roi_shapes = []
    Vector_Road = []
    current_points = []
    is_done = False
    Night_Algo = False

    cv2.namedWindow(ROI_win_Message)
    cv2.setMouseCallback(ROI_win_Message, clickPosition)
    redraw()

    while not is_done:
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break # ESC
    
    cv2.destroyAllWindows()
    return [Roi_shapes, Vector_Road, Night_Algo]