import time
import cv2
import Noon
import Night
import ROI_Manual_click

#cap = cv2.VideoCapture("CCTV//Short Take//Noon//Case 1.mp4")
cap = cv2.VideoCapture("CCTV//Short Take//Night//Case 1.mp4")
#cap = cv2.VideoCapture("CCTV//Long Take//CAM31 Night 2.mp4")
#cap = cv2.VideoCapture("CCTV//Long Take//CAM31 Noon 3.mp4")
#cap = cv2.VideoCapture("Test//Two-lane Road (cut).mp4")
#cap = cv2.VideoCapture("CCTV//Long Take//CAM48 Night 1.mp4")

ret, frame = cap.read()
if not ret: exit()
[Roi_shapes, Vector_Road, Night_Mode] = ROI_Manual_click.ROI_Click(frame)

#print(Roi_shapes)
#print(Vector_Road)

ss = time.time()
if not Night_Mode:
    Result = Noon.detect(Roi_shapes, Vector_Road, cap)
else:
    Result = Night.detect(Roi_shapes, Vector_Road, cap)
print(time.time()-ss)