import time
import cv2
import Noon
import Night
import ROI_Manual_click

#cap = cv2.VideoCapture(0) # Web Cam
cap = cv2.VideoCapture("Video Path") # Video

ret, frame = cap.read()
if not ret: exit()
[Roi_shapes, Vector_Road, Night_Mode] = ROI_Manual_click.ROI_Click(frame)

ss = time.time()
if not Night_Mode:
    Result = Noon.detect(Roi_shapes, Vector_Road, cap)
else:
    Result = Night.detect(Roi_shapes, Vector_Road, cap)
print(time.time()-ss)
