from imutils.video import VideoStream
import imutils
import time
import cv2
from detect import model_load

vs = VideoStream(src= 0).start()

first_frame = None
motion_count = 0

path = "model_ex-094_acc-0.997927.h5"
json_path = "model_class.json"
pred = model_load(path, "fast", json_path)

while True:
    time.sleep(3)
    frame = vs.read()
    text = "Nothing"

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue
    
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < 1000:
            continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
        motion_count += 1
        print(pred.classify(frame))
    first_frame = gray

vs.stop()
cv2.destroyAllWindows()