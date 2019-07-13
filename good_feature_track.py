import numpy as np
import cv2

cap = cv2.VideoCapture('1234.mp4')


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.7,
                       minDistance = 7,
                       blockSize = 7 )

color = np.random.randint(0,255,(100,3))

while(1):
    _,f = cap.read()
    f = cv2.resize(f, (640, 480), interpolation=cv2.INTER_CUBIC)

    old_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    for corner in p0:
        x, y = corner.ravel()
        cv2.circle(f, (x, y), 1, 255, -1)
    #img = cv2.add(frame, mask)
    while(1):
        cv2.imshow('frame', f)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
