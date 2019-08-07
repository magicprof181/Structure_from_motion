import cv2
import numpy as np
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS, 10)
i = 0
dataset = []
while(1):
    ret,frame = video.read()
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    dataset.append(frame)
    i = i+1
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or i ==100:
        break
np.save('dataset_BGR1.npy',dataset)
video.release()
j = 0
dataset = np.load("/home/abhay/PycharmProjects/structure_from_motion/venv/dataset_BGR1.npy")
while(j<100):
    cv2.imshow('frame',dataset[j,:,:])
    cv2.waitKey(500)
    j = j+1
cv2.destroyAllWindows()

