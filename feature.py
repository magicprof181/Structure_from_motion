import numpy as np
import cv2
'''
dataset = np.load("/home/abhay/PycharmProjects/structure_from_motion/venv/dataset_BGR1.npy")
gray = cv2.cvtColor(dataset[20,:,:],cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,20,0.001,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(dataset[20,:,:],(x,y),3,255,-1)






orb = cv2.ORB_create()
kp = orb.detect(dataset[0,:,:],None)

kp, des = orb.compute(dataset[0,:,:], kp)
img2 = cv2.drawKeypoints(dataset[0,:,:],kp,None,color=(0,255,0), flags=0)
print(kp[0].pt)
print(corners)
cv2.imshow('img',dataset[20,:,:])
cv2.waitKey(0)'''
j = 1
dataset = np.load("/home/abhay/PycharmProjects/structure_from_motion/venv/dataset_BGR.npy")
while(j<9):
    gray = cv2.cvtColor(dataset[j, :, :], cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 40, 0.00000000001,10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(dataset[j, :, :], (x, y), 3, 255, -1)
    cv2.imshow('frame',dataset[j,:,:])
    cv2.waitKey(500000)
    j = j+1
cv2.destroyAllWindows()