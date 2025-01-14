import numpy as np
import cv2
import importlib
import matplotlib
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
matplotlib.get_backend()




dataset = np.load('dataset_BGR1.npy')

feature_params = dict( maxCorners = 30,
                       qualityLevel = 0.2,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))
old_frame = dataset[0,:,:]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

mask = np.zeros_like(old_frame)
j=1
w = np.zeros((200,int(len(p0))))
number_of_frame = 100
while(j!=8):
    frame = dataset[j,:,:]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params ,flags=0)

    for k in range(len(p0)):
        if (st[k,:]==1):
            w[j,k] = p1[k,0,0]
            w[100+j,k] = p1[k,0,1]

    good_new = p1[st==1]
    good_old = p0[st==1]

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    j = j+1
    cv2.imshow('frame',img)
    k = cv2.waitKey(1000) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

w_bar = w - np.mean(w,axis=0)
w_bar = w_bar.astype('float32')


u, s_, v = np.linalg.svd(w_bar, full_matrices=False)
s = np.diag(s_)[:3,:3]
u = u[:,0:3]
v = v[0:3,:]


S_cap = np.dot(np.sqrt(s),v[0:3,:])
R_cap = np.dot(u[:,0:3],np.sqrt(s))
R_cap_i = R_cap[0,:]
R_cap_j = R_cap[100,:]
print(R_cap_i,R_cap_j)




#ax.surf(S[0,:], S[1,:],S[2,:])
X = S_cap[0,:]
Y = S_cap[1,:]
Z = S_cap[2,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X,Y,Z, c='r', marker='o')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

fig.savefig('optical_flow.png', bbox_inches='tight')

np.save('Shape.npy',S_cap)
np.savetxt("Shape.csv", S_cap, delimiter=",")
cv2.destroyAllWindows()


