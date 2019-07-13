import numpy as np
import cv2
import importlib
import matplotlib
from matplotlib import cm
import tomasi_kanade

# tomasi kanade is the python file which is used later
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
matplotlib.get_backend()

w = np.asarray(
    [[141,191,213,276,317,127,195,261,121,190,258,316],[142,190,214,276,315,127,195,261,122,190,258,315],[141,191,213,275,318,127,194,261,120,189,258,316],[140,189,212,273,316,124,194,259,119,187,256,315],[138,188,210,273,313,126,193,259,118,187,254,314],[137,187,210,273,312,122,192,257,117,185,253,312],[138,187,210,273,314,124,193,257,117,186,253,313],[139,188,211,273,313,124,193,258,116,185,354,313],[199,206,208,218,224,267,272,278,343,345,350,349],[200,207,210,218,226,265,270,277,343,345,349,348],[202,208,209,220,228,267,273,278,344,346,349,347],[199,206,210,220,227,267,273,279,341,344,348,347],[195,201,205,217,220,260,265,270,334,335,341,340],[191,196,199,209,217,256,260,267,331,333,337,339],[186,191,194,206,211,250,255,263,325,327,334,334],[179,185,189,199,205,242,249,256,321,323,329,329]])



w_bar = w - np.mean(w,axis=1)[:,None]
# Try both
# Finding Q by machine learning
mu = np.mean(w_bar)
std = np.std(w_bar)
w_norm = (w_bar-mu)/std

R,S,R_,S_ = tomasi_kanade.recover_3d_structure(w_norm.astype(np.float32)/10)
X = S[0,:]
Y = S[1,:]
Z = S[2,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X,Y,Z, c='r', marker='o')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

fig.savefig('optical_flow.png', bbox_inches='tight')

'''# without Q matrix
w_bar = w_bar.astype('float32')
u, s_, v = np.linalg.svd(w_bar, full_matrices=False)
s = np.diag(s_)[:3,:3]
u = u[:,0:3]
v = v[0:3,:]


S_cap = np.dot(np.sqrt(s),v[0:3,:])
R_cap = np.dot(u[:,0:3],np.sqrt(s))





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

cv2.destroyAllWindows()'''