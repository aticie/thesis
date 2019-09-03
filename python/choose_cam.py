import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

with open('calibration_160906_pizza1.json') as cfile:
    calib = json.load(cfile)
cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

for k,cam in cameras.items():
    cam['K'] = np.matrix(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.matrix(cam['R'])
    cam['t'] = np.array(cam['t']).reshape((3,1))


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(50, 90)
frame_no = 4905
sel_cameras = [0, 8, 15, 23]
edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1
colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()

# Draw all cameras in black
for k, cam in cameras.items():
    cc = (-cam['R'].transpose() * cam['t'])
    x = cc[0][0,0]
    y = cc[1][0,0]
    z = cc[2][0,0]
    ax.scatter(x,y,z, color=[0, 0, 0])

# Selected camera subset in green
for cam_no in sel_cameras:
    cam = cameras[0, cam_no]
    cc = (-cam['R'].transpose() * cam['t'])
    x = cc[0][0,0]
    y = cc[1][0,0]
    z = cc[2][0,0]
    ax.scatter(x,y,z, color=[0, 1, 0])

try:
    # Load the json file with this frame's skeletons
    skel_json_fname = 'hdPose3d_stage1_coco19/body3DScene_{0:08d}.json'.format(frame_no)
    with open(skel_json_fname) as dfile:
        bframe = json.load(dfile)
except:
    print("Error!")

# Bodies
for ids in range(len(bframe['bodies'])):
    body = bframe['bodies'][ids]
    skel = np.array(body['joints19']).reshape((-1, 4)).transpose()

    for edge in edges:
        ax.plot(skel[0, edge], skel[1, edge], skel[2, edge], color=colors[body['id']])

ax.set_aspect('equal')
ax.set_xlim3d([-300, 300])
ax.set_ylim3d([-300, 300])
ax.set_zlim3d([-300, 300])

ax.set_xlabel('$X$', fontsize=20, rotation=150)
ax.set_ylabel('$Y$')
ax.set_zlabel(r'$\gamma$', fontsize=30, rotation=60)

plt.show()
