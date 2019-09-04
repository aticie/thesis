import numpy as np
from mayavi import mlab
import os
import json
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R

camNo = "08"
frameNo = "00001147"

scale = "Scaled_3"
method = "Lanczos4"

OpenFace_File = f"../data/hd_00_{camNo}/processed/{frameNo}.csv"
OpenFace_File_3 = f"../data/hd_00_{camNo}/{scale}/{method}/processed/{frameNo}.csv"
OpenFace_File_2 = f"../data/hd_00_{camNo}/Scaled_2/{method}/processed/{frameNo}.csv"
OpenFace_File_4 = f"../data/hd_00_{camNo}/Scaled_4/{method}/processed/{frameNo}.csv"


df = pd.read_csv(OpenFace_File)
df2 = pd.read_csv(OpenFace_File_2)
df3 = pd.read_csv(OpenFace_File_3)
df4 = pd.read_csv(OpenFace_File_4)

pose1 = df.values.tolist()
pose2 = df2.values.tolist()
pose3 = df3.values.tolist()
pose4 = df4.values.tolist()

poses = [pose1,pose2,pose3,pose4]

myFig = mlab.figure(fgcolor=(0, 0, 0), bgcolor=(0.8, 0.8, .8))

for file in poses:

    for p_no, face in enumerate(file):

        conf = face[1]
        pose = [face[2],face[3],face[4]]
        rot = face[5:8]

        r = R.from_rotvec(rot)

        points_x = np.array(face[8:76])
        points_y = np.array(face[76:144])
        points_z = np.array(face[144:212])

        if points_x.size == 0:
            break

        points = np.concatenate(([points_x], [points_y], [points_z]), axis=0)
        points = np.matrix(points)
        print(p_no, points)
        points = r.apply(points.T).T
        p = points

        pose = r.apply(pose)

        
        if p_no == 0:
            clr = (0,0,1)
        elif p_no == 1:
            clr = (0,1,0)
        elif p_no == 2:
            clr = (1,0,0)
        else:
            clr = (1,0,1)

        mlab.points3d(p[0,:], p[1,:], p[2,:], scale_factor=25,color=clr)
        #mlab.plot3d(p[0,:],p[1,:],p[2,:], tube_radius=10,tube_sides=20, color=clr)
        #mlab.points3d(pose[0], -pose[1], pose[2], scale_factor=150,color=clr)
        
    
mlab.points3d(0,0,0, scale_factor=1000, color=(0.2,0.2,0.2))
mlab.plot3d([0,0],[0,0],[0,2500], tube_radius=100,tube_sides=20, color=(0,0,1))
mlab.plot3d([0,0],[0,2500],[0,0], tube_radius=100,tube_sides=20, color=(0,1,0))
mlab.plot3d([0,2500],[0,0],[0,0], tube_radius=100,tube_sides=20, color=(1,0,0))

mlab.show()
