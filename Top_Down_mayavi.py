# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# Copyright (c) 2007, Enthought, Inc.
# License: BSD Style.
import numpy as np
from mayavi import mlab
import os
import json
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R

def draw_prediction(camNo, frameNo, myFig, clr, scale, method):

    # Find openface prediction .csv file
    OpenFace_File = "data/"+"hd_00_"+f"{camNo:02d}"+"/"+scale+"/"+method+"/processed/" + frameNo + ".csv"
    df = pd.read_csv(OpenFace_File)
    poses = df.values.tolist()

    current_cam = cameras[0, camNo]

    # To Do --
    for face in poses:
        conf = face[1]
        pose = face[2:5]
        rot = face[5:8] 
        points_x = np.multiply(face[8:76],.1)
        points_y = np.multiply(face[76:144],.1)
        points_z = np.multiply(face[144:212],.1)
        print(points_x)
        print(points_y)
        points = np.concatenate( ([points_x], [points_y], [points_z]), axis=0)
        points = np.matrix(points)
        pose = np.matrix(pose).T*0.1
        init_vec = np.array([[0],[0],[50]]).T
        r = R.from_rotvec(rot)
        init_vec = r.apply(init_vec).T

        

        # Face points relative to camera
        #p = cam_pos-current_cam['R']*points

        # Start point of pose vector
        cc = (-current_cam['R'].transpose() * current_cam['t'])+current_cam['R'].transpose()*pose
        # End point of pose vector
        cc2 = cc+(-current_cam['R']*init_vec)
        cam_pos = -current_cam['R'].transpose() * current_cam['t']

        r = R.from_rotvec((rot[0], rot[1], rot[2]))
        #init_vec_rot = r.apply(current_cam['R'].transpose()*init_vec)

        #cc2 = np.matrix(cc2)
        pose = np.matrix(pose)
        #cc2 = r.apply(cc2) - pose
        #cc = cc+r.apply(pose)
        if conf < 0.6:
            clr_ = (0.4,0.4,0.4)
        else:
            clr_ = clr
        u = float(cc2[0,0])
        v = float(cc2[1,0])
        w = float(cc2[2,0])
        x = [u,float(cc[0,0])]
        y = [v,float(cc[1,0])]
        z = [w,float(cc[2,0])]
        #x = float(cc[0,0])
        #y = float(cc[1,0])
        #z = float(cc[2,0])
        mlab.plot3d(x,y,z,color=clr_, tube_radius=1.5)
        #mlab.quiver3d(x,y,z,color=clr_, scale_factor=1, mode='arrow')
        
        #mlab.points3d(p[0,:], p[1,:], p[2,:], scale_factor=3,color=clr_)

        

    # Extract pose, rotation and confidence


cwd = os.getcwd()

sel_cams = [0, 8, 15, 23]
sel_cams = [0]
frames = ["00001147", "00001613", "00002319", "00003476", "00003961", "00004905", "00005777", "00006078", "00006328",
          "00006577"]
frameNo = frames[4]

face_edges = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[11,12],[12,13],[14,15],[15,16], #outline (ignored)
                [17,18],[18,19],[19,20],[20,21], #right eyebrow
                [22,23],[23,24],[24,25],[25,26], #left eyebrow
                [27,28],[28,29],[29,30],   #nose upper part
                [31,32],[32,33],[33,34],[34,35], #nose lower part
                [36,37],[37,38],[38,39],[39,40],[40,41],[41,36], #right eye
                [42,43],[43,44],[44,45],[45,46],[46,47],[47,42], #left eye
                [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,48], #Lip outline
                [60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,60] #Lip inner line 
                ])

with open("calibration_160906_pizza1.json", 'r') as cfile:
    calib = json.load(cfile)

cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}

scale = "Scaled_3"
method = "Lanczos4"

for k, cam in cameras.items():
    cam['K'] = np.matrix(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.matrix(cam['R'])
    cam['t'] = np.array(cam['t'])

camera_positions = []
sel_cam_pos = []

myFig = mlab.figure(fgcolor=(0, 0, 0), bgcolor=(0.8, 0.8, .8))

i=0
for cam in range(31):
    current_cam = cameras[0, cam]
    cc = (-current_cam['R'].transpose() * current_cam['t'])
    pos_x = int(cc[0])
    pos_y = int(cc[1])
    pos_z = int(cc[2])
    if cam in sel_cams:
        sel_cam_pos.append([pos_x, pos_y, pos_z])

        # Show predictions for selected cam
        color = cv2.cvtColor(np.uint8([[[(130 // 4 * i) + 10, 255, 255]]]), cv2.COLOR_HSV2BGR)
        i+=1
        color = color/255
        color = (color[0][0][0],color[0][0][1],color[0][0][2])
        draw_prediction(cam, frameNo, myFig, color, scale, method)

    else:
        camera_positions.append([pos_x, pos_y, pos_z])



x = []
y = []
z = []
x_ = []
y_ = []
z_ = []

for array in camera_positions:
    x.append(array[0])
    y.append(array[1])
    z.append(array[2])

for array_ in sel_cam_pos:
    x_.append(array_[0])
    y_.append(array_[1])
    z_.append(array_[2])

x = np.array(x)
y = np.array(y)
z = np.array(z)

x_ = np.array(x_)
y_ = np.array(y_)
z_ = np.array(z_)

#mlab.points3d(x, y, z,scale_factor=30, color=(0.8, 0.1, 0.1))
cams = mlab.points3d(x_, y_, z_,scale_factor=30, color=(0.1, 0.8, 0.1))
for i,_ in enumerate(x_):
    mlab.text3d(_-30,y_[i]-30,z_[i], "Cam "+str(i),scale=10)

try:
    gt_pose_file = os.path.join(cwd, "hdFace3d/faceRecon3D_hd" + frameNo + ".json")
    with open(gt_pose_file) as dfile:
        fframe = json.load(dfile)
except IOError as e:
    print('Error reading {0}\n'.format(gt_pose_file) + e.strerror)

try:
    gt_pose_file = os.path.join(cwd, "hdPose3d_stage1_coco19/body3DScene_" + frameNo + ".json")
    with open(gt_pose_file) as bdfile:
        bframe = json.load(bdfile)
except IOError as e:
    print('Error reading {0}\n'.format(gt_pose_file) + e.strerror)

i = 0
for face in fframe['people']:
    face3d = np.array(face['face70']['landmarks']).reshape((-1, 3)).transpose()
    color = cv2.cvtColor(np.uint8([[[(130 // len(fframe['people']) * i) + 10, 255, 255]]]), cv2.COLOR_HSV2BGR)
    i+=1
    color = color/255
    color = (color[0][0][0],color[0][0][1],color[0][0][2])

    x = []
    y = []
    z = []

    for edge in face_edges:
        #print(face3d[0,edge])
        x = face3d[0,edge]
        y = face3d[1,edge]
        z = face3d[2,edge]
        mlab.plot3d(x,y,z, tube_radius=0.75,color=color)

body_edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1

i=0
for body in bframe['bodies']:

    skel = np.array(body['joints19']).reshape((-1,4)).transpose()
    color = cv2.cvtColor(np.uint8([[[(130 // len(fframe['people']) * i) + 10, 255, 255]]]), cv2.COLOR_HSV2BGR)
    i+=1
    color = color/255
    color = (color[0][0][0],color[0][0][1],color[0][0][2])

    for edge in body_edges:

        x = skel[0, edge]
        y = skel[1, edge]
        z = skel[2, edge]
        mlab.plot3d(x,y,z, tube_radius=3, line_width=10, color=color)
    

#mlab.view(.0, - 5.0, 4)
mlab.show()