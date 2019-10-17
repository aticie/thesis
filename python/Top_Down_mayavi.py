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
from scipy.spatial import procrustes


def draw_prediction(camNo, frameNo, myFig, clr, scale, method):

    # Find openface prediction .csv file
    # OpenFace_File = f"../data/hd_00_{camNo:02d}/{scale}/{method}/processed/{frameNo}.csv"
    OpenFace_File = f"../asdf/hd_00_{camNo:02d}/processed/{frameNo}.csv"
    df = pd.read_csv(OpenFace_File)
    poses = df.values.tolist()

    current_cam = cameras[0, camNo]

    all_faces = []

    # To Do --
    for p_no, face in enumerate(poses):

        conf = face[1]

        gaze0 = face[2:5]
        gaze1 = face[5:8]

        gaze_x = face[8]
        gaze_y = face[9]

        eye_lmk_x = face[10:66]
        eye_lmk_y = face[66:122]

        eye_lmk_X = face[122:178]
        eye_lmk_Y = face[178:234]
        eye_lmk_Z = face[234:290]

        pose = face[290:293]
        rot = face[293:296]

        kp_2d_x = face[296:364]
        kp_2d_y = face[364:432]

        kp_3d_X = face[432:500]
        kp_3d_Y = face[500:568]
        kp_3d_Z = face[568:636]

        r = R.from_rotvec(rot)

        if conf < 0.4:
            clr_ = (0.5, 0.5, 0.5)
        else:
            clr_ = clr

        points_x = np.array(kp_3d_X)
        points_y = np.array(kp_3d_Y)
        points_z = np.array(kp_3d_Z)

        cam_pos = -current_cam['R'].transpose() * current_cam['t']
        '''
        #pose_draw = -current_cam['R'].transpose()*((r.apply(np.matrix(pose))).T/10)+cam_pos
        pose_x = -current_cam['R'].transpose()*(np.matrix([30,0,0]).T)
        pose_x = r.apply(pose_x.T)

        pose_y = -current_cam['R'].transpose()*(np.matrix([0,30,0]).T)
        pose_y = r.apply(pose_y.T)

        pose_z = -current_cam['R'].transpose()*(np.matrix([0,0,30]).T)
        pose_z = r.apply(pose_z.T)

        start_pt = np.array([pose_draw[0,0], pose_draw[1,0], pose_draw[2,0]])
        

        person_xx = [start_pt[0],start_pt[0]+pose_x[0,0]]
        person_xy = [start_pt[1],start_pt[1]+pose_x[0,1]]
        person_xz = [start_pt[2],start_pt[2]+pose_x[0,2]]
        person_yx = [start_pt[0],start_pt[0]+pose_y[0,0]]
        person_yy = [start_pt[1],start_pt[1]+pose_y[0,1]]
        person_yz = [start_pt[2],start_pt[2]+pose_y[0,2]]
        person_zx = [start_pt[0],start_pt[0]+pose_z[0,0]]
        person_zy = [start_pt[1],start_pt[1]+pose_z[0,1]]
        person_zz = [start_pt[2],start_pt[2]+pose_z[0,2]]

        mlab.plot3d(person_xx,person_xy,person_xz, tube_radius=3,tube_sides=10, color=(0,0,1))
        mlab.plot3d(person_yx,person_yy,person_yz, tube_radius=3,tube_sides=10, color=(0,1,0))
        mlab.plot3d(person_zx,person_zy,person_zz, tube_radius=3,tube_sides=10, color=(1,0,0))
    
        mlab.text3d(pose_draw[0,0], pose_draw[1,0],
                    pose_draw[2,0], "Person {}".format(p_no), scale=5)
        mlab.points3d(pose_draw[0], pose_draw[1],
                      pose_draw[2], scale_factor=10, color=clr_)
        '''
        points = np.concatenate(([points_x], [points_y], [points_z]), axis=0)
        points = np.matrix(points)
        points /= 10
        points = -current_cam['R'] * points + cam_pos
        points = r.apply(points.T).T
        p = points
        #print(p)
        '''
        pose = np.matrix(pose).T*0.1
        init_vec = np.array([[0], [0], [50]]).T
        init_vec = r.apply(init_vec).T

        # Face points relative to camera

        # Start point of pose vector
        cc = (-current_cam['R'].transpose() *
              current_cam['t'])+-current_cam['R'].transpose()*pose
        # End point of pose vector
        cc2 = cc+(-current_cam['R']*init_vec)

        #init_vec_rot = r.apply(-current_cam['R'].transpose()*init_vec)

        #cc2 = np.matrix(cc2)
        pose = np.matrix(pose)
        #cc2 = r.apply(cc2) - pose
        #cc = cc+r.apply(pose)

        u = float(cc2[0, 0])
        v = float(cc2[1, 0])
        w = float(cc2[2, 0])
        x = [u, float(cc[0, 0])]
        y = [v, float(cc[1, 0])]
        z = [w, float(cc[2, 0])]
        #x = float(cc[0,0])
        #y = float(cc[1,0])
        #z = float(cc[2,0])
        #mlab.plot3d(x,y,z,color=clr_, tube_radius=1.5)
        #mlab.quiver3d(x,y,z,color=clr_, scale_factor=1, mode='arrow')
        #print("Face Points:", p)
        '''
        mlab.points3d(p[0, :], p[1, :], p[2, :], scale_factor=2, color=clr_)
        all_faces.append(p)

    return all_faces
    # Extract pose, rotation and confidence


cwd = os.getcwd()

sel_cams = [0, 8, 15, 23]
sel_cams = [0]
frames = ["00001147", "00001613", "00002319", "00003476", "00003961", "00004905", "00005777", "00006078", "00006328",
          "00006577"]
frameNo = frames[2]

face_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16],  # outline (ignored)
                       [17, 18], [18, 19], [19, 20], [20, 21],  # right eyebrow
                       [22, 23], [23, 24], [24, 25], [25, 26],  # left eyebrow
                       [27, 28], [28, 29], [29, 30],  # nose upper part
                       [31, 32], [32, 33], [33, 34], [34, 35],  # nose lower part
                       [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],  # right eye
                       [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],  # left eye
                       [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48],  # Lip outline
                       [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]  # Lip inner line
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

i = 0
for cam in range(31):
    current_cam = cameras[0, cam]
    cc = (-current_cam['R'].transpose() * current_cam['t'])
    pos_x = int(cc[0])
    pos_y = int(cc[1])
    pos_z = int(cc[2])
    if cam in sel_cams:
        sel_cam_pos.append([pos_x, pos_y, pos_z])
        x_axis = -current_cam['R'].transpose() * [[250], [0], [0]]
        y_axis = -current_cam['R'].transpose() * [[0], [250], [0]]
        z_axis = -current_cam['R'].transpose() * [[0], [0], [250]]

        start = np.array([[pos_x], [pos_y], [pos_z]])
        end_x = start+np.array(x_axis)
        end_y = start+np.array(y_axis)
        end_z = start+np.array(z_axis)

        #mlab.plot3d([start[0], end_x[0]], [start[1], end_x[1]], [
        #            start[2], end_x[2]], tube_radius=5, tube_sides=20, color=(0, 0, 1))
        #mlab.plot3d([start[0], end_y[0]], [start[1], end_y[1]], [
        #            start[2], end_y[2]], tube_radius=5, tube_sides=20, color=(0, 1, 0))
        #mlab.plot3d([start[0], end_z[0]], [start[1], end_z[1]], [
        #            start[2], end_z[2]], tube_radius=5, tube_sides=20, color=(1, 0, 0))

        # Show predictions for selected cam
        color = cv2.cvtColor(
            np.uint8([[[(130 // 4 * i) + 10, 255, 255]]]), cv2.COLOR_HSV2BGR)
        i += 1
        color = color/255
        color = (color[0][0][0], color[0][0][1], color[0][0][2])
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
#mlab.plot3d(x_axis, y_axis, z_axis, tube_radius=5, tube_sides=20)
#mlab.plot3d(x_axis, y_axis, z_axis, tube_radius=5, tube_sides=20)

OpenFace_File = f"../asdf/hd_00_08/processed/{frameNo}.csv"
df = pd.read_csv(OpenFace_File)
poses = df.values.tolist()
for faceNo, face in enumerate(poses):
    if faceNo != 2:
        continue
    kp_3d_X = np.array(face[432:500])
    kp_3d_Y = np.array(face[500:568])
    kp_3d_Z = np.array(face[568:636])

    predicted_face = np.array([kp_3d_X, kp_3d_Y, kp_3d_Z])
    predicted_face = -current_cam['R'] * predicted_face
    #mlab.points3d(predicted_face[0], predicted_face[1], predicted_face[2], scale_factor=1, color=color)

    #print(kp_3d_X)
    #print(predicted_face)

#cams = mlab.points3d(x_, y_, z_, scale_factor=30, color=(0.1, 0.8, 0.1))
#for i, _ in enumerate(x_):
    #mlab.text3d(_-30, y_[i]-30, z_[i], "Cam "+str(i), scale=10)

try:
    gt_pose_file = os.path.join(
        cwd, "../data_old/hdFace3d/faceRecon3D_hd" + frameNo + ".json")
    with open(gt_pose_file) as dfile:
        fframe = json.load(dfile)
except IOError as e:
    print('Error reading {0}\n'.format(gt_pose_file) + e.strerror)

try:
    gt_pose_file = os.path.join(
        cwd, "../data_old/hdPose3d_stage1_coco19/body3DScene_" + frameNo + ".json")
    with open(gt_pose_file) as bdfile:
        bframe = json.load(bdfile)
except IOError as e:
    print('Error reading {0}\n'.format(gt_pose_file) + e.strerror)

i = 0
for face in fframe['people']:
    face3d = np.array(face['face70']['landmarks']).reshape((-1, 3)).transpose()
    color = cv2.cvtColor(np.uint8(
        [[[(130 // len(fframe['people']) * i) + 10, 255, 255]]]), cv2.COLOR_HSV2BGR)
    i += 1
    color = color/255
    color = (color[0][0][0], color[0][0][1], color[0][0][2])

    x = []
    y = []
    z = []
    # print(face3d[:,34])
    # for edge in face_edges:

    x = face3d[0, :68]
    y = face3d[1, :68]
    z = face3d[2, :68]
    if i == 3:
        xyz = np.array([x,y,z])
    x_avg = np.average(x)
    y_avg = np.average(y)
    z_avg = np.average(z)
    mlab.points3d(x, y, z, color=color)
    mlab.text3d(x_avg, y_avg-25, z_avg, "Person "+str(i), scale=10)

print(xyz.shape, predicted_face.shape)
mtx1, mtx2, disparity = procrustes(xyz, predicted_face)
print(mtx1, mtx2, disparity)
#mlab.points3d(mtx1[0], mtx1[1], mtx1[2], color=(0,0,1))
#mlab.points3d(mtx2[0], mtx2[1], mtx2[2], color=(1,0,0))
body_edges = np.array([[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [
                      8, 9], [3, 13], [13, 14], [14, 15], [1, 10], [10, 11], [11, 12]])-1

i = 0
for body in bframe['bodies']:

    skel = np.array(body['joints19']).reshape((-1, 4)).transpose()
    color = cv2.cvtColor(np.uint8(
        [[[(130 // len(fframe['people']) * i) + 10, 255, 255]]]), cv2.COLOR_HSV2BGR)
    i += 1
    color = color/255
    color = (color[0][0][0], color[0][0][1], color[0][0][2])

    for edge in body_edges:

        x = skel[0, edge]
        y = skel[1, edge]
        z = skel[2, edge]
        #mlab.plot3d(x, y, z, tube_radius=3, line_width=10, color=color)


#mlab.view(.0, - 5.0, 4)
mlab.show()
