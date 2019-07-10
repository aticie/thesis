import os
import pandas
import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation as R


def GetNormal(v1, v2, v3):
    a = v1 - v2
    b = v1 - v3

    return np.cross(a, b)


def GetAvg(v1, v2, v3):
    a = (v1 + v2 + v3) / 3

    return a


def load_json(file_name):
    with open(file_name) as cfile:
        calib = json.load(cfile)
    return calib


def scale_translate_cameras(calib, cam_pos_x, cam_pos_z):
    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}

    for k, cam in cameras.items():
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t'])

    camera_positions = []

    for cam in sel_cams:
        current_cam = cameras[0, cam]
        cc = (-current_cam['R'].transpose() * current_cam['t'])
        pos_x = cc[0][0, 0]
        pos_y = cc[1][0, 0]
        pos_z = cc[2][0, 0]
        camera_positions.append([[pos_x], [pos_y], [pos_z]])

    camera_positions = np.array(camera_positions).T.reshape(3, 4)

    cam0_x = camera_positions[0][0]
    cam0_z = camera_positions[2][0]

    camera_positions[0] = camera_positions[0] - cam0_x
    camera_positions[2] = camera_positions[2] - cam0_z

    camera_positions = camera_positions * scale_multiplier

    offset_x = cam_pos_x - camera_positions[0][0]
    offset_z = cam_pos_z - camera_positions[2][0]

    camera_positions[0] = camera_positions[0] + offset_x
    camera_positions[2] = camera_positions[2] + offset_z

    return camera_positions, (cam0_x, cam0_z)


def draw_cameras(camera_positions, img, sel_cams):
    for i in range(len(camera_positions[1])):
        cam_pos_x = int(camera_positions[0][i])
        cam_pos_z = int(camera_positions[2][i])

        camNo = sel_cams[i]

        cv2.circle(img, (cam_pos_x, cam_pos_z), 10, (0, 0, 255), -1)
        cv2.putText(img, 'Cam_' + str(camNo), (cam_pos_x - 45, cam_pos_z + 35), font, 0.75, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return None


def draw_ground_truth(img, init_cam_x, init_cam_z, cam_pos_x, cam_pos_z):
    try:
        gt_pose_file = os.path.join(cwd, "hdFace3d/faceRecon3D_hd" + frameNo + ".json")
        with open(gt_pose_file) as dfile:
            fframe = json.load(dfile)

        i = 0
        for face in fframe['people']:
            face3d = np.array(face['face70']['landmarks']).reshape((-1, 3)).transpose()
            color = cv2.cvtColor(np.uint8([[[(130 // len(fframe['people']) * i) + 10, 255, 255]]]), cv2.COLOR_HSV2BGR)
            color = (int(color[0, 0, 0]), int(color[0, 0, 1]), int(color[0, 0, 2]))
            i += 1
            points = []
            for point in face_points[0]:
                point_x = face3d[0, point]
                point_y = face3d[1, point]
                point_z = face3d[2, point]

                cam_rel_x = (face3d[0, point] - init_cam_x) * scale_multiplier
                cam_rel_z = (face3d[2, point] - init_cam_z) * scale_multiplier

                cam_rel_x += cam_pos_x
                cam_rel_z += cam_pos_z

                cv2.circle(img, (int(cam_rel_x), int(cam_rel_z)), 2, color, -1)
                points.append([point_x, point_y, point_z])

            points = np.array(points)
            c, normal = fitPlaneLTSQ(points)
            [a, b, c] = face3d[:, face_points[0]]
            [start_x, _, start_z] = [np.average(a), np.average(b), np.average(c)]
            n = normal / np.linalg.norm(normal) * 30
            start_x = int((start_x - init_cam_x) * scale_multiplier)
            start_z = int((start_z - init_cam_z) * scale_multiplier)
            start_x += cam_pos_x
            start_z += cam_pos_z
            x = int((n[0] - init_cam_x) * scale_multiplier)
            z = int((n[2] - init_cam_z) * scale_multiplier)
            x += cam_pos_x
            z += cam_pos_z
            cv2.arrowedLine(img, (start_x, start_z), (x, z), (127, 255, 0))
            '''
            for tri in face_tri:
                [a, b, c] = face3d[:, tri].T
                n = GetNormal(a, b, c)
                n = n / np.linalg.norm(n) * 30

                [start_x, _, start_z] = GetAvg(a, b, c)
                start_x = int((start_x - init_cam_x) * scale_multiplier)
                start_z = int((start_z - init_cam_z) * scale_multiplier)
                start_x += cam_pos_x
                start_z += cam_pos_z

                x = int((n[0] - init_cam_x) * scale_multiplier)
                z = int((n[2] - init_cam_z) * scale_multiplier)
                x += cam_pos_x
                z += cam_pos_z

                cv2.line(img, (start_x, start_z), (x, z), (127, 127, 127))
            '''
    except IOError as e:
        print('Error reading {0}\n'.format(gt_pose_file) + e.strerror)

    return None


def fitPlaneLTSQ(XYZ):
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  # X
    G[:, 1] = XYZ[:, 1]  # Y
    Z = XYZ[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return (c, normal)


cwd = os.getcwd()

sel_cams = [0, 8, 15, 23]
camNo = '{0:02d}'.format(sel_cams[0])

frames = ["00001147", "00001613", "00002319", "00003476", "00003961", "00004905", "00005777", "00006078", "00006328",
          "00006577"]
frameNo = frames[5]

face_edges = np.array([[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
                       [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42]])
face_tri = np.array(
    [[36, 37, 38], [37, 38, 39], [38, 39, 40], [39, 40, 41], [42, 43, 44], [43, 44, 45], [44, 45, 46], [45, 46, 47]])
face_points = np.array([[36, 37, 38, 39, 40, 41], [42, 43, 44, 45, 46, 47]])

cam_pos_x = 200
cam_pos_z = 400
scale_multiplier = 1.3

calib = load_json("calibration_160906_pizza1.json")

img = np.zeros((850, 960, 3), np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

camera_positions, (init_cam_x, init_cam_z) = scale_translate_cameras(calib, cam_pos_x, cam_pos_z)

draw_cameras(camera_positions, img, sel_cams)

saveFile = os.path.join(cwd, "TopDown/Frame_" + frameNo + ".jpg")

draw_ground_truth(img, init_cam_x, init_cam_z, cam_pos_x, cam_pos_z)

OpenFace_File = "hd_00_00/processed/" + frameNo + ".csv"
df = pandas.read_csv(OpenFace_File)

poses = df.values.tolist()

cv2.arrowedLine(img, (20, 20), (50, 20), (255, 0, 0), 2)
cv2.putText(img, "Z", (55, 25), font, 0.5, (255, 255, 255), 2)

cv2.arrowedLine(img, (20, 20), (20, 50), (0, 255, 0), 2)
cv2.putText(img, "X", (15, 65), font, 0.5, (255, 255, 255), 2)

people_pose = np.zeros((len(poses), 2))
people_rot = np.zeros((len(poses), 3))
people_conf = np.zeros(len(poses))

max_z = 0

# ---- Create predicted persons ----

#
for i, person in enumerate(poses):
    people_pose[i, 0] = person[2]
    people_pose[i, 1] = person[4]
    people_conf[i] = person[1]
    if people_pose[i, 1] > max_z:
        max_z = people_pose[i, 1]
    people_rot[i, 0] = person[5]
    people_rot[i, 1] = person[6]
    people_rot[i, 2] = person[7]

people_pose = np.divide(people_pose, max_z)
people_pose = np.multiply(people_pose, 450)

cv2.line(img, (940, 20), (940, 470), (255, 255, 255), 2)
cv2.line(img, (930, 470), (950, 470), (255, 255, 255), 2)
cv2.line(img, (930, 20), (950, 20), (255, 255, 255), 2)
cv2.putText(img, str(max_z) + "mm", (850, 225), font, 0.5, (255, 255, 255), 2)

for i in range(len(poses)):

    x = int(people_pose[i, 0])
    z = int(people_pose[i, 1])
    rot_x = people_rot[i, 0]
    rot_y = people_rot[i, 1]
    rot_z = people_rot[i, 2]

    x = (+x + cam_pos_z)
    z = (cam_pos_x + z)

    color = cv2.cvtColor(np.uint8([[[(130 // len(poses) * i) + 10, 255, 255]]]), cv2.COLOR_HSV2BGR)
    color = (int(color[0, 0, 0]), int(color[0, 0, 1]), int(color[0, 0, 2]))
    if people_conf[i] < 0.5:
        color = (127, 127, 127)
    cv2.circle(img, (z, x), 5, color, -1)
    cv2.putText(img, "P" + str(i), (z - 10, x + 20), font, 0.5, color, 2)

    r = R.from_rotvec((0, rot_y, 0))
    P1 = np.array((+1, 0, 0))

    P2 = r.apply(P1)

    print(P2)

    P2 = np.multiply(P2, 50)
    P2 = P2.astype(int)
    start_point = (z, x)
    end_point = (z - P2[0], x + P2[2])
    cv2.arrowedLine(img, start_point, end_point, color, 2)

cv2.imwrite(saveFile, img)
cv2.imshow("Frame", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
