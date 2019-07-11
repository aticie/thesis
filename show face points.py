import os
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

cwd = os.getcwd()
pi = 3.14159265358

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
    return c, normal


try:
    gt_pose_file = os.path.join(cwd, "hdFace3d/faceRecon3D_hd" + "00001147" + ".json")
    with open(gt_pose_file) as dfile:
        fframe = json.load(dfile)
except IOError as e:
    print('Error reading {0}\n'.format(gt_pose_file) + e.strerror)

i = 0
for face in fframe['people']:
    img = np.zeros((500, 500, 3), np.uint8)
    face3d = np.array(face['face70']['landmarks']).reshape((-1, 3))
    face3d = face3d - face3d[30]
    c, normal = fitPlaneLTSQ(face3d)
    perp_vec = [0, 0, 1]
    perp = np.cross(normal, perp_vec)
    perp2 = np.cross(normal, perp)

    eps = 1e-15

    assert -eps < np.dot(perp, normal) < eps, "Should be 0"
    assert -eps < np.dot(perp2, normal) < eps, "Should be 0"
    assert -eps < np.dot(perp, perp2) < eps, "This too"

    rot = R.from_dcm(np.array([normal, perp, perp2]))

    new_face3d = rot.apply(face3d)
    new_face3d = new_face3d*25 + [250, 250, 250]
    for point in new_face3d:
        cv2.circle(img, (int(point[1]), int(point[2])), 5, (200, 200, 200), -1)

    cv2.imwrite("Person"+str(i)+"_face.jpg", img)
    i+=1
