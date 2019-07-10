import json
import subprocess
import os

def load_json(file_name):
    with open(file_name) as cfile:
        calib = json.load(cfile)
    return calib

executable = "C:\\Users\\Administrator\\Downloads\\OpenFace_2.1.0_win_x64\\OpenFace_2.1.0_win_x64\\FaceLandmarkImg.exe"

cams = ["hd_00_08", "hd_00_15", "hd_00_23"]

for cam in cams:
    frames = os.listdir(cam)
    cam_no = cam[-2:]
    cam_no = int(cam_no)
    calib = load_json("calibration_160906_pizza1.json")
    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}
    current_cam = cameras[0, cam_no]
    k = current_cam['K']
    fx = str(k[0][0])
    fy = str(k[1][1])
    cx = str(k[0][2])
    cy = str(k[1][2])
    for frame in frames:
        frame_path = os.path.join(cam, frame)
        command = [executable, "-f", frame_path, "-pose", "-vis-track", "-tracked",
                   "-fx", fx, "-fy", fy, "-cx", cx, "-cy", cy, "-out_dir", os.path.join(cam, "processed")]
        s = " "
        s = s.join(command)
        subprocess.run(s)
