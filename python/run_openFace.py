import json
import subprocess
import os


def load_json(file_name):
    with open(file_name) as cfile:
        calib = json.load(cfile)
    return calib


executable = "/home/openface-build/build/bin/FaceLandmarkImg"

cams = ["hd_00_00", "hd_00_08", "hd_00_15", "hd_00_23"]
calib = load_json("calibration_160906_pizza1.json")

for cam in cams:

    cam_no = cam[-2:]
    cam_no = int(cam_no)
    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}
    cam_folder = os.path.join("copy_to_docker", cam)
    current_cam = cameras[0, cam_no]
    k = current_cam['K']
    fx = str(k[0][0])
    fy = str(k[1][1])
    cx = str(k[0][2])
    cy = str(k[1][2])
    frames = os.listdir(cam_folder)

    for frame in frames:
        frame_path = os.path.join(cam_folder, frame)
        command = [executable, "-f", frame_path, "-fx", fx, "-fy",
                   fy, "-cx", cx, "-cy", cy, "-out_dir", os.path.join(cam_folder, "processed")]
        s = " ".join(command)
        subprocess.call(s, shell=True)
