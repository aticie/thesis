import json
import subprocess
import os

def load_json(file_name):
    with open(file_name) as cfile:
        calib = json.load(cfile)
    return calib

cams = ["hd_00_00", "hd_00_08", "hd_00_15", "hd_00_23"]

scales = [1.25, 1.5, 1.75, 2, 3, 4]

scale_method = ["Linear", "Nearest", "Cubic", "Lanczos4"]

executable = "/home/openface-build/build/bin/FaceLandmarkImg"

cams = ["hd_00_00", "hd_00_08", "hd_00_15", "hd_00_23"]
calib = load_json("calibration_160906_pizza1.json")

for cam in cams:
    
    cam_no = cam[-2:]
    cam_no = int(cam_no)
    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}
    current_cam = cameras[0, cam_no]
    k = current_cam['K']
    fx = str(k[0][0])
    fy = str(k[1][1])
    cx = str(k[0][2])
    cy = str(k[1][2])
    for scale in scales:

        # Create a scaled_x folder
        scaled_folder = os.path.join(cam,"Scaled_"+str(scale))

        for method in scale_method:

            # Create method folder
            method_folder = os.path.join(scaled_folder, method)
            
            frames = os.listdir(method_folder)

            for frame in frames:
                frame_path = os.path.join(method_folder, frame)
                command = [executable, "-f", frame_path, "-pose", "-3Dfp", "-vis-track", "-tracked",
                           "-fx", fx, "-fy", fy, "-cx", cx, "-cy", cy, "-out_dir", os.path.join(method_folder, "processed")]
                s = " ".join(command)
                subprocess.call(s, shell=True)
