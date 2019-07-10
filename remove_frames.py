import os

cwd = os.getcwd()

frames = ["00001147","00001613","00002319","00003476","00003961","00004905","00005777","00006078","00006328","00006577"]

folder = os.path.join(cwd, "hd_00_23")

files = os.listdir(folder)
files.sort()

i = 0

for file in files:
    file_path = os.path.join(folder, file)
    if i<len(frames):
        if frames[i] in file:
            i += 1
            print("Skipping: " + file)
        else:
            os.remove(file_path)
    elif "remove_frames.py" not in file:
        os.remove(file_path)

print("Done!")
