import cv2
import os

cams = ["hd_00_00", "hd_00_08", "hd_00_15", "hd_00_23"]

scales = [1.25, 1.5, 1.75, 2, 3, 4]

scale_method = [cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
scale_method_dict  = {cv2.INTER_LINEAR  : "Linear",
					  cv2.INTER_NEAREST : "Nearest",
					  cv2.INTER_CUBIC   : "Cubic",
					  cv2.INTER_LANCZOS4: "Lanczos4"}

# Directory of cam
for cam in cams:

	# List frames for that cam
	frames = os.listdir(cam)
	
	for scale in scales:

		# Create a scaled_x folder
		scaled_folder = os.path.join(cam,"Scaled_"+str(scale))
		if not os.path.exists(scaled_folder):
			os.mkdir(scaled_folder) 

		for method in scale_method:

			# Create method folder
			method_name = scale_method_dict[method]
			method_folder = os.path.join(scaled_folder, method_name)
			if not os.path.exists(method_folder):
				os.mkdir(method_folder)

			# Scale and save images
			for frame in frames:
				
				frame_path = os.path.join(cam, frame)

				if os.path.isfile(frame_path) and ".jpg" in frame:
					img = cv2.imread(frame_path)
					scaled_img_size = (round(scale*img.shape[0]),round(scale*img.shape[1]))
					scaled_img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=method)
					save_path = os.path.join(method_folder, frame)
					#print(save_path, scaled_img_size)
					cv2.imwrite(save_path, scaled_img)