from __future__ import division
import operator
import numpy as np
import cv2
import math
from threading import Thread
import threading
import sys
from Queue import Queue
from StreamClass import FileVideoStream
from util import *
import time

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global clicked_point, held_point
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x,y)
        
    if flags & cv2.EVENT_FLAG_LBUTTON == 1:
        if event == cv2.EVENT_MOUSEMOVE:
            held_point = (x,y)

vid_name = 'big_buck_bunny_480p_1mb.mp4'

clicked_point = (0,0)
held_point = (0,0)
state = 'Play'
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click_and_crop)
crop_start = 0
crop_end = 0
font = cv2.FONT_HERSHEY_SIMPLEX
fvs = FileVideoStream(vid_name).start()
length = fvs.length
time.sleep(1.0)
i=0

while True:
    if state == 'Quit':
        break
    # Capture frame-by-frame
    if state == 'Play':
        while True:

            frame = fvs.read()
            
            h,w,c = frame.shape
            frame = add_buttons(frame, fvs.curFrame, length)
            frame = add_current_frame_text(frame, fvs.curFrame, length, font)
            frame = add_crop_frame(frame, crop_start, crop_end)
            frame = cv2.rectangle(frame, (340, 400), (400,460), (12, 30, 255), 2)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.imshow('frame', frame)    
            key = cv2.waitKey(30) & 0xFF

            

            # Check if click is on seeking bar (and seek video)
            if tuple_compare(clicked_point,(40,h-100),operator.gt) and tuple_compare(clicked_point,(w-40,h-80),operator.lt):
                fvs.curFrame = int((clicked_point[0]-40)/(w-80)*length)
                state = "Pause"
                break

            # Check if click is on pause button
            clicked_point_on_pause = tuple_compare(clicked_point,(240,400),operator.gt) and tuple_compare(clicked_point,(300,460),operator.lt)
            if key == ord(' ') or key == ord('c') or clicked_point_on_pause:
                clicked_point = (0,0)
                state = 'Pause'
                break

            if cv2.getWindowProperty('frame', 0) < 0:     
                state = "Quit"   
                break

    elif state == 'Pause':
        print("frameNo: ",fvs.curFrame)
        print(threading.current_thread())
        frame = fvs.read()
        frame = add_buttons(frame, fvs.curFrame, length)
        frame = add_current_frame_text(frame, fvs.curFrame, length, font)
        frame = add_crop_frame(frame, crop_start, crop_end)
        frame = cv2.rectangle(frame, (240, 400), (300,460), (12, 30, 255), 2)
        while(True):

            h,w,c = frame.shape

            cv2.imshow('frame', frame)    
            key = cv2.waitKey(1) & 0xFF

            clicked_point_on_resume = tuple_compare(clicked_point,(340,400),operator.gt) and tuple_compare(clicked_point,(400,460),operator.lt)
            clicked_point_on_fwd_frame = tuple_compare(clicked_point,(440,400),operator.gt) and tuple_compare(clicked_point,(500,460),operator.lt)
            clicked_point_on_rwd_frame = tuple_compare(clicked_point,(140,400),operator.gt) and tuple_compare(clicked_point,(200,460),operator.lt)
            if key == ord(' ') or key == ord('v') or clicked_point_on_resume:
                clicked_point = (0,0)
                state = 'Play'
                break

            elif key == ord('x') or clicked_point_on_rwd_frame:
                clicked_point = (0,0)
                state = 'Rwd-frame'
                break

            elif key == ord('b') or clicked_point_on_fwd_frame:
                clicked_point = (0,0)
                state = 'Fwd-frame'
                break

            if key == ord('o'):
                crop_start = fvs.curFrame

            if key == ord('p'):
                crop_end = fvs.curFrame

            if key == ord('r'):
                crop_end = 0
                crop_start = 0

            if key == ord('l'):
                crop_video_start_label(frame, fvs.curFrameNo, crop_start, crop_end, fvs, vid_name)

            # Check if you held down on seek bar
            # if tuple_compare(clicked_point,(40,h-100),operator.gt) and tuple_compare(clicked_point,(w-40,h-80),operator.lt):
            #     fvs.curFrameNo = int((clicked_point[0]-40)/(w-80)*length)
            #     fvs.seek = True
            #     clicked_point = (0,0)
            #     state = "Pause"
            #     break

            # if tuple_compare(held_point,(40,h-100),operator.gt) and tuple_compare(held_point,(w-40,h-80),operator.lt):
            #     fvs.curFrameNo = int((held_point[0]-40)/(w-80)*length)
            #     fvs.seek = True
            #     held_point = (0,0)
            #     state = "Pause"
            #     break

            if cv2.getWindowProperty('frame', 0) < 0:     
                state = "Quit"   
                break

    elif state == 'Rwd-frame':
        fvs.rwd_frame = True
        state = 'Pause'

    elif state == 'Fwd-frame':
        state = 'Pause'

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
