from __future__ import division
import cv2
import numpy as np
import operator
import math
from StreamClass import FileVideoStream


def add_buttons(frame, current_frame, length):

    h, w, c = frame.shape
    button_size = 60
    m0 = 40
    m1 = 20

    for i in range(6):
        button_pos_1 = (m0*(i+1)+button_size*i, h-m1)
        button_pos_2 = ((m0+button_size)*(i+1), h-m1-button_size)
        cv2.rectangle(frame, button_pos_1, button_pos_2, (53,53,53), -1)

        top_left_x = button_pos_1[0]
        top_left_y = button_pos_1[1]
        # Rewind button
        if i == 0:
            # Rewind button
            rewind_pts = [[m0+int(button_size/5), h-m1-int(button_size/2)], #mid point of left triangle
                          [m0+int(button_size*3/5), h-m1-int(button_size/5)], #lower point of left triangle
                          [m0+int(button_size*3/5), h-m1-int(button_size*2/5-3)], # lower point of intersection
                          [m0+int(button_size*4/5), h-m1-int(button_size/5)], #lower point of right triangle
                          [m0+int(button_size*4/5), h-m1-int(button_size*4/5)], #upper point of right triangle
                          [m0+int(button_size*3/5), h-m1-int(button_size*3/5+3)], #upper point of intersection
                          [m0+int(button_size*3/5), h-m1-int(button_size*4/5)]] #upper point of left triangle
            pts = np.array(rewind_pts)
            cv2.fillPoly(frame, np.int32([pts]),(255,255,255))

        # Rewind by frame
        if i == 1:
            cv2.rectangle(frame, (top_left_x+int(button_size/5), top_left_y-int(button_size/5)),(top_left_x+int(button_size*3/10), top_left_y-int(button_size*4/5)), (255,255,255), -1)
            tri_pts = [[top_left_x+int(button_size*3/10),top_left_y-int(button_size/2)],
                       [top_left_x+int(button_size*4/5), top_left_y-int(button_size/5)],
                       [top_left_x+int(button_size*4/5), top_left_y-int(button_size*4/5)]]
            cv2.fillPoly(frame, np.int32([tri_pts]), (255,255,255))
        
        #Pause button
        if i == 2:
            cv2.rectangle(frame, (top_left_x+int(button_size/5),top_left_y-int(button_size*4/5)), (top_left_x+int(button_size*2/5),top_left_y-int(button_size*1/5)), (255,255,255), -1)
            cv2.rectangle(frame, (top_left_x+int(button_size*3/5),top_left_y-int(button_size*4/5)), (top_left_x+int(button_size*4/5),top_left_y-int(button_size*1/5)), (255,255,255), -1)

        # Resume button
        if i == 3:
            tri_pts = [[top_left_x+int(button_size*4/5), top_left_y-int(button_size/2)], #mid point of triangle
                       [top_left_x+int(button_size*1/5), top_left_y-int(button_size*4/5)], #lower point of triangle
                       [top_left_x+int(button_size*1/5), top_left_y-int(button_size*1/5)]]
            cv2.fillPoly(frame, np.int32([tri_pts]), (255,255,255))

        # Forward by frame
        if i == 4: 
            cv2.rectangle(frame, (top_left_x+int(button_size*4/5), top_left_y-int(button_size/5)),(top_left_x+int(button_size*7/10), top_left_y-int(button_size*4/5)), (255,255,255), -1)
            tri_pts = [[top_left_x+int(button_size*7/10), top_left_y-int(button_size/2)],
                       [top_left_x+int(button_size*1/5), top_left_y-int(button_size/5)],
                       [top_left_x+int(button_size*1/5), top_left_y-int(button_size*4/5)]]
            cv2.fillPoly(frame, np.int32([tri_pts]), (255,255,255))

        # Forward button
        if i == 5:
            rewind_pts = [[top_left_x+int(button_size*4/5), h-m1-int(button_size/2)], #mid point of left triangle
                          [top_left_x+int(button_size*2/5), h-m1-int(button_size/5)], #lower point of left triangle
                          [top_left_x+int(button_size*2/5), h-m1-int(button_size*2/5-3)], # lower point of intersection
                          [top_left_x+int(button_size*1/5), h-m1-int(button_size/5)], #lower point of right triangle
                          [top_left_x+int(button_size*1/5), h-m1-int(button_size*4/5)], #upper point of right triangle
                          [top_left_x+int(button_size*2/5), h-m1-int(button_size*3/5+3)], #upper point of intersection
                          [top_left_x+int(button_size*2/5), h-m1-int(button_size*4/5)]] #upper point of left triangle
            pts = np.array(rewind_pts)
            cv2.fillPoly(frame, np.int32([pts]),(255,255,255))

    seeker_bar_top_left = (m0, h-(button_size+m1+10))
    seeker_bar_bottom_right= (w-m0, h-(button_size+m1+15))
    seeker_pos = int((current_frame/length)*(w-m0*2)+m0)
    seeker_tl = (seeker_pos, seeker_bar_top_left[1]-10)
    seeker_br = (seeker_pos+5, seeker_bar_bottom_right[1]+10)
    cv2.rectangle(frame, seeker_bar_top_left, seeker_bar_bottom_right, (53,53,53), -1)
    cv2.rectangle(frame, seeker_tl, seeker_br, (23,10,255), -1)

    return frame

def tuple_compare(tup1, tup2, relate):

    for i,j in zip(tup1, tup2):
        if not relate(i,j):
            return False

    return True


def add_current_frame_text(frame, current_frame, length, font):
    h,w,c = frame.shape
    cv2.putText(frame,str(current_frame)+"/"+str(length),(0,20), font, 0.8 ,(53,33,255),1,cv2.LINE_AA)

    return frame

def add_crop_frame(frame, crop_start, crop_end):
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    if crop_start != 0:
        cv2.putText(frame,"Crop start: "+str(crop_start+1),(0,50), font, 0.8 ,(53,33,255),1,cv2.LINE_AA)
    if crop_end != 0:
        cv2.putText(frame,"Crop end: "+str(crop_end+1),(0,80), font, 0.8 ,(53,33,255),1,cv2.LINE_AA)

    return frame


def crop_video_start_label(frame, current_frame, crop_start, crop_end, fvs, vid_name):

    frame_width = fvs.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = fvs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = fvs.stream.get(cv2.CAP_PROP_FPS)
    fvs2 = FileVideoStream(vid_name).start()
    output = cv2.VideoWriter(vid_name.split(".")[0]+"_out.avi", cv2.VideoWriter_fourcc('M','J','P','G'), int(fps), (int(frame_width),int(frame_height)))

    for i in range(crop_start, crop_end):
        fvs2.frameNo = i
        save_frame = fvs2.read()
        cv2.imshow("Saving", save_frame)
        output.write(save_frame)

    cv2.destroyAllWindows()

    return None
