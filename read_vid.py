import numpy as np
import cv2
import math

cap = cv2.VideoCapture('cam5_out.avi')

def add_buttons(frame):

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
        # Rewind by frame
        if i == 1:
            print((top_left_x+button_size/5, top_left_y+button_size/5), (top_left_x+button_size*3/10, top_left_y+button_size*4/5))
            cv2.rectangle(frame, (top_left_x+button_size/5, top_left_y-button_size/5),(top_left_x+button_size*3/10, top_left_y-button_size*4/5), (255,255,255), -1)
            tri_pts = [[top_left_x+button_size*3/10,top_left_y-button_size/2],
                       [top_left_x+button_size*4/5, top_left_y-button_size/5],
                       [top_left_x+button_size*4/5, top_left_y-button_size*4/5]]
            cv2.fillPoly(frame,np.int32([tri_pts]),(255,255,255))

        # Resume button
        if i == 3:
            tri_pts = [[top_left_x+button_size*4/5, top_left_y-button_size/2], #mid point of triangle
                       [top_left_x+button_size*1/5, top_left_y-button_size*4/5], #lower point of triangle
                       [top_left_x+button_size*1/5, top_left_y-button_size*1/5]]
            cv2.fillPoly(frame, np.int32([tri_pts]), (255,255,255))
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

    # Pause button
    cv2.rectangle(frame, (m0*3+button_size*2+button_size/5,h-m1-button_size*4/5), (m0*3+button_size*2+button_size*2/5,h-m1-button_size*1/5), (255,255,255), -1)
    cv2.rectangle(frame, (m0*3+button_size*2+button_size*3/5,h-m1-button_size*4/5), (m0*3+button_size*2+button_size*4/5,h-m1-button_size*1/5), (255,255,255), -1)


    return frame
while(True):
    # Capture frame-by-frame
    for i in range(100):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i);
        ret, frame = cap.read()

        frame = add_buttons(frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()