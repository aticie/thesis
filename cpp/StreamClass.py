import cv2
from Queue import Queue
from threading import Thread
import threading
import sys

class FileVideoStream:
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.rwd_frame = False
        self.fwd_frame = False
        self.curFrame = 0
        self.seek = False
        self.length = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):

        # keep looping infinitely
        while True:
            if self.rwd_frame:
                self.rwd_frame = False
                self.curFrame = max(self.curFrame - 2, 0)
                self.seek = True

            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if self.seek:    
                
                self.stream.set(cv2.CAP_PROP_POS_FRAMES, self.curFrame)
                self.Q.queue.clear()
                self.seek = False

            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self.Q.put(frame)

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
 
                # add the frame to the queue

    def read(self):
        self.curFrame += 1
        if self.curFrame == self.length:
            self.curFrame = 0
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

