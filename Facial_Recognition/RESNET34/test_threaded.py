# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import pickle
import cv2
import datetime
from threading import Thread
from multiprocessing import Process, Queue
import time


# from common import clock, draw_str, StatValue
# import video

class Face_Process(Process):

    def __init__(self, frame_queue, output_queue):
        Process.__init__(self)
        self.frame_queue = frame_queue
        self.output_queue = output_queue
        self.stop = False

    def get_frame(self):
        if not self.frame_queue.empty():
            return True, self.frame_queue.get()
        else:
            return False, None

    def stopProcess(self):
        self.stop = True

    def face_detect(self, frame):

        # loop over frames from the video file stream
        while True:
            # grab the frame from the threaded video stream
            frame = vs.read()
            # convert the input frame from BGR to RGB then resize it to have
            # a width of 750px (to speedup processing)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width=750)
            r = frame.shape[1] / float(rgb.shape[1])

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input frame, then compute
            # the facial embeddings for each face
            boxes = face_recognition.face_locations(rgb,
                                                    model=args["detection_method"])
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            # loop over the facial embeddings

            for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(data["encodings"],
                                                         encoding)
                name = "Unknown"

                # check to see if we have found a match
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)

                # update the list of names
                names.append(name)

            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom),
                              (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (10, 255, 40), 2)
                print("Person Identified: {}".format(name))
                print()

            if self.output_queue.full():
                self.output_queue.get_nowait()
            self.output_queue.put(rgb)

    def run(self):
        while not self.stop:
            ret, frame = self.get_frame()
            if ret:
                self.face_detect(frame)


if __name__ == '__main__':

    frame_sum = 0
    init_time = time.time()


    def put_frame(frame):
        if Input_Queue.full():
            Input_Queue.get_nowait()
        Input_Queue.put(frame)


    def cap_read(cv2_cap):
        ret, frame = cv2_cap.read()
        if ret:
            put_frame(frame)


    cap = cv2.VideoCapture(0)

    threadn = cv2.getNumberOfCPUs()

    threaded_mode = True

    process_list = []
    Input_Queue = Queue(maxsize=5)
    Output_Queue = Queue(maxsize=5)

    for x in range((threadn - 1)):
        face_process = Face_Process(frame_queue=Input_Queue, output_queue=Output_Queue)
        face_process.daemon = True
        face_process.start()
        process_list.append(face_process)

    ch = cv2.waitKey(1)
    cv2.namedWindow('Threaded Video', cv2.WINDOW_NORMAL)
    while True:
        cap_read(cap)

        if not Output_Queue.empty():
            result = Output_Queue.get()
            cv2.imshow('Threaded Video', result)
            ch = cv2.waitKey(5)

        if ch == ord(' '):
            threaded_mode = not threaded_mode
        if ch == 27:
            break
    cv2.destroyAllWindows()
    
    
class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default="encodings.pickle",
                help="path to serialized db of facial encodings")

ap.add_argument("-o", "--output", default="output/webcam_face_recognition_output.avi", type=str,
                help="path to output video")

ap.add_argument("-n", "--num-frames", type=int, default=500,
                help="# of frames to loop over for FPS test")

ap.add_argument("-y", "--display", type=int, default=1,
                help="whether or not to display output frame to screen")

ap.add_argument("-d", "--detection-method", type=str, default="hog",
                help="face detection model to use: either `hog` or `cnn`")

args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# grab a pointer to the video stream and initialize the FPS counter
# allow the camera sensor to warm up
print("[INFO] starting video and sampling frames stream...")
vs = WebcamVideoStream(src=0).start()
writer = None
time.sleep(0.3)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
