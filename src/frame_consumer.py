import queue
import time
import numpy
import cv2
import threading
from vmbpy import *  # Or import only the necessary modules for your class
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
relative_path = os.path.join(script_dir, '..')
normalized_path = os.path.abspath(relative_path) 
sys.path.append(normalized_path)

from aprilgrid import Detector
# Initialize the detector
detector = Detector('t16h5b1')

def create_dummy_frame() -> numpy.ndarray:
    cv_frame = numpy.zeros((50, 640, 1), numpy.uint8)
    cv_frame[:] = 0

    cv2.putText(cv_frame, 'No Stream available. Please connect a Camera.', org=(30, 30),
                fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

    return cv_frame

class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue):
        self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self.last_time = time.time()
        self.frame_count = 0
        self.frame_accumulated = 0  # To accumulate frame count for averaging

    def run(self):
        IMAGE_CAPTION = 'Downscaled Preview: Press <Enter> to exit'
        KEY_CODE_ENTER = 13

        frames = {}
        alive = True

        self.log.info('\'FrameConsumer\' started.')

        while alive:
            # Update current state by dequeuing all currently available frames.
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame = self.frame_queue.get_nowait()

                except queue.Empty:
                    break

                # Add/Remove frame from current state.
                if frame:
                    frames[cam_id] = frame

                else:
                    frames.pop(cam_id, None)

                frames_left -= 1

            # Construct image by stitching frames together.
            if frames:
                cv_images = [frames[cam_id].as_opencv_image() for cam_id in sorted(frames.keys())]
                
                newimg = []
                for img in cv_images:
                   resized_img = cv2.resize(img, (1006, 759))
                   newimg.append(resized_img)

                cv2.imshow(IMAGE_CAPTION, numpy.concatenate(newimg, axis=1))
            # If there are no frames available, show dummy image instead
            else:
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            self.frame_count += 1
            self.frame_accumulated += 1  # Increment the accumulated frame count
            current_time = time.time()
            elapsed = current_time - self.last_time

            if self.frame_accumulated >= 70:  # Check if we have processed 70 frames
                avg_fps = self.frame_count / elapsed  # Calculate the FPS for the last 70 frames
                print(f"[DISPLAY FPS (avg over 70 frames)] {avg_fps:.2f} frames/sec")
                
                # Reset the counters
                self.frame_count = 0
                self.frame_accumulated = 0  # Reset accumulated frame count
                self.last_time = current_time  # Reset the time for the next set of 70 frames

            # Check for shutdown condition
            if KEY_CODE_ENTER == cv2.waitKey(10):
                cv2.destroyAllWindows()
                alive = False

        self.log.info('\'FrameConsumer\' terminated.')
