import copy
import queue
import threading
from typing import Optional

import cv2
import numpy
import time
import sys
sys.path.append("C:/Users/Toby/Desktop/CapstoneProject/enhanced_python_aprilgrid/src/")
from aprilgrid import Detector
from vmbpy import *

FRAME_QUEUE_SIZE = 10

# Initialize the detector
detector = Detector('t16h5b1')

def print_preamble():
    print('////////////////////////////////////////')
    print('/// VmbPy Multithreading Example ///////')
    print('////////////////////////////////////////\n')
    print(flush=True)


def create_dummy_frame() -> numpy.ndarray:
    cv_frame = numpy.zeros((50, 640, 1), numpy.uint8)
    cv_frame[:] = 0

    cv2.putText(cv_frame, 'No Stream available. Please connect a Camera.', org=(30, 30),
                fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

    return cv_frame


def try_put_frame(q: queue.Queue, cam: Camera, frame: Optional[Frame]):
    try:
        q.put_nowait((cam.get_id(), frame))

    except queue.Full:
        pass

# Thread Objects
class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)

        self.log = Log.get_instance()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()
        
        self.last_time = time.time()
        self.frame_count = 0

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        # This method is executed within VmbC context. All incoming frames
        # are reused for later frame acquisition. If a frame shall be queued, the
        # frame must be copied and the copy must be sent, otherwise the acquired
        # frame will be overridden as soon as the frame is reused.
        if frame.get_status() == FrameStatus.Complete:

            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                try_put_frame(self.frame_queue, cam, frame_cpy)
            
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_time

            if elapsed >= 1.0:
                print(f"[CAPTURE FPS] {self.frame_count:.2f} frames/sec")
                self.frame_count = 0
                self.last_time = current_time

        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        try:
            # Set camera setting here
            self.cam.ExposureAuto.set('Off')
            self.cam.ExposureTime.set(10000) 
            self.cam.Gain.set(20)
            self.cam.AcquisitionFrameRateEnable.set(True)
            self.cam.AcquisitionFrameRate.set(35.04)
            self.cam.AcquisitionFrameRateMode.set('Basic')
            self.cam.BinningVertical.set(1)
            self.cam.BinningHorizontal.set(1)
            # self.cam.BinningHorizontalMode.set('Average')
            # self.cam.BinningVerticalMode.set('Average')
            self.cam.set_pixel_format(PixelFormat.Mono8)
        except (AttributeError, VmbFeatureError):
            print("Bozo check bandwidth!")
            pass

    def run(self):
        self.log.info('Thread \'FrameProducer({})\' started.'.format(self.cam.get_id()))
        try:
            with self.cam:
                self.setup_camera()

                try:
                    self.cam.start_streaming(self)
                    self.killswitch.wait()

                finally:
                    self.cam.stop_streaming()

        except VmbCameraError:
            pass

        finally:
            try_put_frame(self.frame_queue, self.cam, None)

        self.log.info('Thread \'FrameProducer({})\' terminated.'.format(self.cam.get_id()))

def resize_if_required(frame: Frame) -> numpy.ndarray:
    # Helper function resizing the given frame, if it has not the required dimensions.
    # On resizing, the image data is copied and resized, the image inside the frame object
    # is untouched.
    cv_frame = frame.as_opencv_image()
    return cv_frame

class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue):
        self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self.last_time = time.time()
        self.frame_count = 0

    def run(self):
        IMAGE_CAPTION = 'Multithreading Example: Press <Enter> to exit'
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
                # cv2.imshow(IMAGE_CAPTION, numpy.concatenate(cv_images, axis=1))
                newimg = []
                for img in cv_images:
                   # Initialize the detector 
                   resized_img = cv2.resize(img, (1006, 759))
                   newimg.append(resized_img)

                cv2.imshow(IMAGE_CAPTION, numpy.concatenate(newimg, axis=1))
            # If there are no frames available, show dummy image instead
            else:
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_time

            if elapsed >= 1.0:  
                print(f"[DISPLAY FPS] {self.frame_count:.2f} frames/sec")
                self.frame_count = 0
                self.last_time = current_time

            # Check for shutdown condition
            if KEY_CODE_ENTER == cv2.waitKey(10):
                cv2.destroyAllWindows()
                alive = False

        self.log.info('\'FrameConsumer\' terminated.')


class Application:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()

    def __call__(self, cam: Camera, event: CameraEvent):
        # New camera was detected. Create FrameProducer, add it to active FrameProducers
        if event == CameraEvent.Detected:
            with self.producers_lock:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()].start()

        # An existing camera was disconnected, stop associated FrameProducer.
        elif event == CameraEvent.Missing:
            with self.producers_lock:
                producer = self.producers.pop(cam.get_id())
                producer.stop()
                producer.join()

    def run(self):
        log = Log.get_instance()
        consumer = FrameConsumer(self.frame_queue)

        vmb = VmbSystem.get_instance()
        vmb.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)

        log.info('\'Application\' started.')

        with vmb:
            # Construct FrameProducer threads for all detected cameras
            for cam in vmb.get_all_cameras():
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)

            # Start FrameProducer threads
            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()

            # Run the frame consumer to display the recorded images
            vmb.register_camera_change_handler(self)
            consumer.run()
            vmb.unregister_camera_change_handler(self)

            # Stop all FrameProducer threads
            with self.producers_lock:
                # Initiate concurrent shutdown
                for producer in self.producers.values():
                    producer.stop()

                # Wait for shutdown to complete
                for producer in self.producers.values():
                    producer.join()

        log.info('\'Application\' terminated.')


if __name__ == '__main__':
    print_preamble()
    app = Application()
    app.run()
