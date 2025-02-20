# frame_producer.py
import copy
import queue
import threading
import time
from typing import Optional
from vmbpy import *  # Ensure the necessary VmbPy imports are here

def try_put_frame(q: queue.Queue, cam: Camera, frame: Optional[Frame]):
    try:
        q.put_nowait((cam.get_id(), frame))
    except queue.Full:
        pass

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
            self.cam.ExposureAuto.set('Off')
            self.cam.ExposureTime.set(10000) 
            self.cam.Gain.set(20)
            self.cam.AcquisitionFrameRateEnable.set(True)
            self.cam.AcquisitionFrameRate.set(35.04)
            self.cam.AcquisitionFrameRateMode.set('Basic')
            self.cam.BinningVertical.set(1)
            self.cam.BinningHorizontal.set(1)
            self.cam.set_pixel_format(PixelFormat.Mono8)
        except (AttributeError, VmbFeatureError):
            print("Error configuring camera!")
            pass

    def run(self):
        self.log.info(f"Thread 'FrameProducer({self.cam.get_id()})' started.")
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

        self.log.info(f"Thread 'FrameProducer({self.cam.get_id()})' terminated.")
