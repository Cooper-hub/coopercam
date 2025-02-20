import queue
import threading
from vmbpy import *
from frame_producer import FrameProducer
from frame_consumer import FrameConsumer
FRAME_QUEUE_SIZE = 10


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