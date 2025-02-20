## Based on the example code from the Vimba Python API
import sys
from typing import Optional
from queue import Queue
from vmbpy import *

# All frames will either be recorded in this format, or transformed to it before being displayed
opencv_display_format = PixelFormat.Bgr8


def print_usage():
    print('Usage:')
    print('    python asynchronous_grab_opencv.py [camera_id]')
    print('    python asynchronous_grab_opencv.py [/h] [-h]')
    print()
    print('Parameters:')
    print('    camera_id   ID of the camera to use (using first camera if not specified)')
    print()


def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')
    if usage:
        print_usage()
    sys.exit(return_code)


def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)
    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)
    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)
    return None if argc == 0 else args[0]


def get_camera(camera_id: Optional[str]) -> Camera:
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)
            except VmbCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))
        else:
            cams = vmb.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')
            return cams[0]


def setup_camera(cam: Camera):
    with cam:
        try:
            # Set camera setting here
            cam.ExposureAuto.set('Off')
            cam.ExposureTime.set(20000) 
            cam.Gain.set(16)
            cam.AcquisitionFrameRateEnable.set(True)
            cam.AcquisitionFrameRate.set(35.0)
            cam.AcquisitionFrameRateMode.set('Basic')
            cam.BinningHorizontal.set(1)
            cam.BinningVertical.set(1)
        except (AttributeError, VmbFeatureError):
            pass
        except (AttributeError, VmbFeatureError):
            pass
        except (AttributeError, VmbFeatureError):
            pass


def setup_pixel_format(cam: Camera):
    # Query available pixel formats. Prefer color formats over monochrome formats
    cam_formats = cam.get_pixel_formats()
    print("Available Pixel Formats:", cam_formats)
    cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    convertible_color_formats = tuple(f for f in cam_color_formats
                                      if opencv_display_format in f.get_convertible_formats())
    cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    convertible_mono_formats = tuple(f for f in cam_mono_formats
                                     if opencv_display_format in f.get_convertible_formats())
    # if OpenCV compatible color format is supported directly, use that
    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)
    # else if existing color format can be converted to OpenCV format do that
    elif convertible_color_formats:
        cam.set_pixel_format(convertible_color_formats[0])
    # fall back to a mono format that can be converted
    elif convertible_mono_formats:
        cam.set_pixel_format(convertible_mono_formats[0])
    else:
        abort('Camera does not support an OpenCV compatible format. Abort.')


class Handler:
    def __init__(self):
        self.display_queue = Queue(10)
    def get_image(self):
        return self.display_queue.get(True)
    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            print('{} acquired {}'.format(cam, frame), flush=True)
            print("FPS:", cam.AcquisitionFrameRate.get_range()[1])
            # Convert frame if it is not already the correct format
            if frame.get_pixel_format() == opencv_display_format:
                display = frame
            else:
                # This creates a copy of the frame. The original `frame` object can be requeued
                # safely while `display` is used
                display = frame.convert_pixel_format(opencv_display_format)
            self.display_queue.put(display.as_opencv_image(), True)
        cam.queue_frame(frame)



def main():
    cam_id = parse_args()
    with VmbSystem.get_instance():
        with get_camera(cam_id) as cam:
            # setup general camera settings and the pixel format in which frames are recorded
            setup_camera(cam)
            setup_pixel_format(cam)
            handler = Handler()
            try:
                # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                cam.start_streaming(handler=handler, buffer_count=10)
                msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
                import cv2
                ENTER_KEY_CODE = 13
                while True:
                    key = cv2.waitKey(1)
                    if key == ENTER_KEY_CODE:
                        cv2.destroyWindow(msg.format(cam.get_name()))
                        break
                    display = handler.get_image()
                    display = cv2.resize(display, (640, 480)) # Resize to fit screen, doesn't change camera resolution
                    cv2.imshow(msg.format(cam.get_name()), display)
            finally:
                cam.stop_streaming()


if __name__ == '__main__':
    main()
