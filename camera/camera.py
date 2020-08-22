import io
import time
import picamera
from base_camera import BaseCamera


class Camera(BaseCamera):
    @staticmethod
    def frames():
        """Initialize Camera and stream it as jpeg video for flask app
        Called by `BaseCamera._thread`
        """
        with picamera.PiCamera() as camera:
            camera.resolution = (640, 480)
            time.sleep(2)

            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, 'jpeg',
                                               use_video_port=True):
                stream.seek(0)
                yield stream.read()

                stream.seek(0)
                stream.truncate()
