import time
import threading


class CameraEvent(object):
    """An Event-like class that signals all active clients when a new frame is available.
    """
    def __init__(self):
        self.events = {}
        self.identity = threading.get_ident()
        print(f'[DEBUG] current thread is: {self.identity}')

    def wait(self):
        """Invoked from each client's thread to wait for the next frame
        """
        # identity = threading.get_ident()  # current thread id
        identity = self.identity
        if identity not in self.events:
            self.events[identity] = {'evt': threading.Event(), 'time': time.time()}

        # Block until nother thread calls set() to set the flag to true
        return self.events[identity]['evt'].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available.
        """
        now = time.time()
        remove_id = None
        for identity, event in self.events.items():
            if not event['evt'].is_set():
                # if the client event is not set
                # set it and update timestamp to be now
                # `wait` threads will not block
                event['evt'].set()
                event['time'] = now
            else:
                # if the client event is already set, this means the client
                #   did not process a previous frame
                # if the event stays set for more than 5 seconds,
                #   then assume the client is gone and "remove" it
                if now - event['time'] > 5:
                    remove_id = identity
        if remove_id:
            del self.events[remove_id]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        # Reset the internal flag to false
        # identity = threading.get_ident()
        self.events[self.identity]['evt'].clear()


class BaseCamera(object):
    thread = None       # background thread that reads frames from camers
    frame = None        # current frame is stored here by background thread
    last_access = 0     # time of last client access to the camera
    event = CameraEvent()

    def __init__(self):
        # print(f'[DEBUG] Current threads active_count: {threading.active_count()}')
        if BaseCamera.thread is None:
            # init camera thread
            print(f'[INFO] Initialize a new Camera thread ...')
            BaseCamera.last_access = time.time()

            # start _thread to capture camera frame from camera generator frames()
            BaseCamera.thread = threading.Thread(target=self._thread)
            BaseCamera.thread.start()

            # wait for camera frame
            while self.get_frame() is None:
                print(f'[DEBUG] wait for camera frame')
                time.sleep(0)

    def get_frame(self):
        """called by generator `camera_gen` in flask app.py
        """
        # new frame `last_access` time
        BaseCamera.last_access = time.time()

        # wait for a signal from the camera thread
        BaseCamera.event.wait()
        # threads calling wait() will block until set() is called
        #   to set the internal flag to true again.
        BaseCamera.event.clear()

        return BaseCamera.frame

    @staticmethod
    def frames():
        raise RuntimeError('Must be implementd by subclasses')

    @classmethod
    def _thread(cls):
        """Camera background thread"""
        print('[INFO] Starting background Camera thread ...')
        # initialize Camera and capture the frames generator
        frames_iterator = cls.frames()
        for frame in frames_iterator:
            BaseCamera.frame = frame
            BaseCamera.event.set()
            time.sleep(0)

            if time.time() - BaseCamera.last_access > 10:
                frames_iterator.close()
                print('[INFO] Stopping Camera thread due to inactivity ...')
                break

        # clear current Camera thread
        BaseCamera.thread = None