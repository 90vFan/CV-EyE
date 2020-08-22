import os
from flask import Flask, render_template, Response
from camera import Camera

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def camera_gen(camera_instance):
    while True:
        frame = camera_instance.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    camera_generator = camera_gen(Camera())
    return Response(camera_generator,
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
