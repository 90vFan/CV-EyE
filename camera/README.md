
## Debug
```sh
$ python3 app.py
[DEBUG] current thread is: 3069340368
 * Serving Flask app "app" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
192.168.0.107 - - [18/Jul/2020 16:56:32] "GET / HTTP/1.1" 200 -
[DEBUG] init class BaseCamera
[INFO] Starting background camera thread
[DEBUG] get frame
192.168.0.107 - - [18/Jul/2020 16:56:35] "GET /video_feed HTTP/1.1" 200 -
[INFO] Stopping camera thread due to inactivity
192.168.0.107 - - [18/Jul/2020 16:57:00] "GET / HTTP/1.1" 200 -
[DEBUG] init class BaseCamera
[INFO] Starting background camera thread
[DEBUG] get frame
192.168.0.107 - - [18/Jul/2020 16:57:03] "GET /video_feed HTTP/1.1" 200 -
```