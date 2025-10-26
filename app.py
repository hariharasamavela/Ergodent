from flask import Flask, render_template, Response, redirect, url_for, jsonify, request
import cv2
import os
from ergonomics import process_frame

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

cap = None
camera_on = False
video_mode = False
upload_mode = False
latest_warnings = []
last_ip_url = ""  # store last entered IP URL


def gen_frames():
    """Generator for streaming frames from camera or uploaded video."""
    global cap, camera_on, video_mode, latest_warnings
    while (camera_on or video_mode) and cap is not None and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame, warnings = process_frame(frame)
        latest_warnings = warnings  # update latest warnings
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template("index.html",
                           camera_on=camera_on,
                           video_mode=video_mode,
                           upload_mode=upload_mode,
                           last_ip_url=last_ip_url)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera')
def start_camera():
    global cap, camera_on, video_mode, upload_mode
    if not camera_on:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            camera_on = True
            video_mode = False
            upload_mode = False
    return redirect(url_for('index'))


@app.route('/start_ip_camera', methods=['POST'])
def start_ip_camera():
    global cap, camera_on, video_mode, upload_mode, last_ip_url
    ip_url = request.form.get("ip_url")
    if not ip_url:
        return redirect(url_for('index'))

    last_ip_url = ip_url  # save last entered URL

    # Release previous stream
    if cap is not None:
        cap.release()

    cap = cv2.VideoCapture(ip_url)
    if cap.isOpened():
        camera_on = True
        video_mode = False
        upload_mode = False
        print(f"✅ IP camera started: {ip_url}")
    else:
        camera_on = False
        print(f"❌ Failed to open IP camera: {ip_url}")

    return redirect(url_for('index'))


@app.route('/start_maxilla')
def start_maxilla():
    # just redirect to IP camera page, user can enter URL
    return redirect(url_for('index'))


@app.route('/start_mandible')
def start_mandible():
    global cap, camera_on, video_mode, upload_mode
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        camera_on = True
        video_mode = False
        upload_mode = False
        print("✅ Switched to laptop webcam (Mandible).")
    else:
        camera_on = False
        print("❌ Failed to open laptop webcam.")
    return redirect(url_for('index'))


@app.route('/stop_camera')
def stop_camera():
    global cap, camera_on, video_mode, upload_mode
    if camera_on or video_mode or upload_mode:
        camera_on = False
        video_mode = False
        upload_mode = False
        if cap is not None:
            cap.release()
            cap = None
    return redirect(url_for('index'))


@app.route('/choose_upload')
def choose_upload():
    global upload_mode, camera_on, video_mode
    upload_mode = True
    camera_on = False
    video_mode = False
    return redirect(url_for('index'))


@app.route('/upload_video', methods=["POST"])
def upload_video():
    global cap, video_mode, camera_on, upload_mode
    file = request.files.get("video")
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            video_mode = True
            camera_on = False
            upload_mode = False
    return redirect(url_for('index'))


@app.route('/warnings')
def get_warnings():
    return jsonify(latest_warnings)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
