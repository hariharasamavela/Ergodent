# ergonomics_modified_with_tts_final.py
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from math import degrees, acos
import pyttsx3
import time
import threading

mp_pose = mp.solutions.pose

# === Thresholds ===
NECK_FORWARD_LIMIT = 140
BACK_LEAN_LIMIT = 15
ELBOW_WRIST_TOL = 15  # degrees tolerance for arm parallelism

# === Initialize YOLO + Pose ===
model = YOLO("yolov8n.pt")
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Initialize text-to-speech (TTS) engine ===
engine = pyttsx3.init()
engine.setProperty('rate', 170)  # speaking speed
engine.setProperty('volume', 1.0)  # max volume

last_spoken = {}
SPEAK_COOLDOWN = 5.0  # seconds between repeats for the same message


# === Helper functions for TTS ===
def _speak_blocking(text):
    """Speaks the given text (blocking call, runs in background thread)."""
    engine.say(text)
    engine.runAndWait()


def speak_warning(text):
    """Speak a warning if cooldown period for that message has expired."""
    now = time.time()

    # each unique message has its own cooldown
    if text in last_spoken and (now - last_spoken[text]) < SPEAK_COOLDOWN:
        return  # recently spoken; skip

    last_spoken[text] = now
    threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()


# === Geometry helpers ===
def to_point(landmark, shape, bbox_offset=(0, 0)):
    h, w = shape[:2]
    x = int(landmark.x * w) + bbox_offset[0]
    y = int(landmark.y * h) + bbox_offset[1]
    return np.array([x, y])


def angle_with_horizontal(p1, p2):
    vec = p2 - p1
    if np.linalg.norm(vec) == 0:
        return 0
    horizontal = np.array([1, 0])
    cosang = np.dot(vec, horizontal) / (np.linalg.norm(vec) * np.linalg.norm(horizontal))
    cosang = np.clip(cosang, -1.0, 1.0)
    return degrees(acos(cosang))


def vertical_angle(p1, p2):
    vec = p2 - p1
    if np.linalg.norm(vec) == 0:
        return 0
    vertical = np.array([0, 1])
    cosang = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical))
    cosang = np.clip(cosang, -1.0, 1.0)
    return degrees(acos(cosang))


# === Main frame processor ===
def process_frame(frame):
    img = frame.copy()
    warnings = []
    h, w = img.shape[:2]

    # detect person(s)
    results = model(img, verbose=False)
    persons = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append((x1, y1, x2, y2))

    if len(persons) == 0:
        warnings.append("No person detected")
        return img, warnings

    # take first (largest) detected person
    person_box = sorted(persons, key=lambda b: (b[1], -(b[3] - b[1])))[0]
    x1, y1, x2, y2 = person_box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return img, ["Invalid crop region"]

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = pose.process(crop_rgb)

    if not res.pose_landmarks:
        return img, ["No pose landmarks detected"]

    lm = res.pose_landmarks.landmark

    def gp(idx):
        return to_point(lm[idx], crop.shape, bbox_offset=(x1, y1))

    # === Key landmarks ===
    nose = gp(mp_pose.PoseLandmark.NOSE.value)
    left_sh = gp(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_sh = gp(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    shoulders_mid = (left_sh + right_sh) / 2
    left_hip = gp(mp_pose.PoseLandmark.LEFT_HIP.value)
    right_hip = gp(mp_pose.PoseLandmark.RIGHT_HIP.value)
    hips_mid = (left_hip + right_hip) / 2
    right_elbow = gp(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    right_wrist = gp(mp_pose.PoseLandmark.RIGHT_WRIST.value)

    # === Calculate angles ===
    neck_angle = vertical_angle(shoulders_mid, nose)
    back_angle = vertical_angle(shoulders_mid, hips_mid)
    right_arm_angle = angle_with_horizontal(right_elbow, right_wrist)

    # === Draw bounding box ===
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, "Person", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # === Check ergonomic conditions ===
    if neck_angle > NECK_FORWARD_LIMIT:
        warnings.append("⚠ Correct Neck Posture")
    if back_angle > BACK_LEAN_LIMIT:
        warnings.append("⚠ Correct Back Posture")
    if abs(right_arm_angle - 0) > ELBOW_WRIST_TOL and abs(right_arm_angle - 180) > ELBOW_WRIST_TOL:
        warnings.append("⚠ Right arm not parallel")

    # === Speak warnings with stagger to avoid overlap ===
    for i, warn in enumerate(warnings):
        if "Neck" in warn:
            threading.Timer(i * 0.5, speak_warning, args=("Please correct your neck posture.",)).start()
        elif "Back" in warn:
            threading.Timer(i * 0.5, speak_warning, args=("Please correct your back posture.",)).start()
        elif "arm" in warn or "Arm" in warn:
            threading.Timer(i * 0.5, speak_warning, args=("Your right arm is not aligned properly.",)).start()

    # === Overlay angles (top-right corner) ===
    overlay_x = w - 300
    y_text = 40
    cv2.putText(img, f"Neck Angle: {neck_angle:.1f}°", (overlay_x, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_text += 30
    cv2.putText(img, f"Back Angle: {back_angle:.1f}°", (overlay_x, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_text += 30
    cv2.putText(img, f"Right Arm Angle: {right_arm_angle:.1f}°", (overlay_x, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # # === Display warnings visually ===
    # y_off = 30
    # for warn in warnings:
    #     cv2.putText(img, warn, (30, y_off),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    #     y_off += 30

    return img, warnings
