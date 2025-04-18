from flask import Flask, Response, jsonify, request
import os
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from flask_cors import CORS
from yt_dlp import YoutubeDL
from datetime import datetime
import time

app = Flask(__name__)
CORS(app)

TARGET_CLASSES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
TARGET_CLASSES_RICKSHAW = {0: "Rickshaw"}

np.random.seed(42)
CLASS_COLORS = {
    label: np.random.randint(0, 255, 3).tolist()
    for label in list(TARGET_CLASSES.values()) + list(TARGET_CLASSES_RICKSHAW.values())
}

total_counts = {}

def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            return info_dict['url']
    except Exception as e:
        print(f"Error retrieving YouTube stream: {e}")
        return None

def generate_frames(cap):
    detected_ids = set()
    global total_counts
    total_counts = {}
    last_sent = time.time()

    model_default = YOLO("models/yolo11s.pt")
    model_rickshaw = YOLO("models/best.pt")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results_default = model_default.track(frame, persist=True, conf=0.5, iou=0.5, classes=list(TARGET_CLASSES.keys()))
        results_rickshaw = model_rickshaw.track(frame, persist=True, conf=0.6, iou=0.5, classes=list(TARGET_CLASSES_RICKSHAW.keys()))

        combined_results = [(results_default, 'default'), (results_rickshaw, 'rickshaw')]

        for results, source in combined_results:
            label_map = TARGET_CLASSES if source == 'default' else TARGET_CLASSES_RICKSHAW

            for i in range(len(results.boxes)):
                box = results.boxes.xyxy.cpu().numpy()[i]
                cls_id = int(results.boxes.cls.cpu().numpy()[i])
                obj_id = int(results.boxes.id.cpu().numpy()[i]) if results.boxes.id is not None else None
                label = label_map.get(cls_id)

                if label and obj_id is not None:
                    obj_uid = f"{source}_{obj_id}"
                    if obj_uid not in detected_ids:
                        detected_ids.add(obj_uid)
                        total_counts[label] = total_counts.get(label, 0) + 1

                    x1, y1, x2, y2 = map(int, box)
                    color = CLASS_COLORS.get(label, [0, 255, 0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if time.time() - last_sent > 10 and total_counts:
            try:
                detection_at = datetime.utcnow().isoformat()
                payload = {
                    "detections": total_counts,
                    "detection_at": detection_at
                }
                api_url = os.environ.get("API_URL", "http://127.0.0.1:8000/api/store-detections")
                response = requests.post(api_url, json=payload)
                print(f"[SEND] Deteksi terkirim ke Laravel ({response.status_code}): {response.text}")
                last_sent = time.time()
                total_counts.clear()
                detected_ids.clear()
            except Exception as e:
                print(f"[ERROR] Gagal kirim ke Laravel: {e}")

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/video_feed")
def video_feed():
    camera_id = request.args.get("camera_id")
    cctv_url = request.args.get("cctv_url")
    youtube_url = request.args.get("youtube_url")

    cap = None

    if youtube_url:
        stream_url = get_youtube_stream_url(youtube_url)
        if stream_url:
            cap = cv2.VideoCapture(stream_url)
        else:
            return jsonify({"error": "Failed to load YouTube stream"}), 400
    elif cctv_url:
        cap = cv2.VideoCapture(cctv_url)
    else:
        cam_id = int(camera_id) if camera_id is not None else 0
        cap = cv2.VideoCapture(cam_id)

    if not cap or not cap.isOpened():
        return jsonify({"error": "Failed to open video stream"}), 400

    return Response(generate_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/counts")
def counts():
    return jsonify(total_counts)

@app.route("/send_data", methods=["POST"])
def send_data():
    api_url = os.environ.get("API_URL", "http://127.0.0.1:8000/api/store-detections")
    try:
        detection_at = datetime.utcnow().isoformat()
        payload = {
            "detections": total_counts,
            "detection_at": detection_at
        }
        response = requests.post(api_url, json=payload)
        return jsonify({"status": "sent", "response": response.json()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
