from flask import Flask, Response, jsonify, request
import os, cv2, numpy as np, requests, time
from collections import deque, Counter
from datetime import datetime
from flask_cors import CORS
from ultralytics import YOLO
from yt_dlp import YoutubeDL
from dotenv import load_dotenv

# Load konfigurasi dari .env
load_dotenv()

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Konstanta dan konfigurasi
TARGET_CLASSES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
TARGET_CLASSES_RICKSHAW = {0: "Rickshaw"}

WEIGHTS = {
    "Person": 0.25, "Bicycle": 0.5, "Motorcycle": 0.5,
    "Car": 1, "Rickshaw": 0.75, "Bus": 3, "Truck": 3.5
}

ALPHA = 1.0
BETA = 0.1
np.random.seed(42)

CLASS_COLORS = {
    label: np.random.randint(0, 255, 3).tolist()
    for label in list(TARGET_CLASSES.values()) + list(TARGET_CLASSES_RICKSHAW.values())
}

# Load model
model_default = YOLO("models/yolo11m.pt")
model_rickshaw = YOLO("models/best.pt")

# Histori deteksi
total_counts = {}
direction_history = {}
position_history = {}
last_known_direction = {}

# Fungsi bantu YouTube
def get_youtube_stream_url(youtube_url):
    ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True, 'no_warnings': True}
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            return info_dict['url']
    except Exception as e:
        print(f"[ERROR] YouTube stream failed: {e}")
        return None

# Fungsi utama pemrosesan frame
def generate_frames(cap):
    detected_ids = set()
    global total_counts, position_history, direction_history, last_known_direction
    total_counts = {}
    last_sent = time.time()
    movement_threshold = 8

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            print("[WARNING] Frame kosong atau tidak bisa dibaca.")
            break

        frame = cv2.resize(frame, (640, 480))

        try:
            results_default = model_default.track(frame, persist=True, conf=0.6, iou=0.5, classes=list(TARGET_CLASSES.keys()))
            results_rickshaw = model_rickshaw.track(frame, persist=True, conf=0.7, iou=0.5, classes=list(TARGET_CLASSES_RICKSHAW.keys()))
        except Exception as e:
            print(f"[ERROR] Gagal tracking: {e}")
            continue

        combined_results = [(results_default[0], 'default'), (results_rickshaw[0], 'rickshaw')]

        for results, source in combined_results:
            label_map = TARGET_CLASSES if source == 'default' else TARGET_CLASSES_RICKSHAW

            if results.boxes is not None:
                for i in range(len(results.boxes)):
                    box = results.boxes.xyxy.cpu().numpy()[i]
                    cls_id = int(results.boxes.cls.cpu().numpy()[i])
                    obj_id = int(results.boxes.id.cpu().numpy()[i]) if results.boxes.id is not None else None
                    label = label_map.get(cls_id)

                    if label and obj_id is not None:
                        obj_uid = f"{source}_{obj_id}"
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        position_history.setdefault(obj_uid, deque(maxlen=5)).append((cx, cy))

                        direction = last_known_direction.get(obj_uid, "Unknown")
                        if len(position_history[obj_uid]) >= 2:
                            prev_cx, prev_cy = position_history[obj_uid][0]
                            dx, dy = cx - prev_cx, cy - prev_cy
                            if abs(dx) > movement_threshold or abs(dy) > movement_threshold:
                                if abs(dx) > abs(dy):
                                    direction = "East" if dx > 0 else "West"
                                else:
                                    direction = "South" if dy > 0 else "North"

                        direction_history.setdefault(obj_uid, deque(maxlen=5)).append(direction)
                        direction = Counter(direction_history[obj_uid]).most_common(1)[0][0]
                        last_known_direction[obj_uid] = direction

                        if obj_uid not in detected_ids and direction != "Unknown":
                            detected_ids.add(obj_uid)
                            total_counts.setdefault(label, {}).setdefault(direction, 0)
                            total_counts[label][direction] += 1

                        color = CLASS_COLORS.get(label, [0, 255, 0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} {obj_id} {direction}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # Kirim data setiap 10 detik
        if time.time() - last_sent > 10 and total_counts:
            try:
                detection_at = datetime.utcnow().isoformat()
                load_per_direction = {}
                person_count_per_direction = {}

                for label, directions in total_counts.items():
                    for direction, count in directions.items():
                        weight = WEIGHTS.get(label, 1)
                        load_per_direction.setdefault(direction, 0)
                        load_per_direction[direction] += count * weight
                        if label == "Person":
                            person_count_per_direction[direction] = person_count_per_direction.get(direction, 0) + count

                final_load = {
                    direction: ALPHA * load_per_direction[direction] + BETA * person_count_per_direction.get(direction, 0)
                    for direction in load_per_direction
                }

                payload = {
                    "detections": total_counts,
                    "detection_at": detection_at,
                    "load_per_direction": final_load
                }

                api_url = os.getenv("API_URL", "http://127.0.0.1:8000/api/store-detections")
                api_token = os.getenv("API_TOKEN")

                headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
                response = requests.post(api_url, json=payload, headers=headers)
                print(f"[SEND] Deteksi terkirim ke Laravel ({response.status_code}): {response.text}")

                last_sent = time.time()
                total_counts.clear()
                detected_ids.clear()
                position_history.clear()
                direction_history.clear()
                last_known_direction.clear()
            except Exception as e:
                print(f"[ERROR] Gagal kirim ke Laravel: {e}")

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/video_feed")
def video_feed():
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
        return jsonify({"error": "No valid stream source provided"}), 400

    if not cap or not cap.isOpened():
        return jsonify({"error": "Failed to open video stream"}), 400

    return Response(generate_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/counts")
def counts():
    return jsonify(total_counts)

@app.route("/send_data", methods=["POST"])
def send_data():
    try:
        detection_at = datetime.utcnow().isoformat()
        payload = {
            "detections": total_counts,
            "detection_at": detection_at
        }
        api_url = os.getenv("API_URL", "http://127.0.0.1:8000/api/store-detections")
        api_token = os.getenv("API_TOKEN")
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        response = requests.post(api_url, json=payload, headers=headers)
        return jsonify({"status": "sent", "response": response.json()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
