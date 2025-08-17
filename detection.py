import cv2
import time
import requests
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# URLs
esp32_url = os.getenv("ESP32_URL", "http://127.0.0.1/capture")
moisture_status_url = os.getenv("MOISTURE_URL", "http://127.0.0.1/moisture")

# Model
model_path = os.getenv("MODEL_PATH", "weights/best.pt")
try:
    model = YOLO(model_path)
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error: Could not load model. {e}")
    exit()

# Output video setup
output_filename = os.getenv("OUTPUT_VIDEO", "esp32_stream_record.avi")
frame_width, frame_height, fps = 640, 480, 10
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Telegram
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message):
    if not telegram_bot_token or not telegram_chat_id:
        print("Telegram credentials not set. Skipping alert.")
        return
    url = f'https://api.telegram.org/bot{telegram_bot_token}/sendMessage'
    try:
        res = requests.post(url, data={'chat_id': telegram_chat_id, 'text': message})
        if res.status_code != 200:
            print(f"Telegram error: {res.text}")
    except Exception as e:
        print(f"Telegram send failed: {e}")

def fetch_frame(url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                arr = np.frombuffer(res.content, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
        except Exception as e:
            print(f"[Attempt {attempt+1}/{retries}] Frame fetch error: {e}")
        time.sleep(delay)
    return None  

def get_moisture_status():
    try:
        res = requests.get(moisture_status_url)
        if res.status_code == 200 and res.text.strip():
            return res.text.strip()
    except Exception as e:
        print(f"Moisture fetch error: {e}")

# Start monitoring
send_telegram_alert("Wax moth detection system started. Monitoring in progress...")

detection_log = []
check_interval = timedelta(minutes=2)
min_confidence = 0.5
high_detection_count = 5
low_detection_count = 2
last_check_time = datetime.now()
connection_lost = False

try:
    while True:
        frame = fetch_frame(esp32_url)

        if frame is None:
            if not connection_lost:
                send_telegram_alert("Connection lost with ESP32 camera. Retrying...")
                connection_lost = True
            time.sleep(5)
            continue
        else:
            if connection_lost:
                send_telegram_alert("ESP32 camera reconnected. Resuming monitoring.")
                connection_lost = False

        frame = cv2.resize(frame, (frame_width, frame_height))
        out.write(frame)

        try:
            results = model.predict(source=frame, conf=0.1)
            wax_moth_detected = False

            if results:
                annotated_frame = results[0].plot()
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    if cls_name == 'wax_moth':
                        detection_log.append((datetime.now(), conf))
                        wax_moth_detected = True
                        break
            else:
                annotated_frame = frame
        except Exception as e:
            print(f"Inference error: {e}")
            annotated_frame = frame

        now = datetime.now()
        if now - last_check_time >= check_interval:
            recent_detections = [conf for t, conf in detection_log if now - t <= check_interval]
            moisture_status = get_moisture_status()

            if len(recent_detections) >= high_detection_count and all(c >= min_confidence for c in recent_detections):
                send_telegram_alert(f"Wax moth detected multiple times with high confidence.\nAction required.\nMoisture: {moisture_status}")
            elif len(recent_detections) >= low_detection_count:
                send_telegram_alert(f"Possible wax moth activity detected.\nPlease inspect the bee box.\nMoisture: {moisture_status}")
            else:
                send_telegram_alert(f"No significant wax moth activity detected.\nMoisture: {moisture_status}")

            detection_log.clear()
            last_check_time = now

        cv2.imshow("ESP32 Camera Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

        time.sleep(0.1)

except KeyboardInterrupt:
    print("User interruption received.")
finally:
    out.release()
    cv2.destroyAllWindows()
    print("Resources released.")
