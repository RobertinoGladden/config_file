import cv2
import torch
from ultralytics import YOLO
import time
import os
import multiprocessing

multiprocessing.freeze_support()

# --- Konfigurasi ---
RTSP_URL = "rtsp://36.92.47.218:7430/video4"
OUTPUT_DIR = r'C:\Users\lapt1\Downloads\Hailo AI\lib\capture-img'
COOLDOWN_SECONDS = 5
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# --- Model Deteksi Objek ---
YOLO_MODEL_PATH = r'C:\Users\lapt1\Downloads\Hailo AI\src\model\people-model\people.pt'
model = YOLO(YOLO_MODEL_PATH)

# --- Bounding Box Parameter Video 4 ---
PARAM_BOX_X_MIN = 1000
PARAM_BOX_Y_MIN = 280
PARAM_BOX_X_MAX = 2000
PARAM_BOX_Y_MAX = 1200

# Buat direktori output jika belum ada
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Untuk debouncing
last_capture_time = 0

# --- Fungsi Bantuan untuk Cek Interseksi Bounding Box ---
def intersect(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    return inter_x_max > inter_x_min and inter_y_max > inter_y_min

# --- Main Application Logic ---
def run_rtsp_detection():
    global last_capture_time

    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka stream RTSP dari {RTSP_URL}")
        print("Pastikan URL RTSP benar dan kamera dapat diakses.")
        return

    print("Stream RTSP berhasil dibuka.")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolusi Stream: {frame_width}x{frame_height}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Menggunakan device: {device}")

    print("Nama kelas yang dikenal model:")
    print(model.names)
    
    PERSON_CLASS_ID = 0 
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame, mencoba menghubungkan ulang...")
                cap.release()
                cap = cv2.VideoCapture(RTSP_URL)
                time.sleep(1)
                continue

            display_frame = frame.copy() 

            # Inferensi YOLO
            results = model(frame, stream=True, device=device, verbose=False, conf=DETECTION_CONFIDENCE_THRESHOLD)

            person_detected_in_param_box = False

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()

                for i, box in enumerate(boxes):
                    cls_id = int(classes[i])
                    conf = confidences[i]
                    x1, y1, x2, y2 = map(int, box)

                    if cls_id == PERSON_CLASS_ID and intersect([x1, y1, x2, y2], [PARAM_BOX_X_MIN, PARAM_BOX_Y_MIN, PARAM_BOX_X_MAX, PARAM_BOX_Y_MAX]):
                        person_detected_in_param_box = True
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{model.names[cls_id]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.rectangle(display_frame, (PARAM_BOX_X_MIN, PARAM_BOX_Y_MIN), (PARAM_BOX_X_MAX, PARAM_BOX_Y_MAX), (255, 0, 0), 2)
            cv2.putText(display_frame, "Capture Zone", (PARAM_BOX_X_MIN, PARAM_BOX_Y_MIN - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            current_time = time.time()
            if person_detected_in_param_box and (current_time - last_capture_time > COOLDOWN_SECONDS):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                crop_y1 = max(0, PARAM_BOX_Y_MIN)
                crop_y2 = min(frame_height, PARAM_BOX_Y_MAX)
                crop_x1 = max(0, PARAM_BOX_X_MIN)
                crop_x2 = min(frame_width, PARAM_BOX_X_MAX)

                cropped_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                image_filename = os.path.join(OUTPUT_DIR, f"cropped_person_in_box_{timestamp}.jpg")
                cv2.imwrite(image_filename, cropped_img)
                print(f"Captured cropped image: {image_filename}")
                
                last_capture_time = current_time

            # Tampilkan display_frame
            cv2.imshow("RTSP Stream - People Detector", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Aplikasi dimatikan.")

if __name__ == "__main__":
    run_rtsp_detection()