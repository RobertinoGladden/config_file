import cv2
import torch
from ultralytics import YOLO
import time
import os

# --- Konfigurasi ---
rtsp_url = "rtsp://36.92.47.218:7430/video4" 
detectconf_threshold = 0.5 

yolo_model = r'/home/jetson/yolo11m.pt' 

param_box_x_min = 1000
param_box_y_min = 280
param_box_x_max = 2000
param_box_y_max = 1200

def intersect(box1, box2):
    """
    Mengecek apakah dua bounding box berpotongan.
    box format: [x_min, y_min, x_max, y_max]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    return inter_x_max > inter_x_min and inter_y_max > inter_y_min

## Main Application Logic
def run_rtsp_detection():
    print("Memuat model YOLO (all classes)...")
    model = YOLO(yolo_model)
    print("Model YOLO berhasil dimuat.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Menggunakan device: **{device}**")
    if device == 'cuda':
        print(f"CUDA tersedia. Nama GPU: **{torch.cuda.get_device_name(0)}**")

    print("\nNama kelas yang dikenal model:")
    print(model.names)

    try:
        PERSON_CLASS_ID = list(model.names.keys())[list(model.names.values()).index('person')]
        print(f"Class ID untuk 'person' adalah: {PERSON_CLASS_ID}")
    except ValueError:
        print("Error: 'person' class not found in the loaded YOLO model's names.")
        print("Pastikan model yang digunakan mendukung deteksi 'person'.")
        return 

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka stream RTSP dari {rtsp_url}")
        print("Pastikan URL RTSP benar, kamera dapat diakses, dan jaringan stabil.")
        print("Mencoba ulang setelah 5 detik...")
        time.sleep(5)
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("Gagal membuka stream setelah percobaan ulang. Keluar.")
            return

    print("Stream RTSP berhasil dibuka.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolusi Stream: {frame_width}x{frame_height}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame, mencoba menghubungkan ulang stream...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    print("Gagal menghubungkan ulang stream. Menghentikan aplikasi.")
                    break 
                continue

            display_frame = frame.copy() 

            results = model(
                frame, 
                stream=True, 
                device=device, 
                verbose=False, 
                conf=detectconf_threshold,
            )

            for r in results:
                boxes_tensor = r.boxes.xyxy 
                classes_tensor = r.boxes.cls
                confidences_tensor = r.boxes.conf

                boxes = boxes_tensor.cpu().numpy()
                classes = classes_tensor.cpu().numpy()
                confidences = confidences_tensor.cpu().numpy()

                for i, box in enumerate(boxes):
                    cls_id = int(classes[i])
                    conf = confidences[i]
                    x1, y1, x2, y2 = map(int, box)

                    is_inside_param_box = intersect([x1, y1, x2, y2], [param_box_x_min, param_box_y_min, param_box_x_max, param_box_y_max])
                    
                    if is_inside_param_box:
                        if cls_id == PERSON_CLASS_ID:
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{model.names[cls_id]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
                        cv2.putText(display_frame, f"{model.names[cls_id]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.rectangle(display_frame, (param_box_x_min, param_box_y_min), (param_box_x_max, param_box_y_max), (255, 0, 0), 2) # Biru
            cv2.putText(display_frame, "Capture Zone", (param_box_x_min, param_box_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("RTSP Stream - Single YOLO Dual Focus", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Tombol 'q' ditekan. Menghentikan aplikasi.")
                break

    except Exception as e:
        print(f"!!! Error terjadi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Aplikasi dimatikan. Sumber daya dirilis.")

if __name__ == "__main__":
    run_rtsp_detection()