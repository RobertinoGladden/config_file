import cv2
import torch
from ultralytics import YOLO
import time
import os

# --- Konfigurasi ---
rtsp_url = r"rtsp://36.92.47.218:7430/video1" 
detectconf_threshold = 0.5 

# Path ke model YOLO yang akan digunakan (yolo11l.pt)
# Pastikan path ini benar di sistem Anda (Windows atau Linux/Jetson)
yolo_model = r'C:\Users\lapt1\Downloads\testing all\yolo11l.pt' 
# Untuk Jetson: yolo_model = r'/home/jetson/your_models/yolo11l.pt'

# Bounding box parameter (dihapus dari logika deteksi, hanya tersisa sebagai variabel kosong atau komentar)
# param_box_x_min = 1000
# param_box_y_min = 280
# param_box_x_max = 2000
# param_box_y_max = 1200

# Fungsi intersect tidak lagi diperlukan karena tidak ada pembatasan area
# def intersect(box1, box2):
#     x1_min, y1_min, x1_max, y1_max = box1
#     x2_min, y2_min, x2_max, y2_max = box2
#     inter_x_min = max(x1_min, x2_min)
#     inter_y_min = max(y1_min, y2_min)
#     inter_x_max = min(x1_max, x2_max)
#     inter_y_max = min(y1_max, y2_max)
#     return inter_x_max > inter_x_min and inter_y_max > inter_y_min

## Main Application Logic
def run_rtsp_detection():
    print("Memuat model YOLO...")
    # Model ini (yolo11l.pt) seharusnya sudah bisa mendeteksi 'person'
    model = YOLO(yolo_model)
    print("Model YOLO berhasil dimuat.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Menggunakan device: **{device}**")
    if device == 'cuda':
        print(f"CUDA tersedia. Nama GPU: **{torch.cuda.get_device_name(0)}**")

    print("\nNama kelas yang dikenal model:")
    print(model.names)

    # Dapatkan ID kelas 'person' dari model yang dimuat
    try:
        PERSON_CLASS_ID = list(model.names.keys())[list(model.names.values()).index('person')]
        print(f"Class ID untuk 'person' adalah: {PERSON_CLASS_ID}")
    except ValueError:
        print("Error: 'person' class not found in the loaded YOLO model's names.")
        print("Pastikan model yang digunakan mendukung deteksi 'person'.")
        return # Keluar jika 'person' tidak ditemukan

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

            # Inferensi YOLO untuk mendeteksi semua kelas
            # Kemudian kita akan memfilter hasilnya hanya untuk 'person'
            results = model(
                frame, 
                stream=True, 
                device=device, 
                verbose=False, 
                conf=detectconf_threshold,
                # imgsz=640 # Anda bisa tambahkan ini jika ingin ukuran inferensi spesifik
            )

            for r in results:
                boxes_tensor = r.boxes.xyxy 
                classes_tensor = r.boxes.cls
                confidences_tensor = r.boxes.conf

                # Pindahkan ke CPU dan konversi ke NumPy hanya sekali setelah loop
                # agar lebih efisien jika ada banyak deteksi
                boxes = boxes_tensor.cpu().numpy()
                classes = classes_tensor.cpu().numpy()
                confidences = confidences_tensor.cpu().numpy()

                for i, box in enumerate(boxes):
                    cls_id = int(classes[i])
                    conf = confidences[i]
                    x1, y1, x2, y2 = map(int, box)

                    # Hanya gambar bounding box jika kelasnya adalah 'person'
                    if cls_id == PERSON_CLASS_ID:
                        # Gambar kotak hijau untuk deteksi orang
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                        cv2.putText(display_frame, f"{model.names[cls_id]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Tidak ada lagi bounding box parameter yang digambar
            # cv2.rectangle(display_frame, (param_box_x_min, param_box_y_min), (param_box_x_max, param_box_y_max), (255, 0, 0), 2) 
            # cv2.putText(display_frame, "Capture Zone", (param_box_x_min, param_box_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("RTSP Stream - Person Detector", display_frame)

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