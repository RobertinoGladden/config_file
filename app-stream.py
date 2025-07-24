# This script playing a video from an RTSP stream and performs real-time object detection using a YOLO model. It displays the detected objects and their bounding boxes in a window, and allows exiting the stream with the 'q' key. It also includes error handling for model loading and stream opening.
from ultralytics import YOLO
import cv2
import time

# Tentukan path model secara manual (ubah sesuai kebutuhan)
model_path = r'C:\Users\lapt1\Downloads\Hailo AI\ppe-12x.pt'
#model_path = r'C:\Users\lapt1\Downloads\Hailo AI\ppe-model\ppe.pt'

# Tentukan URL RTSP stream (ubah sesuai kebutuhan)
rtsp_url = 'rtsp://36.92.47.218:7430/video2'  # Ganti dengan URL RTSP Anda

# Load model
try:
    model = YOLO(model_path)
    print(f"Model dimuat dari {model_path}")
except Exception as e:
    print(f"Gagal memuat model dari {model_path}: {e}")
    exit()

# Buka stream RTSP dengan konfigurasi tambahan
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print(f"Gagal membuka stream dari {rtsp_url}! Periksa URL atau koneksi.")
    exit()

# Atur resolusi dan frame rate (opsional, sesuaikan dengan stream Anda)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lebar frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Tinggi frame
cap.set(cv2.CAP_PROP_FPS, 15)  # Frame rate (kurangi jika lambat)

print("Stream terbuka, memulai deteksi...")
start_time = time.time()

# Proses stream secara real-time
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame, keluar...")
        break

    # Pastikan frame tidak kosong
    if frame is None or frame.size == 0:
        print("Frame kosong, lewati...")
        continue

    # Lakukan prediksi dengan ukuran input tetap
    try:
        results = model.predict(source=frame, imgsz=640, conf=0.5, save=False, save_txt=False)
    except Exception as e:
        print(f"Gagal memproses frame: {e}")
        break

    # Visualisasi hasil
    for result in results:
        annotated_img = result.plot()  # Gambar dengan bounding box
        cv2.imshow("Deteksi RTSP", annotated_img)

        # Tampilkan jumlah deteksi
        detections = result.boxes.xyxy  # Koordinat bounding box
        print(f"Jumlah deteksi: {len(detections)} pada frame ini")

    # Keluar dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Hitung dan tampilkan FPS
    current_time = time.time()
    fps = 1 / (current_time - start_time)
    start_time = current_time
    cv2.putText(annotated_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Tutup stream dan jendela
cap.release()
cv2.destroyAllWindows()
print("Pengujian selesai!")
