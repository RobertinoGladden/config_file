import cv2
import torch
from ultralytics import YOLO
import os
import time
import shutil

# --- Konfigurasi Umum ---
dataset = r'/home/jetson/your_project/ppe-dataset/data.yaml' 

yolo_model = 'yolo11m.pt' 

tensorrt_output = 'best.engine' 

def run_fine_tuning_and_export():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: **{device}**")
    if device == 'cuda':
        print(f"GPU: **{torch.cuda.get_device_name(0)}**")
        torch.cuda.empty_cache()
        print("CUDA empty cache")
    else:
        print("CUDA Not Available")
        print("Error Convert to TensorRT")

    # 1. Muat Model Dasar YOLO
    print(f"\nLoading model '{yolo_model}'...")
    try:
        model = YOLO(yolo_model)
        print("Model loaded")
    except Exception as e:
        print(f"Error: Model Not Found '{yolo_model}'.")
        print(e)
        return

    print(f"{dataset}:")
    try:
        with open(dataset, 'r') as file:
            print(file.read())
    except FileNotFoundError:
        print(f"Error: File '{dataset}' Not Found")
        return
    except Exception as e:
        print(f"Error read '{dataset}': {e}")
        return

    # 3. Jalankan Proses Fine-tuning
    print("\n--- Fine Tuning Process ---")
    try:
        results = model.train(
            data=dataset,  
            epochs=50,         
            imgsz=640,           
            batch=8,      
            device=device,          
            patience=10,   
            lr0=0.01,              
            augment=True,        
        )
        print("\nFine Tuning Ended")
    except Exception as e:
        print(f"Error Fine Tuning: {e}")
        import traceback
        traceback.print_exc()
        return
    
    runs_dir = 'runs/detect'
    if not os.path.exists(runs_dir):
        print(f"Error: Directory '{runs_dir}' Not Found")
        return

    latest_run_dir = None
    for d in sorted(os.listdir(runs_dir), reverse=True):
        full_path = os.path.join(runs_dir, d)
        if os.path.isdir(full_path) and d.startswith('train'):
            latest_run_dir = full_path
            break
            
    if not latest_run_dir:
        print("Error: Can't find latest training directory'.")
        return

    fine_tuned_model_path = os.path.join(latest_run_dir, 'weights', 'best.pt')

    if not os.path.exists(fine_tuned_model_path):
        print(f"Error: Best Model Not Found '{fine_tuned_model_path}'.")
        return

    print(f"\nModel yang telah di-fine-tune ditemukan di: {fine_tuned_model_path}")

    if device == 'cuda':
        print("\n--- TensorRT Process ---")
        try:
            model_to_export = YOLO(fine_tuned_model_path)
            
            model_to_export.export(format='engine', device=0, half=True, verbose=True, 
                                   filename=tensorrt_output) 
            
            tensorrt_output_path = os.path.join(os.path.dirname(fine_tuned_model_path), tensorrt_output)
            print(f"\nProcess Done: {tensorrt_output_path}")

        except Exception as e:
            print(f"Error TensorRT: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nCUDA Not Available. TensorRT export skipped.")

    print("\nProcess Ended.")

if __name__ == "__main__":
    run_fine_tuning_and_export()