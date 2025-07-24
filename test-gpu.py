# This script tests the GPU capabilities of the system by performing matrix multiplication using PyTorch.

import torch
import time

if torch.cuda.is_available():
    print(f"GPU terdeteksi: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("Tidak ada GPU yang terdeteksi. Menggunakan CPU.")

    device = torch.device("cpu")

def test_gpu():
    matrix_size = 5000
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    start_time = time.time()
    result = torch.matmul(a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    print(f"Waktu eksekusi perkalian matriks ({matrix_size}x{matrix_size}): {end_time - start_time:.4f} detik")

if __name__ == "__main__":
    test_gpu()