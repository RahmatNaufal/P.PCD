import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

nama_file = "Citra5.jpg"
citra = cv2.imread(nama_file, 0)

if citra is None:
    raise FileNotFoundError(f"Citra {nama_file} tidak ditemukan.")

tinggi, lebar = citra.shape

print("=== INFORMASI CITRA ===")
print("Nama file    :", nama_file)
print("Lebar        :", lebar)
print("Tinggi       :", tinggi)
print("Dimensi      :", citra.shape)
print("Tipe data    :", citra.dtype)
print("Nilai piksel :", citra.min(), "-", citra.max())
print("Diagonal     :", round(math.sqrt(lebar**2 + tinggi**2), 2))

def konvolusi(citra, kernel):
    batas = kernel.shape[0] // 2
    hasil = np.zeros(citra.shape, dtype=np.float32)
    citra = citra.astype(np.float32)

    for i in range(batas, citra.shape[0] - batas):
        for j in range(batas, citra.shape[1] - batas):
            hasil[i, j] = np.sum(
                citra[i-batas:i+batas+1, j-batas:j+batas+1] * kernel
            )
    return hasil

kernel_a = np.ones((3, 3), dtype=np.float32)
kernel_b = np.array([[6, 0, -6], [6, 0, -6], [6, 0, -6]], dtype=np.float32)

hasil_a = konvolusi(citra, kernel_a)
hasil_b = konvolusi(citra, kernel_b)

hasil_a = cv2.normalize(hasil_a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
hasil_b = cv2.normalize(np.abs(hasil_b), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

print("\n=== INFORMASI HASIL KONVOLUSI ===")
for nama, hasil in [("Kernel A", hasil_a), ("Kernel B", hasil_b)]:
    print(f"\n{nama}:")
    print("Nilai minimum :", hasil.min())
    print("Nilai maksimum:", hasil.max())

plt.figure(figsize=(12, 5))

for i, (judul, gambar) in enumerate([
    ("Citra Asli", citra),
    ("Konvolusi Kernel A", hasil_a),
    ("Konvolusi Kernel B", hasil_b)
], 1):
    plt.subplot(1, 3, i)
    plt.imshow(gambar, cmap="gray")
    plt.title(judul)
    plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("d1_konvolusi_kernel_a.jpg", hasil_a)
cv2.imwrite("d1_konvolusi_kernel_b.jpg", hasil_b)

print("\nHasil konvolusi berhasil disimpan.")