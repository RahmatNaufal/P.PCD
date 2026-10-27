import cv2
import numpy as np
import matplotlib.pyplot as plt
from konvolusi import konvolusi

img = cv2.imread("Citra5.jpg")

if img is None:
    raise FileNotFoundError("Citra5.jpg tidak ditemukan.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=== INFORMASI CITRA ===")
print("Ukuran       :", gray.shape)
print("Tinggi       :", gray.shape[0], "piksel")
print("Lebar        :", gray.shape[1], "piksel")
print("Jumlah kanal :", img.shape[2])
print("Tipe data    :", gray.dtype)

kernel = np.array([
    [1, 5, 7, 5, 1],
    [5, 20, 33, 20, 5],
    [7, 33, 55, 33, 7],
    [5, 20, 33, 20, 5],
    [1, 5, 7, 5, 1]
], dtype=np.float64) / 345

hasil = np.clip(konvolusi(gray, kernel), 0, 255).astype(np.uint8)

plt.figure(figsize=(10, 5))

for i, (judul, gambar) in enumerate([
    ("Citra Asli", gray),
    ("Hasil Gaussian Filter", hasil)
], 1):
    plt.subplot(1, 2, i)
    plt.imshow(gambar, cmap="gray")
    plt.title(judul)
    plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("d3_gaussian_filter.jpg", hasil)
print("Hasil disimpan sebagai d3_gaussian_filter.jpg")