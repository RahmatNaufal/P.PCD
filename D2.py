import cv2
import numpy as np
import matplotlib.pyplot as plt
from konvolusi import konvolusi

img = cv2.imread("Citra5.jpg")

if img is None:
    raise FileNotFoundError("Citra5.jpg tidak ditemukan.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=== INFORMASI CITRA ===")
print("Ukuran citra :", gray.shape)
print("Tinggi       :", gray.shape[0], "piksel")
print("Lebar        :", gray.shape[1], "piksel")
print("Jumlah kanal :", img.shape[2])
print("Tipe data    :", gray.dtype)

kernel = np.ones((3, 3), dtype=np.float32) / 9
hasil = np.clip(konvolusi(gray, kernel), 0, 255).astype(np.uint8)

plt.figure(figsize=(10, 5))

for i, (judul, gambar) in enumerate([
    ("Citra Asli", gray),
    ("Hasil Mean Filter", hasil)
], 1):
    plt.subplot(1, 2, i)
    plt.imshow(gambar, cmap="gray")
    plt.title(judul)
    plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("d2_mean_filter.jpg", hasil)
print("Hasil disimpan sebagai d2_mean_filter.jpg")