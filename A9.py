import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

nama_file = "Citra5.jpg"
citra = cv2.imread(nama_file)

if citra is None:
    raise FileNotFoundError(f"Citra {nama_file} tidak ditemukan.")

gray = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)
tinggi, lebar = gray.shape

print("=== INFORMASI CITRA GRAYSCALE ===")
print("Nama file      :", nama_file)
print("Lebar          :", lebar)
print("Tinggi         :", tinggi)
print("Jumlah kanal   : 1")
print("Dimensi        :", gray.shape)
print("Tipe data      :", gray.dtype)
print("Nilai piksel   :", gray.min(), "-", gray.max())
print("Diagonal       :", math.sqrt(lebar**2 + tinggi**2))

histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])

print("\n=== INFORMASI HISTOGRAM ===")
print("Jumlah piksel  :", gray.size)
print("Tingkat intensitas:", len(histogram))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Citra Grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.hist(gray.ravel(), 256, [0, 256])
plt.title("Histogram Citra Grayscale")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Frekuensi")
plt.xlim([0, 255])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hasil_a9_histogram.png", dpi=300)
cv2.imwrite("hasil_a9_grayscale.jpg", gray)

print("\nHasil disimpan sebagai:")
print("- hasil_a9_grayscale.jpg")
print("- hasil_a9_histogram.png")

plt.show()