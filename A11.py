import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

nama_file = "Citra5.jpg"
citra = cv2.imread(nama_file, cv2.IMREAD_GRAYSCALE)

if citra is None:
    raise FileNotFoundError(f"Citra {nama_file} tidak ditemukan.")

tinggi, lebar = citra.shape

print("=== INFORMASI CITRA ===")
print("Nama file     :", nama_file)
print("Lebar         :", lebar)
print("Tinggi        :", tinggi)
print("Dimensi       :", citra.shape)
print("Tipe data     :", citra.dtype)
print("Nilai piksel  :", citra.min(), "-", citra.max())
print("Diagonal      :", round(math.sqrt(lebar**2 + tinggi**2), 2))

hist = np.bincount(citra.ravel(), minlength=256)
cdf = hist.cumsum()
cdf_min = cdf[cdf > 0].min()
lookup = np.clip((cdf - cdf_min) * 255 / (citra.size - cdf_min), 0, 255).astype(np.uint8)
citra_equal = lookup[citra]

print("\n=== HASIL EQUALIZATION ===")
print("Nilai minimum sebelum :", citra.min())
print("Nilai maksimum sebelum:", citra.max())
print("Nilai minimum sesudah :", citra_equal.min())
print("Nilai maksimum sesudah:", citra_equal.max())

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(citra, cmap="gray")
plt.title("Sebelum Equalization")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(citra_equal, cmap="gray")
plt.title("Setelah Equalization")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.hist(citra.ravel(), 256, [0, 256], color="gray")
plt.title("Histogram Sebelum")
plt.xlabel("Intensitas")
plt.ylabel("Frekuensi")

plt.subplot(2, 2, 4)
plt.hist(citra_equal.ravel(), 256, [0, 256], color="red")
plt.title("Histogram Setelah")
plt.xlabel("Intensitas")
plt.ylabel("Frekuensi")

plt.tight_layout()
cv2.imwrite("hasil_a11_equalization.jpg", citra_equal)
plt.savefig("hasil_a11_histogram.png", dpi=300)

print("\nHasil disimpan:")
print("- hasil_a11_equalization.jpg")
print("- hasil_a11_histogram.png")

plt.show()