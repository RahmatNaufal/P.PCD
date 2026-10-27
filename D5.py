import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Citra5.jpg")

if img is None:
    print("Gambar Citra5.jpg tidak ditemukan!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Informasi Citra")
print("Ukuran citra:", gray.shape)
print("Tinggi citra:", gray.shape[0], "piksel")
print("Lebar citra:", gray.shape[1], "piksel")
print("Jumlah kanal citra asli:", img.shape[2])
print("Tipe data:", gray.dtype)

hasil = gray.copy()

tinggi, lebar = gray.shape

for i in range(3, tinggi - 3):
    for j in range(3, lebar - 3):

        neighbors = []

        for k in range(-3, 4):
            for l in range(-3, 4):
                nilai = gray[i + k, j + l]
                neighbors.append(int(nilai))

        neighbors.sort()

        median = neighbors[24]

        hasil[i, j] = median

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hasil, cmap="gray")
plt.title("Hasil Median Filter 7x7")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("d5_median_filter.jpg", hasil)

print("Hasil Median Filter berhasil disimpan sebagai d5_median_filter.jpg")