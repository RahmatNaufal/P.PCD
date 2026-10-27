import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Citra5.jpg")

if img is None:
    print("Gambar Citra5.jpg tidak ditemukan!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=== INFORMASI CITRA ===")
print("Ukuran citra:", gray.shape)
print("Tinggi citra:", gray.shape[0], "piksel")
print("Lebar citra:", gray.shape[1], "piksel")
print("Jumlah kanal citra asli:", img.shape[2])
print("Tipe data:", gray.dtype)

hasil = gray.copy()

tinggi, lebar = gray.shape

for i in range(3, tinggi - 3):
    for j in range(3, lebar - 3):

        nilai_maksimum = 0

        for k in range(-3, 4):
            for l in range(-3, 4):

                nilai = int(gray[i + k, j + l])

                if nilai > nilai_maksimum:
                    nilai_maksimum = nilai

        hasil[i, j] = nilai_maksimum

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hasil, cmap="gray")
plt.title("Hasil Max Filtering 7x7")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("d6_max_filtering.jpg", hasil)

print("Hasil Max Filtering berhasil disimpan sebagai d6_max_filtering.jpg")