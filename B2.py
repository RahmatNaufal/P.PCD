import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

nama_file = "Citra5.jpg"
citra = cv2.imread(nama_file)

if citra is None:
    raise FileNotFoundError(f"Citra {nama_file} tidak ditemukan.")

rgb = cv2.cvtColor(citra, cv2.COLOR_BGR2RGB)
tinggi, lebar, kanal = citra.shape

print("=== INFORMASI CITRA ===")
print("Nama file    :", nama_file)
print("Lebar        :", lebar)
print("Tinggi       :", tinggi)
print("Jumlah kanal :", kanal)
print("Dimensi      :", citra.shape)
print("Tipe data    :", citra.dtype)
print("Nilai piksel :", citra.min(), "-", citra.max())
print("Diagonal     :", round(math.sqrt(lebar**2 + tinggi**2), 2))

def rotasi(gambar, sudut):
    tinggi, lebar = gambar.shape[:2]
    matriks = cv2.getRotationMatrix2D((lebar / 2, tinggi / 2), sudut, 1)
    cos, sin = abs(matriks[0, 0]), abs(matriks[0, 1])
    lebar_baru = int(tinggi * sin + lebar * cos)
    tinggi_baru = int(tinggi * cos + lebar * sin)
    matriks[0, 2] += lebar_baru / 2 - lebar / 2
    matriks[1, 2] += tinggi_baru / 2 - tinggi / 2
    return cv2.warpAffine(gambar, matriks, (lebar_baru, tinggi_baru))

sudut_rotasi = [-45, 45, -90, 90, 180]
hasil_rotasi = {}

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(rgb)
plt.title("Citra Asli")
plt.axis("off")

for i, sudut in enumerate(sudut_rotasi, start=2):
    hasil = rotasi(rgb, sudut)
    hasil_rotasi[sudut] = hasil
    print(f"Rotasi {sudut} derajat: {hasil.shape[1]} x {hasil.shape[0]} piksel")

    plt.subplot(2, 3, i)
    plt.imshow(hasil)
    plt.title(f"Rotasi {sudut} Derajat")
    plt.axis("off")

    cv2.imwrite(f"pnl_rotasi_{sudut}.jpg", cv2.cvtColor(hasil, cv2.COLOR_RGB2BGR))

plt.tight_layout()
plt.show()

transpose = cv2.transpose(rgb)
print("\nUkuran sebelum transpose :", lebar, "x", tinggi)
print("Ukuran setelah transpose:", transpose.shape[1], "x", transpose.shape[0])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(transpose)
plt.title("Citra Transpose")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("pnl_transpose.jpg", cv2.cvtColor(transpose, cv2.COLOR_RGB2BGR))
print("\nSemua hasil rotasi dan transpose berhasil disimpan.")