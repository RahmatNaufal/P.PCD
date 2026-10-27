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

print("=== INFORMASI CITRA ASLI ===")
print("Nama file    :", nama_file)
print("Lebar        :", lebar)
print("Tinggi       :", tinggi)
print("Jumlah kanal :", kanal)
print("Dimensi      :", citra.shape)
print("Tipe data    :", citra.dtype)
print("Nilai piksel :", citra.min(), "-", citra.max())
print("Diagonal     :", round(math.sqrt(lebar**2 + tinggi**2), 2))

y1, x1 = int(tinggi * 0.1), int(lebar * 0.1)
y2, x2 = int(tinggi * 0.5), int(lebar * 0.5)

crop = rgb[y1:y2, x1:x2]

print("\n=== INFORMASI CROP ===")
print("Koordinat awal :", y1, x1)
print("Koordinat akhir:", y2, x2)
print("Lebar          :", crop.shape[1])
print("Tinggi         :", crop.shape[0])
print("Dimensi        :", crop.shape)

plt.figure(figsize=(10, 5))

for i, (gambar, judul) in enumerate([
    (rgb, "Citra Asli"),
    (crop, "Hasil Crop")
], 1):
    plt.subplot(1, 2, i)
    plt.imshow(gambar)
    plt.title(judul)
    plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("pnl_crop.jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
print("\nHasil crop disimpan sebagai pnl_crop.jpg")