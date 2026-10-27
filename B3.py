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

zoom_in = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
zoom_out = cv2.resize(rgb, None, fx=0.5, fy=0.5)
skewed = cv2.resize(rgb, (900, 400), interpolation=cv2.INTER_AREA)

hasil = [
    ("Zoom In", zoom_in),
    ("Zoom Out", zoom_out),
    ("Skewed", skewed)
]

for nama, gambar in hasil:
    print(f"\n{nama}: {gambar.shape[1]} x {gambar.shape[0]} piksel")

plt.figure(figsize=(12, 8))

for i, (judul, gambar) in enumerate([
    ("Citra Asli", rgb),
    ("Zoom In - 2 Kali", zoom_in),
    ("Zoom Out - 0.5 Kali", zoom_out),
    ("Skewed - 900 x 400", skewed)
], 1):
    plt.subplot(2, 2, i)
    plt.imshow(gambar)
    plt.title(judul)
    plt.axis("off")

plt.tight_layout()
plt.show()

for nama, gambar in [
    ("pnl_zoom_in.jpg", zoom_in),
    ("pnl_zoom_out.jpg", zoom_out),
    ("pnl_skewed.jpg", skewed)
]:
    cv2.imwrite(nama, cv2.cvtColor(gambar, cv2.COLOR_RGB2BGR))

print("\nSemua hasil resize berhasil disimpan.")