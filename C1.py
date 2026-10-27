import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img1 = cv2.imread("Citra1.png", 0)
img2 = cv2.imread("Citra2.png", 0)

if img1 is None or img2 is None:
    raise FileNotFoundError("Citra1.png atau Citra2.png tidak ditemukan.")

if img1.shape != img2.shape:
    raise ValueError("Ukuran kedua citra harus sama.")

tinggi, lebar = img1.shape

for nama, img in [("Citra 1", img1), ("Citra 2", img2)]:
    print(f"\n=== INFORMASI {nama.upper()} ===")
    print("Ukuran       :", lebar, "x", tinggi)
    print("Dimensi      :", img.shape)
    print("Tipe data    :", img.dtype)
    print("Nilai piksel :", img.min(), "-", img.max())

print("Diagonal citra:", round(math.sqrt(lebar**2 + tinggi**2), 2))

tambah = cv2.add(img1, img2)
kurang = cv2.subtract(img1, img2)
kali = cv2.multiply(img1.astype(np.float32), img2.astype(np.float32), scale=1/255).astype(np.uint8)

bagi = np.divide(
    img1.astype(np.float32), img2.astype(np.float32),
    out=np.zeros_like(img1, dtype=np.float32), where=img2 != 0
)
bagi = cv2.normalize(bagi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

hasil = [
    ("Citra 1", img1),
    ("Citra 2", img2),
    ("Penjumlahan", tambah),
    ("Pengurangan", kurang),
    ("Perkalian", kali),
    ("Pembagian", bagi)
]

plt.figure(figsize=(12, 8))

for i, (judul, img) in enumerate(hasil, 1):
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(judul)
    plt.axis("off")

plt.tight_layout()
plt.show()

for nama, img in [
    ("c1_penjumlahan.jpg", tambah),
    ("c1_pengurangan.jpg", kurang),
    ("c1_perkalian.jpg", kali),
    ("c1_pembagian.jpg", bagi)
]:
    cv2.imwrite(nama, img)

print("\nSemua hasil operasi aritmatika berhasil disimpan.")