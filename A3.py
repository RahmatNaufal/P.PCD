import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

nama_file = "Citra5.jpg"

citra = cv2.imread(nama_file, cv2.IMREAD_COLOR)

if citra is None:
    raise FileNotFoundError(
        f"Citra '{nama_file}' tidak ditemukan. "
        "Pastikan file berada di folder program."
    )
citra_rgb = cv2.cvtColor(citra, cv2.COLOR_BGR2RGB)
tinggi, lebar, kanal = citra.shape

print("INFORMASI CITRA DIGITAL")
print("-----------------------")
print(f"Nama file       : {nama_file}")
print(f"Lebar citra     : {lebar} piksel")
print(f"Tinggi citra    : {tinggi} piksel")
print(f"Jumlah kanal    : {kanal}")
print(f"Dimensi array   : {citra.shape}")
print(f"Tipe data       : {citra.dtype}")
print(f"Nilai piksel    : {citra.min()} - {citra.max()}")

diagonal = math.sqrt(lebar**2 + tinggi**2)
print(f"Diagonal citra  : {diagonal:.2f} piksel")

citra_gray = cv2.cvtColor(
    citra, cv2.COLOR_BGR2GRAY
)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(citra_rgb)
plt.title("Citra Asli (RGB)")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(citra_gray, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Grayscale")
plt.axis("off")

plt.tight_layout()
plt.show()
nama_hasil = "pnl_grayscale.jpg"

berhasil = cv2.imwrite(nama_hasil, citra_gray)

if berhasil:
    print(f"\nCitra grayscale berhasil disimpan: {nama_hasil}")
else:
    print("\nCitra grayscale gagal disimpan.")