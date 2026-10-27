
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

lebar_baru = 300
tinggi_baru = 300

citra_resize = cv2.resize(
    citra_rgb,
    (lebar_baru, tinggi_baru),
    interpolation=cv2.INTER_AREA
)

pusat = (lebar // 2, tinggi // 2)
sudut = 30

matriks_rotasi = cv2.getRotationMatrix2D(
    pusat, sudut, 1.0
)

citra_rotasi = cv2.warpAffine(
    citra_rgb,
    matriks_rotasi,
    (lebar, tinggi)
)

histogram = cv2.calcHist(
    [citra_gray],
    [0],
    None,
    [256],
    [0, 256]
)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(citra_rgb)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(citra_gray, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Grayscale")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(citra_resize)
plt.title("Citra Resize")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(citra_rotasi)
plt.title("Citra Rotasi 30 Derajat")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.plot(np.arange(256), histogram.ravel())
plt.title("Histogram Grayscale")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Jumlah Piksel")
plt.xlim([0, 255])

plt.tight_layout()
plt.show()