import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

nama_file = "Citra5.jpg"
citra = cv2.imread(nama_file)

if citra is None:
    raise FileNotFoundError(f"Citra {nama_file} tidak ditemukan.")

tinggi, lebar, kanal = citra.shape

print("=== INFORMASI CITRA RGB ===")
print("Nama file      :", nama_file)
print("Lebar          :", lebar)
print("Tinggi         :", tinggi)
print("Jumlah kanal   :", kanal)
print("Dimensi        :", citra.shape)
print("Tipe data      :", citra.dtype)
print("Nilai piksel   :", np.min(citra), "-", np.max(citra))
print("Diagonal       :", math.sqrt(lebar**2 + tinggi**2))

warna = ["b", "g", "r"]
nama_warna = ["Biru", "Hijau", "Merah"]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(citra, cv2.COLOR_BGR2RGB))
plt.title("Citra Asli RGB")
plt.axis("off")

plt.subplot(1, 2, 2)

for i in range(3):
    hist = cv2.calcHist([citra], [i], None, [256], [0, 256])
    print(f"Jumlah piksel {nama_warna[i]}:", int(np.sum(hist)))
    plt.plot(hist, color=warna[i], label=nama_warna[i])

plt.title("Histogram Citra RGB")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Frekuensi")
plt.xlim([0, 256])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("hasil_a10_histogram_rgb.png", dpi=300)
print("Grafik disimpan sebagai hasil_a10_histogram_rgb.png")
plt.show()