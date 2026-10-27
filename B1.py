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

tx, ty = lebar / 4, tinggi / 4
matriks = np.float32([[1, 0, tx], [0, 1, ty]])

print("\n=== INFORMASI TRANSLASI ===")
print("Translasi X:", tx, "piksel")
print("Translasi Y:", ty, "piksel")
print("Matriks translasi:\n", matriks)

hasil = cv2.warpAffine(rgb, matriks, (lebar, tinggi))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hasil)
plt.title("Citra Setelah Translasi")
plt.axis("off")

plt.tight_layout()
plt.savefig("hasil_translasi.png", dpi=300)
plt.show()

cv2.imwrite("pnl_translasi.jpg", cv2.cvtColor(hasil, cv2.COLOR_RGB2BGR))
print("\nHasil disimpan sebagai pnl_translasi.jpg")