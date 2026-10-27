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

citra_gray = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)

tinggi, lebar = citra_gray.shape
kanal = 1
diagonal = math.sqrt(lebar**2 + tinggi**2)

print("=== INFORMASI CITRA ASLI ===")
print("Nama file       :", nama_file)
print("Lebar citra     :", lebar)
print("Tinggi citra    :", tinggi)
print("Jumlah kanal    :", kanal)
print("Dimensi citra   :", citra_gray.shape)
print("Tipe data       :", citra_gray.dtype)
print("Nilai minimum   :", np.min(citra_gray))
print("Nilai maksimum  :", np.max(citra_gray))
print("Diagonal citra  :", diagonal)

citra_negatif = 255 - citra_gray

print("\n=== INFORMASI CITRA NEGATIF ===")
print("Dimensi citra   :", citra_negatif.shape)
print("Tipe data       :", citra_negatif.dtype)
print("Nilai minimum   :", np.min(citra_negatif))
print("Nilai maksimum  :", np.max(citra_negatif))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(citra_gray, cmap="gray")
plt.title("Citra Asli (Grayscale)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(citra_negatif, cmap="gray")
plt.title("Citra Negatif")
plt.axis("off")

plt.tight_layout()
plt.show()

nama_output = "hasil_a7.jpg"
cv2.imwrite(nama_output, citra_negatif)

print(f"\nHasil citra negatif disimpan sebagai '{nama_output}'")