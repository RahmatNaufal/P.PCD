import cv2
import numpy as np
import matplotlib.pyplot as plt

gambar = cv2.imread("Citra5.jpg")
if gambar is None:
    raise FileNotFoundError("Citra5.jpg tidak ditemukan!")

gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
print("Ukuran:", gambar.shape, "| Kanal:", gambar.shape[2], "| Tipe data:", gray.dtype)

threshold = 127
maksimum = 255

metode = {
    "Binary": cv2.THRESH_BINARY,
    "Binary Invers": cv2.THRESH_BINARY_INV,
    "Trunc": cv2.THRESH_TRUNC,
    "To Zero": cv2.THRESH_TOZERO,
    "To Zero Invers": cv2.THRESH_TOZERO_INV
}

hasil = {}
for nama, jenis in metode.items():
    _, citra = cv2.threshold(gray, threshold, maksimum, jenis)
    hasil[nama] = citra

    print(f"\n{nama}")
    print("Piksel minimum:", citra.min())
    print("Piksel maksimum:", citra.max())
    print("Matriks 5x5:\n", citra[:5, :5])

    cv2.imwrite(f"h1_{nama.lower().replace(' ', '_')}.jpg", citra)

print("\nSemua hasil berhasil disimpan.")

# Tampilkan perbandingan
daftar = [("Grayscale", gray)] + list(hasil.items())

plt.figure(figsize=(14, 9))
for i, (nama, citra) in enumerate(daftar, 1):
    plt.subplot(2, 3, i)
    plt.imshow(citra, cmap="gray", vmin=0, vmax=255)
    plt.title(nama)
    plt.axis("off")

plt.tight_layout()
plt.show()