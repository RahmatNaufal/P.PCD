import cv2
import numpy as np
import matplotlib.pyplot as plt
from konvolusi import konvolusi

img = cv2.imread("Citra5.jpg")
 
if img is None:
    print("Gambar Citra5.jpg tidak ditemukan!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Informasi Citra")
print("Ukuran citra:", gray.shape)
print("Tinggi citra:", gray.shape[0], "piksel")
print("Lebar citra:", gray.shape[1], "piksel")
print("Jumlah kanal citra asli:", img.shape[2])
print("Tipe data:", gray.dtype)

laplace = (1.0 / 16) * np.array([
    [0,  0, -1,  0,  0],
    [0, -1, -2, -1,  0],
    [-1, -2, 16, -2, -1],
    [0, -1, -2, -1,  0],
    [0,  0, -1,  0,  0]
], dtype=np.float64)

hasil = konvolusi(gray, laplace)

hasil = np.clip(hasil, 0, 255).astype(np.uint8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hasil, cmap="gray")
plt.title("Hasil Image Sharpening")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("d4_image_sharpening.jpg", hasil)

print("Hasil penajaman disimpan sebagai d4_image_sharpening.jpg")