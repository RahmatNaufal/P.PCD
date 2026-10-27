import cv2
import numpy as np
from matplotlib import pyplot as plt

def info_citra(nama, img):
    h, w = img.shape[:2]
    kanal = img.shape[2] if len(img.shape) == 3 else 1
    diag = np.sqrt(w**2 + h**2)
    print("INFORMASI CITRA DIGITAL\n-----------------------")
    print(f"Nama file      : {nama}\nLebar citra    : {w} piksel\nTinggi citra   : {h} piksel")
    print(f"Jumlah kanal   : {kanal}\nDimensi array  : {img.shape}\nTipe data      : {img.dtype}")
    print(f"Nilai piksel   : {img.min()} - {img.max()}\nDiagonal citra : {diag:.2f} piksel\n")

nama_file = "Citra5.jpg"
img = cv2.imread(nama_file)
info_citra(nama_file, img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_out = cv2.normalize(cv2.magnitude(sobel_x, sobel_y), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Prewitt
Px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
Py = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
prewitt_x = cv2.filter2D(gray, cv2.CV_64F, Px)
prewitt_y = cv2.filter2D(gray, cv2.CV_64F, Py)
prewitt_out = cv2.normalize(cv2.magnitude(prewitt_x, prewitt_y), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

plt.figure("Deteksi Tepi Sobel & Prewitt", figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(sobel_out, cmap="gray")
plt.title("Sobel")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(prewitt_out, cmap="gray")
plt.title("Prewitt")
plt.axis("off")
plt.tight_layout()
plt.show()