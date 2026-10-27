import cv2
import numpy as np
import matplotlib.pyplot as plt

gambar = cv2.imread("Citra5.jpg")
if gambar is None:
    raise FileNotFoundError("Citra5.jpg tidak ditemukan!")

abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
print("Ukuran:", gambar.shape, "| Tipe data:", gambar.dtype)

# Gaussian Filter
kernel = np.array([
    [0, 1, 2, 1, 0],
    [1, 3, 5, 3, 1],
    [2, 5, 9, 5, 2],
    [1, 3, 5, 3, 1],
    [0, 1, 2, 1, 0]
], dtype=np.float32) / 57

gaussian = cv2.filter2D(abu_abu, cv2.CV_32F, kernel)
gaussian = np.clip(gaussian, 0, 255).astype(np.uint8)

# Gradient Sobel
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

gx = cv2.filter2D(gaussian, cv2.CV_32F, sx)
gy = cv2.filter2D(gaussian, cv2.CV_32F, sy)
magnitudo = np.sqrt(gx**2 + gy**2)
gradien = np.uint8(magnitudo / magnitudo.max() * 255) if magnitudo.max() > 0 else np.zeros_like(abu_abu)

# Non-Maximum Suppression
arah = np.degrees(np.arctan2(gy, gx)) % 180
tinggi, lebar = abu_abu.shape
nms = np.zeros_like(abu_abu)

for y in range(1, tinggi - 1):
    for x in range(1, lebar - 1):
        sudut = arah[y, x]
        m = magnitudo[y, x]

        if sudut < 22.5 or sudut >= 157.5:
            a, b = magnitudo[y, x-1], magnitudo[y, x+1]
        elif sudut < 67.5:
            a, b = magnitudo[y-1, x+1], magnitudo[y+1, x-1]
        elif sudut < 112.5:
            a, b = magnitudo[y-1, x], magnitudo[y+1, x]
        else:
            a, b = magnitudo[y-1, x-1], magnitudo[y+1, x+1]

        if m >= a and m >= b:
            nms[y, x] = gradien[y, x]

# Hysteresis Thresholding
bawah, atas = 100, 150
hysteresis = np.zeros_like(abu_abu)
hysteresis[nms >= atas] = 255
hysteresis[(nms >= bawah) & (nms < atas)] = 100

for y in range(1, tinggi - 1):
    for x in range(1, lebar - 1):
        if hysteresis[y, x] == 100:
            tetangga = hysteresis[y-1:y+2, x-1:x+2]
            hysteresis[y, x] = 255 if np.any(tetangga == 255) else 0

# Canny OpenCV
canny = cv2.Canny(abu_abu, bawah, atas)

# Simpan hasil
hasil = {
    "1": (abu_abu, "Grayscale"),
    "2": (gaussian, "Gaussian Filter"),
    "3": (gradien, "Gradient Sobel"),
    "4": (nms, "Non-Maximum Suppression"),
    "5": (hysteresis, "Hysteresis Thresholding"),
    "6": (canny, "Canny OpenCV")
}

for nomor, (data, judul) in hasil.items():
    nama = ["", "grayscale", "gaussian", "gradien", "nms", "hysteresis", "canny_opencv"][int(nomor)]
    cv2.imwrite(f"f2_{nama}.jpg", data)

# Menu tampilan
print("\n=== MENU PRAKTIK F2 ===")
for nomor, (_, judul) in hasil.items():
    print(f"{nomor}. Tampilkan {judul}")
print("7. Bandingkan semua tahapan")

while True:
    pilihan = input("Pilih opsi (1-7): ").strip()
    if pilihan in [str(i) for i in range(1, 8)]:
        break
    print("Pilihan tidak valid!")

if pilihan != "7":
    data, judul = hasil[pilihan]
    plt.imshow(data, cmap="gray", vmin=0, vmax=255)
    plt.title(judul)
    plt.axis("off")
else:
    plt.figure(figsize=(14, 8))
    for i, (data, judul) in enumerate(hasil.values(), 1):
        plt.subplot(2, 3, i)
        plt.imshow(data, cmap="gray", vmin=0, vmax=255)
        plt.title(judul)
        plt.axis("off")

plt.tight_layout()
plt.show()
print("Semua hasil berhasil disimpan.")