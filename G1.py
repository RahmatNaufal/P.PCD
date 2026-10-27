import cv2
import numpy as np
import matplotlib.pyplot as plt

citra = cv2.imread("Citra2.png")
if citra is None:
    raise FileNotFoundError("Citra2.png tidak ditemukan!")

gray = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)
print("Ukuran:", citra.shape, "| Tipe data:", citra.dtype)

_, biner = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

def morfologi(bentuk):
    kernel = cv2.getStructuringElement(bentuk, (5, 5))
    return {
        "Erosi": cv2.erode(biner, kernel),
        "Dilasi": cv2.dilate(biner, kernel),
        "Opening": cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel),
        "Closing": cv2.morphologyEx(biner, cv2.MORPH_CLOSE, kernel)
    }

hasil = morfologi(cv2.MORPH_CROSS)

print("\n=== JUMLAH PIKSEL PUTIH ===")
print("Citra biner:", np.count_nonzero(biner))

for bentuk, nama in [
    (cv2.MORPH_CROSS, "CROSS"),
    (cv2.MORPH_RECT, "RECT"),
    (cv2.MORPH_ELLIPSE, "ELLIPSE")
]:
    print(nama, {
        k: int(np.count_nonzero(v))
        for k, v in morfologi(bentuk).items()
    })

# Skeleton
def skeleton(gambar):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kerangka = np.zeros_like(gambar)
    sementara = gambar.copy()

    while cv2.countNonZero(sementara):
        buka = cv2.morphologyEx(sementara, cv2.MORPH_OPEN, kernel)
        kerangka = cv2.bitwise_or(kerangka, cv2.subtract(sementara, buka))
        sementara = cv2.erode(sementara, kernel)

    return kerangka

kerangka = skeleton(biner)
print("Piksel skeleton:", np.count_nonzero(kerangka))

# Simpan hasil
semua = {
    "biner": biner,
    **{k.lower(): v for k, v in hasil.items()},
    "skeleton": kerangka
}

for nama, gambar in semua.items():
    cv2.imwrite(f"hasil_G1_{nama}.png", gambar)

# Tampilkan hasil
gambar = [gray, *hasil.values(), kerangka]
judul = ["Grayscale", "Erosi", "Dilasi", "Opening", "Closing", "Skeleton"]

plt.figure(figsize=(12, 7))
for i, (img, nama) in enumerate(zip(gambar, judul), 1):
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap="gray")
    plt.title(nama)
    plt.axis("off")

plt.tight_layout()
plt.show()