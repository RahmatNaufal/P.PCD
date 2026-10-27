import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("Citra1.png")
img2 = cv2.imread("Citra2.png")

if img1 is None or img2 is None:
    raise FileNotFoundError("Citra1.png atau Citra2.png tidak ditemukan.")

img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
img1, img2 = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in (img1, img2)]

hasil = [
    ("Citra 1", img1),
    ("Citra 2", img2),
    ("Operasi AND", cv2.bitwise_and(img1, img2)),
    ("Operasi OR", cv2.bitwise_or(img1, img2)),
    ("Operasi XOR", cv2.bitwise_xor(img1, img2))
]

print("=== INFORMASI CITRA ===")
for nama, img in hasil[:2]:
    print(f"\n{nama}:")
    print("Ukuran   :", img.shape)
    print("Tipe data:", img.dtype)

y, x = img1.shape[0] // 2, img1.shape[1] // 2
print("\n=== NILAI PIXEL TENGAH ===")
for nama, img in hasil:
    print(f"{nama}: {img[y, x]}")

for nama, img in hasil[2:]:
    cv2.imwrite(
        nama.lower().replace("operasi ", "hasil_C2_").replace(" ", "_") + ".png",
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    )

plt.figure(figsize=(12, 8))
for i, (judul, img) in enumerate(hasil, 1):
    plt.subplot(2, 3, i)
    plt.imshow(img)
    plt.title(judul)
    plt.axis("off")

plt.tight_layout()
plt.show()