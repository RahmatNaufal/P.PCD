import cv2
import numpy as np
import matplotlib.pyplot as plt


def tampilkan_histogram(gambar, judul):
    plt.figure(figsize=(8, 5))

    plt.hist(
        gambar.ravel(),
        bins=256,
        range=(0, 256),
        color="gray"
    )

    plt.title(judul)
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Jumlah Piksel")
    plt.xlim([0, 256])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

gambar = cv2.imread("Citra5.jpg")

if gambar is None:
    print("Gambar Citra2.png tidak ditemukan!")
    exit()

abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

print("\n=== INFORMASI CITRA ===")
print("Ukuran citra:", abu_abu.shape)
print("Tinggi citra:", abu_abu.shape[0], "piksel")
print("Lebar citra:", abu_abu.shape[1], "piksel")
print("Jumlah kanal citra asli:", gambar.shape[2])
print("Tipe data:", abu_abu.dtype)

nilai_min = int(np.min(abu_abu))
nilai_max = int(np.max(abu_abu))

print("\n=== NILAI INTENSITAS CITRA ===")
print("Nilai piksel minimum:", nilai_min)
print("Nilai piksel maksimum:", nilai_max)

if nilai_max > nilai_min:

    hasil = (
        (abu_abu.astype(np.float64) - nilai_min)
        / (nilai_max - nilai_min)
    ) * 255

    hasil = np.clip(hasil, 0, 255).astype(np.uint8)

else:
    hasil = abu_abu.copy()

print("\nPeregangan kontras selesai.")
print("Nilai minimum hasil:", int(np.min(hasil)))
print("Nilai maksimum hasil:", int(np.max(hasil)))

cv2.imwrite("hasil_a6.jpg", hasil)

print("Hasil disimpan sebagai hasil_a6.jpg")

print("\n========== MENU PRAKTIK A6 ==========")
print("1. Tampilkan gambar grayscale asli")
print("2. Tampilkan hasil peregangan kontras")
print("3. Bandingkan gambar asli dan hasil")
print("4. Tampilkan histogram gambar asli")
print("5. Tampilkan histogram hasil peregangan")
print("6. Tampilkan gambar dan histogram")
print("=====================================")

while True:

    pilihan = input("Pilih opsi (1-6): ").strip()

    if pilihan in ["1", "2", "3", "4", "5", "6"]:
        break

    print("Pilihan tidak valid! Masukkan angka 1 sampai 6.")


if pilihan == "1":

    plt.figure(figsize=(7, 5))
    plt.imshow(abu_abu, cmap="gray")
    plt.title("Gambar Grayscale Asli")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


elif pilihan == "2":

    plt.figure(figsize=(7, 5))
    plt.imshow(hasil, cmap="gray")
    plt.title("Hasil Peregangan Kontras")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


elif pilihan == "3":

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(abu_abu, cmap="gray")
    plt.title("Gambar Grayscale Asli")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hasil, cmap="gray")
    plt.title("Hasil Peregangan Kontras")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


elif pilihan == "4":

    tampilkan_histogram(
        abu_abu,
        "Histogram Gambar Asli"
    )


elif pilihan == "5":

    tampilkan_histogram(
        hasil,
        "Histogram Hasil Peregangan Kontras"
    )


elif pilihan == "6":

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(abu_abu, cmap="gray")
    plt.title("Gambar Grayscale Asli")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(hasil, cmap="gray")
    plt.title("Hasil Peregangan Kontras")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.hist(
        abu_abu.ravel(),
        bins=256,
        range=(0, 256),
        color="gray"
    )
    plt.title("Histogram Gambar Asli")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Jumlah Piksel")
    plt.xlim([0, 256])
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.hist(
        hasil.ravel(),
        bins=256,
        range=(0, 256),
        color="gray"
    )
    plt.title("Histogram Hasil Peregangan")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Jumlah Piksel")
    plt.xlim([0, 256])
    plt.grid(True)

    plt.tight_layout()
    plt.show()