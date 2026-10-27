import numpy as np

def konvolusi(citra, kernel):
    # """Menerapkan kernel 2D pada citra dan mengembalikan array berukuran sama.

    # Kernel digunakan dalam orientasi aslinya, sehingga operasi ini secara
    # matematis merupakan korelasi silang. Posisi kernel di luar citra dianggap
    # bernilai nol.
    # """
    tinggi, lebar = citra.shape
    tinggi_kernel, lebar_kernel = kernel.shape

    # Titik tengah kernel menjadi pusat untuk setiap piksel citra.
    tengah_y = tinggi_kernel // 2
    tengah_x = lebar_kernel // 2

    # Hasil memakai ukuran citra; float64 menampung hasil perkalian dan jumlah.
    hasil = np.zeros((tinggi, lebar), dtype=np.float64)

    # Hitung nilai hasil untuk setiap posisi piksel.
    for y in range(tinggi):
        for x in range(lebar):
            jumlah = 0

            # Kalikan setiap elemen kernel dengan piksel tetangganya.
            for ky in range(tinggi_kernel):
                for kx in range(lebar_kernel):

                    posisi_y = y + ky - tengah_y
                    posisi_x = x + kx - tengah_x

                    # Lewati posisi di luar citra, setara dengan padding nol.
                    if (0 <= posisi_y < tinggi and
                            0 <= posisi_x < lebar):

                        jumlah += (
                            citra[posisi_y, posisi_x]
                            * kernel[ky, kx]
                        )

            hasil[y, x] = jumlah

    return hasil