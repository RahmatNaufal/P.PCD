import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

def info_citra(nama, img):
    h, w = img.shape[:2]
    kanal = img.shape[2] if len(img.shape) == 3 else 1
    diag = np.sqrt(w**2 + h**2)
    print("INFORMASI CITRA DIGITAL\n-----------------------")
    print(f"Nama file      : {nama}\nLebar citra    : {w} piksel\nTinggi citra   : {h} piksel")
    print(f"Jumlah kanal   : {kanal}\nDimensi array  : {img.shape}\nTipe data      : {img.dtype}")
    print(f"Nilai piksel   : {img.min()} - {img.max()}\nDiagonal citra : {diag:.2f} piksel\n")

nama_file = "cat.jpg"
img = cv2.imread(nama_file)
info_citra(nama_file, img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

init_thres = 100
_, biner_img = cv2.threshold(gray, init_thres, 255, cv2.THRESH_BINARY)

fig = plt.figure("Biner Image (Interactive Slider)", figsize=(12, 5))
plt.subplots_adjust(bottom=0.25)

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original RGB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

ax_biner = plt.subplot(1, 3, 3)
img_plot = plt.imshow(biner_img, cmap="gray", vmin=0, vmax=255)
title_obj = plt.title(f"Biner Image (T={init_thres})")
plt.axis("off")

# Menambahkan Slider di bagian bawah
ax_slider = plt.axes([0.25, 0.08, 0.5, 0.04])
slider_thres = Slider(ax_slider, 'Threshold (T)', 0, 255, valinit=init_thres, valstep=1)

def update(val):
    t = int(slider_thres.val)
    _, new_biner = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    img_plot.set_data(new_biner)
    title_obj.set_text(f"Biner Image (T={t})")
    fig.canvas.draw_idle()

slider_thres.on_changed(update)
plt.show()