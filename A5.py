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

nama_file = "Citra5.jpg"
img = cv2.imread(nama_file)
info_citra(nama_file, img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

init_contrast = 1.6
contrast_img = np.clip(np.ceil(gray * init_contrast), 0, 255).astype(np.uint8)

fig = plt.figure("Pengaturan Kontras (Interactive Slider)", figsize=(12, 5))
plt.subplots_adjust(bottom=0.25)

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original RGB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

ax_cont = plt.subplot(1, 3, 3)
img_plot = plt.imshow(contrast_img, cmap="gray", vmin=0, vmax=255)
title_obj = plt.title(f"Contrast (x{init_contrast:.1f})")
plt.axis("off")

# Menambahkan Slider di bagian bawah
ax_slider = plt.axes([0.25, 0.08, 0.5, 0.04])
slider_contrast = Slider(ax_slider, 'Contrast', 0.1, 3.0, valinit=init_contrast, valstep=0.1)

def update(val):
    c = slider_contrast.val
    new_img = np.clip(np.ceil(gray * c), 0, 255).astype(np.uint8)
    img_plot.set_data(new_img)
    title_obj.set_text(f"Contrast (x{c:.1f})")
    fig.canvas.draw_idle()

slider_contrast.on_changed(update)
plt.show()