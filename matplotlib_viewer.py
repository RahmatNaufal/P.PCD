from math import ceil
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def show_images(images: Sequence[np.ndarray], titles: Sequence[str]) -> None:
    if not images:
        raise ValueError("Minimal satu gambar diperlukan.")
    if len(images) != len(titles):
        raise ValueError("Jumlah judul harus sama dengan jumlah gambar.")

    columns = min(3, len(images))
    rows = ceil(len(images) / columns)
    figure = plt.figure(
        num=1,
        figsize=(5 * columns, 4 * rows),
        clear=True,
    )
    axes = figure.subplots(rows, columns, squeeze=False)

    for axis, image, title in zip(axes.flat, images, titles):
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        elif image.ndim == 3 and image.shape[2] == 3:
            axis.imshow(image[..., ::-1])
        elif image.ndim == 3 and image.shape[2] == 4:
            axis.imshow(image[..., [2, 1, 0, 3]])
        else:
            raise ValueError(f"Format gambar tidak didukung: {image.shape}")

        axis.set_title(title)
        axis.axis("off")

    for axis in axes.flat[len(images):]:
        axis.set_visible(False)

    figure.tight_layout()
    plt.show()
