import numpy as np
from PIL import Image, ImageOps


def convert_array_to_image(x: np.ndarray) -> Image:
    """ Takes numpy array x and converts it to RGB image. """

    y = np.zeros(shape=(x.shape[1], x.shape[2], 3))

    # Rearrange the coordinates:
    for i in range(3):
        y[:, :, i] = x[i, :, :]

    im = Image.fromarray(y.astype(np.uint8), mode='RGB')
    im = ImageOps.mirror(im.rotate(270))

    return im
