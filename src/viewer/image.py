import numpy as np
from PIL import Image, ImageOps
from functools import wraps
from time import time


def convert_array_to_image(x: np.ndarray) -> Image:
    """ Takes numpy array x and converts it to RGB image. """

    y = np.zeros(shape=(x.shape[1], x.shape[2], 3))

    # Rearrange the coordinates:
    for i in range(3):
        y[:, :, i] = x[i, :, :]

    im = Image.fromarray(y.astype(np.uint8), mode='RGB')
    im = ImageOps.mirror(im.rotate(270))

    return im


def timed(f):
    """ Timing decorator. """

    @wraps(f)
    def wrap(*args, **kw):
        start_time = time()
        result = f(*args, **kw)
        end_time = time()

        print(f'Function: {f.__name__} timed run took: {1000 * (end_time - start_time):,.1f} ms')
        return result

    return wrap
