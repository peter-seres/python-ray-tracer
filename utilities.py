from numpy.linalg import norm
from numpy import arccos, clip, dot, rot90, uint8, zeros, array
from time import time
from functools import wraps
from PIL import Image


def timed(f):
    """ Decorator to print the runtime of function f."""

    @wraps(f)
    def wrap(*args, **kw):
        start = time()
        result = f(*args, **kw)
        end = time()
        print(f'Time test function {f.__name__}')
        print(f'Duration: ', '{:10.4f}'.format(end - start))
        return result

    return wrap


def unit_vector(vector):
    """ Normalizes the input vector.  """

    return vector / norm(vector)


def angle_between(v1, v2):
    """ Returns the angle between two vectors in radians. """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return arccos(clip(dot(v1_u, v2_u), -1.0, 1.0))


def reflect(vector, normal):
    """ Returns the direction of a vector reflected off a surface."""

    vector = unit_vector(vector)
    normal = unit_vector(normal)
    return vector - 2 * (vector * normal) * normal


def clip_rgb(intensity):
    """ Ensure the final pixel intensities are in the range 0-255."""

    intensity = int(round(intensity))
    return min(max(0, intensity), 255)


def pixelize(c):
    return array([clip_rgb(color) for color in c], dtype=uint8)


def save_image(result, name='output.png'):
    pixel_array = zeros(shape=(result.shape[1], result.shape[2], 3), dtype=uint8)
    for RGB in range(3):
        pixel_array[:, :, RGB] = result[RGB, :, :]
    Image.fromarray(rot90(pixel_array)).save(name)



