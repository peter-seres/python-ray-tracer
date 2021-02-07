from typing import Tuple
from .common import Vector3D
import numpy as np
from .rotation import euler_rotation


class Camera:
    def __init__(self, resolution: Tuple[int, int], position: Vector3D, euler: Vector3D, fov: float = 45.):
        self.resolution = resolution
        self._position = position
        self.rotation = euler_rotation(euler[0], euler[1], euler[2])
        self.field_of_view = fov

    @property
    def position(self):
        return np.array(self._position)

    def generate_pixel_locations(self):
        """ Generate the pixel locations on the camera plane. """

        width, height = self.resolution
        AR = int(width / height)
        yy, zz = np.mgrid[AR:-AR:complex(0, width), 1:-1:complex(0, height)]
        xx = np.ones(shape=yy.shape) * (1 / np.tan(np.radians(self.field_of_view) / 2))  # Distance of grid from origin

        return np.array([xx, yy, zz])
