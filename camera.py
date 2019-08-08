import numpy as np


class Camera:
    def __init__(self, field_of_view):

        self.field_of_view = field_of_view
        # todo: Allow camera coordinate system to be moved and rotated
        self.origin: np.ndarray = np.array([0, 0, 0])
        self.cam_i: np.ndarray = np.array([1, 0, 0])
        self.cam_j: np.ndarray = np.array([0, -1, 0])
        self.cam_k: np.ndarray = np.array([0, 0, 1])
        self.R = np.array([self.cam_i, self.cam_j, self.cam_k])

    def create_pixel_locations(self, width, height):
        # Pixel locations in local coord system:
        pixel_locations = np.zeros(shape=(3, width, height))                            # Allocate numpy 3D array
        yy, zz = np.mgrid[-1:1:complex(0, width), -1:1:complex(0, height)]              # Create pixel grid
        xx = np.ones(shape=yy.shape) * (1 / np.tan(np.radians(self.field_of_view)/2))   # Distance of grid from origin
        pixel_locations[0, :, :] = xx
        pixel_locations[1, :, :] = yy
        pixel_locations[2, :, :] = zz

        return pixel_locations

