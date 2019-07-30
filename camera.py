import numpy as np
from sphere import Sphere


def enum_pixel_array(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            yield i, j


def rot_matrix(u, alpha):
    """ Rotate 3D vector _p along axis _u with angle _alpha( deg!!! ) using R.dot(p) """
    """" Might be useful later."""

    alpha = np.radians(alpha)

    u = u / np.linalg.norm(u)
    qr = np.cos(alpha / 2)
    qijk = np.sin(alpha / 2) * u
    qi = qijk[0]
    qj = qijk[1]
    qk = qijk[2]

    R = np.zeros([3, 3])

    R[0, 0] = 1 - 2 * (qj * qj + qk * qk)
    R[0, 1] = 2 * (qi * qj - qk * qr)
    R[0, 2] = 2 * (qi * qk + qj * qr)

    R[1, 0] = 2 * (qi * qj + qk * qr)
    R[1, 1] = 1 - 2 * (qi * qi + qk * qk)
    R[1, 2] = 2 * (qj * qk - qi * qr)

    R[2, 0] = 2 * (qi * qk - qj * qr)
    R[2, 1] = 2 * (qj * qk - qi * qr)
    R[2, 2] = 1 - 2 * (qi * qi + qj * qj)

    return R


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.dir = direction

    def __repr__(self):
        return f'Ray with origin: {self.origin} and direction: {self.dir}'


class Camera:
    def __init__(self, width, height, origin=None, direction=None, field_of_view=45):
        if direction is None:
            direction = [1, 0, 0]
        else:
            raise NotImplementedError
            # todo: Current definition of cam_i, _j, _k coordinate system is incorrect.
        if origin is None:
            origin = [0, 0, 0]

        self.origin = np.array(origin)                              # Camera origin

        cam_i = direction / np.linalg.norm(direction)               # Direction of camera pointing - local X
        cam_j = np.matmul(rot_matrix([0, 0, 1], -90), direction)    # Rotate local X 90 degrees clockwise along Z-axis
        cam_k = np.cross(cam_j, cam_i)                              # Direction of local Z axis.

        self.R = np.array([cam_i, cam_j, cam_k])                    # Transformation matrix from camera to global

        # Generate pixel locations in camera-body coordinate-frame:
        self.x = 1 / np.tan(np.radians(field_of_view)/2)             # Distance of pixels from camera origin
        self.yy = np.linspace(-1, 1, width)                          # Distances of pixels along width
        self.zz = np.linspace(-1, 1, height)                         # Distances ff pixels along height

        self.cam_i = cam_i
        self.cam_j = cam_j
        self.cam_k = cam_k

    def get_local_coord_system(self):
        return self.cam_i, self.cam_j, self.cam_k

    def generate_rays(self):
        # Loop through the pixel locations and generate a rays for pixel location j, k
        for j, y in enumerate(self.yy):
            for k, z in enumerate(self.zz):
                pix_loc_local = [self.x, y, z]
                pix_loc_global = self.origin + np.matmul(self.R, np.transpose(pix_loc_local))

                ray_direction = pix_loc_global - self.origin
                ray_direction = ray_direction / np.linalg.norm(ray_direction)

                ray = Ray(self.origin, ray_direction)

                yield j, k, ray, pix_loc_global

    def cast_rays(self, object_list):
        pixel_array = np.zeros(shape=(len(self.yy), len(self.zz), 3), dtype=np.uint8)

        # Loop through the pixel locations and cast rays:
        for j, k, ray, _ in self.generate_rays():

            dist = np.inf
            closes_obj_index = 0
            for obj_index, obj in enumerate(object_list):
                dist_ray = obj.intersect_ray(ray)

                if dist_ray < dist:
                    dist = dist_ray
                    closes_obj_index = obj_index

            if dist == np.inf:
                color = np.array([0, 0, 0])
            else:
                color = object_list[closes_obj_index].color

            pixel_array[j, k] = color

        return np.rot90(pixel_array)






