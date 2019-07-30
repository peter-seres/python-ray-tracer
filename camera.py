import numpy as np


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


class Camera:
    def __init__(self, width, height, origin=np.array([0, 0, 0]), direction=np.array([1, 0, 0]), field_of_view=45):
        self.origin = origin                                        # Camera origin

        cam_i = direction / np.linalg.norm(direction)               # Direction of camera pointing - local X
        cam_j = np.matmul(rot_matrix([0, 0, 1], -90), direction)    # Rotate local X 90 degrees clockwise along Z-axis
        cam_z = np.cross(cam_j, cam_i)                              # Direction of local Z axis.

        self.R = np.array([cam_i, cam_j, cam_z])                    # Transformation matrix from camera to global

        # Generate pixel locations in camera-body coordinate-frame:
        x = 1 / np.tan(field_of_view/2)                             # Distance of pixels from camera origin
        yy = np.linspace(-1, 1, width)                              # Distances of pixels along width
        zz = np.linspace(-1, 1, height)                             # Distances ff pixels along height

        # Loop through the pixel locations and cast ray directions:
        self.rays = []
        for j, y in enumerate(yy):
            for k, z in enumerate(zz):
                pix_loc_local = [x, y, z]

                pix_loc_global = self.origin + np.matmul(self.R, np.transpose(pix_loc_local))

                ray_direction = pix_loc_global - self.origin
                ray_direction = ray_direction / np.linalg.norm(ray_direction)

                ray = {'origin': self.origin,
                       'direction': ray_direction}

                self.rays.append(ray)
