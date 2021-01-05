from numba import cuda
from .trace import sample
from .common import clip_color
from math import sqrt


@cuda.jit
def render(pixel_loc, result, camera_origin, camera_rotation, spheres, lights, planes, amb, lamb, refl, refl_depth):
    R = camera_rotation
    x, y = cuda.grid(2)

    # Calculate the ray direction
    if x <= pixel_loc.shape[1] and y <= pixel_loc.shape[2]:

        # Ray origin:
        R0_X, R0_Y, R0_Z = camera_origin[0], camera_origin[1], camera_origin[2]

        # Pixel location
        PX = pixel_loc[0, x, y]
        PY = pixel_loc[1, x, y]
        PZ = pixel_loc[2, x, y]

        # Ray direction
        RD_X = R[0, 0] * PX + R[0, 1] * PY + R[0, 2] * PZ
        RD_Y = R[1, 0] * PX + R[1, 1] * PY + R[1, 2] * PZ
        RD_Z = R[2, 0] * PX + R[2, 1] * PY + R[2, 2] * PZ
        norm = sqrt(RD_X * RD_X + RD_Y * RD_Y + RD_Z * RD_Z)
        RD_X, RD_Y, RD_Z = RD_X / norm, RD_Y / norm, RD_Z / norm

        # Sample:
        R, G, B = sample((R0_X, R0_Y, R0_Z), (RD_X, RD_Y, RD_Z), spheres, lights, planes, amb, lamb, refl, refl_depth)

        # Save the final color the array:
        result[0, x, y] = clip_color(R)
        result[1, x, y] = clip_color(G)
        result[2, x, y] = clip_color(B)
