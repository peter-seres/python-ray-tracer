from numba import cuda
from .trace import trace
from .common import clip_color
from math import sqrt


@cuda.jit()
def render_kernel(pixel_array, rays, spheres, lights, planes,
                  ambient_int, lambert_int, reflection_int, depth):
    """ This kernel render one pixel by casting a ray from a specific pixel location."""

    # Location of pixel
    x, y = cuda.grid(2)

    # Check if the thread is within the bounds and run the ray trace for this pixel:
    if x <= pixel_array.shape[1] and y <= pixel_array.shape[2]:

        # Get the ray's origin at pixel location
        R0_X = rays[0, x, y]
        R0_Y = rays[1, x, y]
        R0_Z = rays[2, x, y]

        # Get the ray's direction at pixel location
        RD_X = rays[3, x, y]
        RD_Y = rays[4, x, y]
        RD_Z = rays[5, x, y]

        # Run the tracing for this pixel to get R, G, B values
        RGB, POINT, REFLECTION_DIR = trace((R0_X, R0_Y, R0_Z), (RD_X, RD_Y, RD_Z), spheres, lights, planes, ambient_int, lambert_int)

        R, G, B = RGB

        # Run the reflection "depth" amount of times:
        for i in range(depth):
            if POINT[0] == 404. and POINT[1] == 404. and POINT[2] == 404.:
                continue

            RGB_refl, POINT, REFLECTION_DIR = trace(POINT, REFLECTION_DIR, spheres, lights, planes, ambient_int, lambert_int)

            R += RGB_refl[0] * reflection_int**(i+1)
            G += RGB_refl[1] * reflection_int**(i+1)
            B += RGB_refl[2] * reflection_int**(i+1)

        # Save the final color the array:
        pixel_array[0, x, y] = clip_color(R)
        pixel_array[1, x, y] = clip_color(G)
        pixel_array[2, x, y] = clip_color(B)


@cuda.jit
def ray_dir_kernel(pixel_locations, rays, camera_origin, R):

    """ Calculates the unit ray direction vector for the pixel at x, y for a camera origin and camera rotation R.
        The output is then written into a 6 by width by height size 3D array called rays.
    """

    x, y = cuda.grid(2)

    if x <= pixel_locations.shape[1] and y <= pixel_locations.shape[2]:

        PX = pixel_locations[0, x, y]
        PY = pixel_locations[1, x, y]
        PZ = pixel_locations[2, x, y]

        # Ray direction is:
        RD_X = R[0, 0] * PX + R[0, 1] * PY + R[0, 2] * PZ
        RD_Y = R[1, 0] * PX + R[1, 1] * PY + R[1, 2] * PZ
        RD_Z = R[2, 0] * PX + R[2, 1] * PY + R[2, 2] * PZ

        norm = sqrt(RD_X * RD_X + RD_Y * RD_Y + RD_Z * RD_Z)

        # Fill in the rays array:
        rays[0, x, y] = camera_origin[0]
        rays[1, x, y] = camera_origin[1]
        rays[2, x, y] = camera_origin[2]
        rays[3, x, y] = RD_X / norm
        rays[4, x, y] = RD_Y / norm
        rays[5, x, y] = RD_Z / norm
