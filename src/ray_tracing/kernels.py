from numba import cuda
from .trace import sample
from .common import matmul, normalize, linear_comb, to_tuple3, clip_color_vector


@cuda.jit
def render(pixel_loc, result, camera_origin, camera_rotation, spheres, lights, planes, amb, lamb, refl, refl_depth):
    R = camera_rotation
    (R0, R1, R2) = R[0, :], R[1, :], R[2, :]
    x, y = cuda.grid(2)

    # Calculate the ray direction
    if x <= pixel_loc.shape[1] and y <= pixel_loc.shape[2]:

        # Ray origin:
        ray_origin = camera_origin[0], camera_origin[1], camera_origin[2]

        # Pixel location
        P = to_tuple3(pixel_loc[0:3, x, y])

        # Primary Ray direction
        ray_dir = matmul((R0, R1, R2), P)
        ray_dir = normalize(ray_dir)

        # Primary Sample:
        (R, G, B) = sample(ray_origin, ray_dir, spheres, lights, planes, amb, lamb, refl, refl_depth)

        # Take extra samples except on the edges
        if x+1 <= pixel_loc.shape[1] and x-1 >= 0 and y+1 <= pixel_loc.shape[2] and y-1 >= 0:

            # Pixel locations to the left, right top and bottom
            P_left = to_tuple3(pixel_loc[0:3, x-1, y])
            P_right = to_tuple3(pixel_loc[0:3, x+1, y])
            P_top = to_tuple3(pixel_loc[0:3, x, y+1])
            P_bot = to_tuple3(pixel_loc[0:3, x, y-1])

            P_left = linear_comb(P, P_left, 0.5, 0.5)
            P_right = linear_comb(P, P_right, 0.5, 0.5)
            P_top = linear_comb(P, P_top, 0.5, 0.5)
            P_bot = linear_comb(P, P_bot, 0.5, 0.5)

            # Sampled result:
            # (R_s, G_s, B_s) = (0., 0., 0.)

            for P in [P_left, P_right, P_top, P_bot]:
                ray_dir = matmul((R0, R1, R2), P)
                ray_dir = normalize(ray_dir)
                (R_s, G_s, B_s) = sample(ray_origin, ray_dir, spheres, lights, planes, amb, lamb, refl, refl_depth)

                R += R_s
                G += B_s
                B += G_s

            # (R, G, B) = linear_comb((R, G, B), (R_s, G_s, B_s), 0.2, 0.8)

            R = R / 4
            G = G / 4
            B = B / 4

        # Save the final color the array:
        # Clip the color to an 8bit range:
        (R, G, B) = clip_color_vector((R, G, B))

        result[0, x, y] = R
        result[1, x, y] = G
        result[2, x, y] = B
