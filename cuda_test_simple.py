import time
from PIL import Image
from numba import cuda
import numpy as np
import math
from main import Camera


@cuda.jit(device=True)
def intersect_ray_sphere(ray_origin, ray_direction, sphere_origin, sphere_radius):
    L0 = sphere_origin[0] - ray_origin[0]
    L1 = sphere_origin[1] - ray_origin[1]
    L2 = sphere_origin[2] - ray_origin[2]

    R0 = ray_direction[0]
    R1 = ray_direction[1]
    R2 = ray_direction[2]

    t_ca = L0 * R0 + L1 * R1 + L2 * R2

    if t_ca < 0:
        return -999.0

    d = math.sqrt(L0 * L0 + L1 * L1 + L2 * L2 - t_ca**2)

    if d < 0 or d > sphere_radius:
        return -999.0

    t_hc = math.sqrt(sphere_radius ** 2 - d ** 2)

    t_0 = t_ca - t_hc
    t_1 = t_ca + t_hc

    if t_0 < 0:
        t_0 = t_1
        if t_0 < 0:
            return -999.0

    return t_0


@cuda.jit(device=True)
def clip(color):
    return min(max(0, int(round(color))), 255)


@cuda.jit(device=True)
def get_intersection(ray_origin, ray_dir, spheres):

    intersect_dist = 999.0
    sphere_index = -999
    for idx in range(spheres.shape[1]):
        dist = intersect_ray_sphere(ray_origin, ray_dir, spheres[0:3, idx], spheres[3, idx])

        # If it hits the sphere: get the color
        if intersect_dist > dist > 0:
            intersect_dist = dist
            sphere_index = idx

    return intersect_dist, sphere_index


@cuda.jit(device=True)
def get_intersection_reflected(ray_origin, ray_dir, spheres, self_index):

    intersect_dist = 999.0
    sphere_index = -999
    for idx in range(spheres.shape[1]):
        if idx == self_index:
            continue

        dist = intersect_ray_sphere(ray_origin, ray_dir, spheres[0:3, idx], spheres[3, idx])

        # If it hits the sphere: get the color
        if intersect_dist > dist > 0:
            intersect_dist = dist
            sphere_index = idx

    return intersect_dist, sphere_index


@cuda.jit(func_or_sig=None, device=True)
def trace_reflection(ray_origin, ray_dir, spheres, lights, sphere_index):
    """ Trace the ray and return the R, G, B values"""

    # Ambient light:
    ambient_intensity = 0.3
    lambert_shading = 1.0

    # Start with black
    R = 0.0
    G = 0.0
    B = 0.0

    # Declare the color of the object
    R_obj = 0.0
    G_obj = 0.0
    B_obj = 0.0

    # Check whether the ray hits any of the spheres:
    intersect_dist, sphere_index = get_intersection_reflected(ray_origin, ray_dir, spheres, sphere_index)

    # If not: return black
    if intersect_dist == 999.0 or sphere_index == -999:
        return clip(R), clip(B), clip(G)

    # The color of the sphere:
    R_obj = spheres[4, sphere_index]
    G_obj = spheres[5, sphere_index]
    B_obj = spheres[6, sphere_index]

    # Add ambient light
    R = R + R_obj * ambient_intensity
    G = G + G_obj * ambient_intensity
    B = B + B_obj * ambient_intensity

    # Get point of intersection P
    P0 = ray_origin[0] + ray_dir[0] * intersect_dist
    P1 = ray_origin[1] + ray_dir[1] * intersect_dist
    P2 = ray_origin[2] + ray_dir[2] * intersect_dist

    # Get unit vector point from intersection point P to the light L
    L0 = lights[0, 0] - P0
    L1 = lights[1, 0] - P1
    L2 = lights[2, 0] - P2
    norm = math.sqrt(L0 * L0 + L1 * L1 + L2 * L2)
    L0 = L0 / norm
    L1 = L1 / norm
    L2 = L2 / norm

    # Get the unit normal vector N at the intersection point P
    N0 = P0 - spheres[0, sphere_index]
    N1 = P1 - spheres[1, sphere_index]
    N2 = P2 - spheres[2, sphere_index]
    norm = math.sqrt(N0 * N0 + N1 * N1 + N2 * N2)
    N0 = N0 / norm
    N1 = N1 / norm
    N2 = N2 / norm

    # If there is a line of sight to the light source, do the lambert shading:
    light_intersect_dist, light_idx = get_intersection((P0, P1, P2), (L0, L1, L2), spheres)
    if light_idx == -999:
        lambert_intensity = L0 * N0 + L1 * N1 + L2 * N2
        if lambert_intensity > 0:
            R = R + R_obj * lambert_intensity * lambert_shading
            G = G + G_obj * lambert_intensity * lambert_shading
            B = B + B_obj * lambert_intensity * lambert_shading

    return clip(R), clip(B), clip(G)


@cuda.jit(func_or_sig=None, device=True)
def trace(ray_origin, ray_dir, spheres, lights, bounce):
    """ Trace the ray and return the R, G, B values"""

    # Ambient light:
    ambient_intensity = 0.3
    lambert_shading = 1.0
    max_bounce = 2
    specular_shading = 0.5

    # Start with black
    R = 0.0
    G = 0.0
    B = 0.0

    if bounce > max_bounce:
        return clip(R), clip(B), clip(G)

    # Declare the color of the object
    R_obj = 0.0
    G_obj = 0.0
    B_obj = 0.0

    # Check whether the ray hits any of the spheres:
    intersect_dist, sphere_index = get_intersection(ray_origin, ray_dir, spheres)

    # If not: return black
    if intersect_dist == 999.0 or sphere_index == -999:
        return clip(R), clip(B), clip(G)

    # The color of the sphere:
    R_obj = spheres[4, sphere_index]
    G_obj = spheres[5, sphere_index]
    B_obj = spheres[6, sphere_index]

    # Add ambient light
    R = R + R_obj * ambient_intensity
    G = G + G_obj * ambient_intensity
    B = B + B_obj * ambient_intensity

    # Get point of intersection P
    P0 = ray_origin[0] + ray_dir[0] * intersect_dist
    P1 = ray_origin[1] + ray_dir[1] * intersect_dist
    P2 = ray_origin[2] + ray_dir[2] * intersect_dist

    # Get unit vector point from intersection point P to the light L
    L0 = lights[0, 0] - P0
    L1 = lights[1, 0] - P1
    L2 = lights[2, 0] - P2
    norm = math.sqrt(L0 * L0 + L1 * L1 + L2 * L2)
    L0 = L0 / norm
    L1 = L1 / norm
    L2 = L2 / norm

    # Get the unit normal vector N at the intersection point P
    N0 = P0 - spheres[0, sphere_index]
    N1 = P1 - spheres[1, sphere_index]
    N2 = P2 - spheres[2, sphere_index]
    norm = math.sqrt(N0 * N0 + N1 * N1 + N2 * N2)
    N0 = N0 / norm
    N1 = N1 / norm
    N2 = N2 / norm

    # If there is a line of sight to the light source, do the lambert shading:
    light_intersect_dist, light_idx = get_intersection((P0, P1, P2), (L0, L1, L2), spheres)
    if light_idx == -999:
        lambert_intensity = L0 * N0 + L1 * N1 + L2 * N2
        if lambert_intensity > 0:
            R = R + R_obj * lambert_intensity * lambert_shading
            G = G + G_obj * lambert_intensity * lambert_shading
            B = B + B_obj * lambert_intensity * lambert_shading

    # Reflect direction:
    R0 = ray_dir[0] - 2 * (ray_dir[0] * N0) * N0
    R1 = ray_dir[1] - 2 * (ray_dir[1] * N1) * N1
    R2 = ray_dir[2] - 2 * (ray_dir[2] * N2) * N2
    norm = math.sqrt(R0 * R0 + R1 * R1 + R2 * R2)
    R0 = R0 / norm
    R1 = R1 / norm
    R2 = R2 / norm

    bounce = bounce + 1

    # Calculate reflected light by calling trace again recursively
    R_refl, G_refl, B_refl = trace_reflection((P0, P1, P2),
                                              (R0, R1, R2),
                                              spheres,
                                              lights,
                                              sphere_index)

    R = R + R_refl * specular_shading
    G = G + G_refl * specular_shading
    B = B + B_refl * specular_shading

    return clip(R), clip(B), clip(G)


@cuda.jit()
def render_kernel(pixel_array, rays, spheres, lights):

    # Location of pixel
    x, y = cuda.grid(2)

    # Check if the thread is within the bounds and run the ray trace for this pixel:
    if x <= pixel_array.shape[1] and y <= pixel_array.shape[2]:

        # Get the ray's origin
        R0_X = rays[0, x, y]
        R0_Y = rays[1, x, y]
        R0_Z = rays[2, x, y]

        # Get the ray's direction
        RD_X = rays[3, x, y]
        RD_Y = rays[4, x, y]
        RD_Z = rays[5, x, y]

        # Run the recursive tracing to get R, G, B values
        pixel_array[0, x, y], pixel_array[1, x, y], pixel_array[2, x, y] = trace((R0_X, R0_Y, R0_Z),
                                                                                 (RD_X, RD_Y, RD_Z),
                                                                                 spheres,
                                                                                 lights,
                                                                                 0)


def main():
    # Resolution settings:
    w = 1000
    h = 1000

    # Generate rays:
    print('Generating Rays...')
    c = Camera(field_of_view=45)
    rays_host = np.zeros((6, w, h), dtype=np.float32)
    rays_host[3:6, :, :] = c.create_rays(w, h)

    # Define sphere positions, radiii and colors
    spheres_host = np.zeros((7, 2), dtype=np.float32)

    spheres_host[0:3, 0]    = np.array([10.0, 0.0, -0.5])
    spheres_host[3, 0]      = 2.5
    spheres_host[4:7, 0]    = np.array([255, 0, 0])

    spheres_host[0:3, 1]    = np.array([5.0, 0.5, 0.0])
    spheres_host[3, 1]      = 1.0
    spheres_host[4:7, 1]    = np.array([0, 0, 255])

    # Define the light positions:
    lights_host = np.zeros((3, 1), dtype=np.float32)
    lights_host[0:3, 0] = np.array([0.0, -1.0, 0.5])

    # Make an empty pixel array:
    A = cuda.to_device(np.zeros((3, w, h), dtype=np.uint8))
    rays = cuda.to_device(rays_host)
    spheres = cuda.to_device(spheres_host)
    lights = cuda.to_device(lights_host)

    # Setup the cuda kernel grid:
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(A.shape[1] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(A.shape[2] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Compile it:
    print('Compiling...')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres, lights)
    end = time.time()
    print(f'Compile time: {1000*(end-start)} ms')

    # Render it:
    print('Rendering...')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres, lights)
    end = time.time()
    print(f'Render time: {1000*(end-start)} ms')

    # Save the image
    name = 'cuda_output.png'
    print(f'Saving image to {name}')
    pixel_array = np.zeros(shape=(A.shape[1], A.shape[2], 3), dtype=np.uint8)
    for RGB in range(3):
        pixel_array[:, :, RGB] = A[RGB, :, :]
    Image.fromarray(np.rot90(pixel_array)).save(name)

    return 0


if __name__ == '__main__':
    main()
