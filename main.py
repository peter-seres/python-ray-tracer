import numpy as np
from PIL import Image
import time
from functools import wraps
from scene_objects import Sphere


# def plot_3d_point(ax, point):
#     from mpl_toolkits.mplot3d import Axes3D
#     import matplotlib.pyplot as plt
#
#     fig = plt.figure()
#     ax = Axes3D(fig)
#
#     # Plot camera coordinate system:
#     for x, y, z, i, j, k in scene.camera.enum_coordinate_system():
#         ax.quiver(x, y, z, i, j, k, length=0.2)
#
#     # Plot spheres in the scene:
#     for obj in scene.objects:
#         ax.scatter(obj.origin[0], obj.origin[1], obj.origin[2], s=5)
#
#     # # Plot pixel locations:
#     for corner in scene.camera.generate_corners():
#         ax.scatter(corner[0], corner[1], corner[2], s=10)
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     plt.show()


def timed(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        print(f'Time test function {f.__name__}')
        print(f'Duration: ', '{:10.10f}'.format(end-start))
        return result
    return wrap


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def reflect(vector, normal):
    vector = unit_vector(vector)
    normal = unit_vector(normal)
    return vector - 2 * (vector * normal) * normal


def shading(surface_color, specular, diffuse):

    R = (surface_color[0]/255 * diffuse + specular) * 255
    G = (surface_color[1]/255 * diffuse + specular) * 255
    B = (surface_color[2]/255 * diffuse + specular) * 255

    return np.array([clip(R), clip(G), clip(B)])


def clip(intensity):
    """ Ensure the final pixel intensities are in the range 0-255."""
    intensity = int(round(intensity))
    return min(max(0, intensity), 255)


class Camera:
    def __init__(self, field_of_view):

        self.field_of_view = field_of_view
        # todo: Allow camera coordinate system to be moved and rotated
        self.origin: np.ndarray = np.array([0, 0, 0])
        self.cam_i: np.ndarray = np.array([1, 0, 0])
        self.cam_j: np.ndarray = np.array([0, -1, 0])
        self.cam_k: np.ndarray = np.array([0, 0, 1])
        self.R = np.array([self.cam_i, self.cam_j, self.cam_k])

    def enum_coordinate_system(self):
        for vec in [self.cam_i, self.cam_j, self.cam_k]:
            yield self.origin[0], self.origin[1], self.origin[2], vec[0], vec[1], vec[2]

    def generate_corners(self):
        # Generate pixel locations in camera-body coordinate-frame:
        x = 1 / np.tan(np.radians(self.field_of_view)/2)    # Distances of pixels from camera origin
        yy = np.linspace(-1, 1, 2)                          # Distances of pixels along width
        zz = np.linspace(-1, 1, 2)                          # Distances of pixels along height

        for y in yy:
            for z in zz:
                pix_loc_local = [x, y, z]
                yield self.origin + np.matmul(self.R, np.transpose(pix_loc_local))

    def dir(self, loc):
        """ Calculates the ray direction for a given pixel location <loc>"""
        # todo: Right now this assumes that the camera.origin is the same as global coordinate system origin
        direction = np.matmul(self.R, np.transpose(loc))
        return direction

    def create_rays(self, width, height):

        # Pixel locations in local coord system:
        pixel_locations = np.zeros(shape=(3, width, height))                            # Allocate numpy 3D array
        yy, zz = np.mgrid[-1:1:complex(0, width), -1:1:complex(0, height)]              # Create pixel grid
        xx = np.ones(shape=yy.shape) * (1 / np.tan(np.radians(self.field_of_view)/2))   # Distance of grid from origin
        pixel_locations[0, :, :] = xx
        pixel_locations[1, :, :] = yy
        pixel_locations[2, :, :] = zz

        # Calculate ray direction vectors:
        ray_directions = np.apply_along_axis(self.dir, 0, pixel_locations)

        return ray_directions


class Scene:
    def __init__(self):
        self.ambient = 0.5
        self.objects = []
        self.light = None
        self.camera = Camera(field_of_view=45)
        self.background = np.array([25, 25, 25])

    def get_intersection(self, ray_origin, ray_direction):

        """ For a given ray -> it returns the object the ray hits and the distance to that point """

        intersection = None
        for obj in self.objects:
            dist = obj.intersect_ray(ray_origin, ray_direction)

            # If we get an intersection and this intersection is the closest one:
            if dist is not None and (intersection is None or dist < intersection[1]):
                intersection = obj, dist

        return intersection

    def trace(self, ray_direction, ray_origin=None, bounce=0, max_bounce=1, ax=None):

        """ For a given ray:

        0) If this trace is called from a reflection: limit the amount of recursion that can occur:
        1) Loop through the objects within the scene and check for intersection.
        2) If there are multiple collisions take the closest one.

        3) Find the position of the intersection point.
        4) Cast a ray from that point to the light source.
            4a) Find if there is clear line of sight
            4b) Find out the angles of the light vector hitting the surface.

            5) Determine the color of the ray as a function of: object color, angles, shadow

          """

        color = np.array([0.0, 0.0, 0.0])             # Start with darkness

        if bounce > max_bounce:                 # If this is a reflected ray, make sure we don't reflect too many times
            return pixelize(color)

        if ray_origin is None:                  # If ray origin is not specified, it is coming from the camera
            ray_origin = self.camera.origin

        # Loop through all scene objects to check for intersections:
        intersect = self.get_intersection(ray_origin, ray_direction)

        if intersect is None:                   # If there is no collision  -> the color = background and we return
            return pixelize(color)

        obj, dist = intersect

        # Lighting and shading
        color += obj.color * self.ambient  # ambient light

        if self.light is None:                      # If there is no light source: dont do shading
            return pixelize(color)

        # # Vectors of interest:
        P = ray_origin + unit_vector(ray_direction) * dist       # Point of collision.
        L = unit_vector(self.light.origin - P)      # From intersection to light direction
        N = obj.normal(P)                           # Sphere normal vector
        RP = P - ray_origin                         # From ray source to intersection Ray vector

        # If there is a line of sight to the light source:
        if self.get_intersection(ray_origin=P, ray_direction=L) is None:
            lambert_intensity = np.dot(L, N)
            # print(lambert_intensity)
            if lambert_intensity > 0:
                color += obj.color * obj.k_lambert * lambert_intensity

        # Reflected light:
        # reflect_ray_direction = reflect(ray_direction, N)
        # reflected_light = self.trace(ray_direction=reflect_ray_direction, ray_origin=P)
        # color += reflected_light * obj.k_specular

        if ax is not None:
            # Plot with a 2% change:
            import random
            ax.scatter(obj.origin[0], obj.origin[1], obj.origin[2], s=50)
            if random.randint(0, 50) == 0:
                # ax.scatter(ray[0], P[1], P[2])
                ax.quiver(P[0], P[1], P[2], N[0], N[1], N[2])
        return pixelize(color)

    @timed
    def render(self, ray_directions, plot=False):

        if plot:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = Axes3D(fig)

            result = np.apply_along_axis(self.trace, 0, ray_directions, ax=ax)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.show()

        else:
            result = np.apply_along_axis(self.trace, 0, ray_directions)

        return result


class Light(object):
    def __init__(self, origin, radius, intensity):
        self.origin = origin
        self.radius = radius
        self.intensity = intensity


def pixelize(c):
    return np.array([clip(color) for color in c], dtype=np.int8)


def main():
    # Resolution settings:
    w = 1000
    h = 1000

    # Create scene with camera:
    scene = Scene()

    # Populate the scene:
    sphere1 = Sphere(origin=np.array([8, 0, 0]), radius=1.0, color=np.array([236, 33, 33]),
                     k_lambert=0.5, k_specular=0.5)
    sphere2 = Sphere(origin=np.array([10, 2, 0]), radius=1.0, color=np.array([51, 153, 255]),
                     k_lambert=0.5, k_specular=0.5)
    scene.light = Light(origin=np.array([8, -2, 0]), radius=1.0, intensity=1.0)

    scene.objects.append(sphere1)
    scene.objects.append(sphere2)

    # Calculate the ray directions
    print('Generating rays...')
    ray_directions = scene.camera.create_rays(w, h)

    # Render the image:
    print('Rendering...')
    result = scene.render(ray_directions)

    # Save the image to a .png file:
    pixel_array = np.zeros(shape=(result.shape[1], result.shape[2], 3), dtype=np.uint8)
    for RGB in range(3):
        pixel_array[:, :, RGB] = result[RGB, :, :]
    Image.fromarray(np.rot90(pixel_array)).save('output.png')

    return 0


if __name__ == '__main__':
    main()
