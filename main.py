from dataclasses import dataclass
import numpy as np
from PIL import Image
import time
from functools import wraps


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


def shading(surface_color, specular, diffuse):

    R = (surface_color[0]/255 * diffuse + specular) * 255
    G = (surface_color[1]/255 * diffuse + specular) * 255
    B = (surface_color[2]/255 * diffuse + specular) * 255

    return np.array([clip(R), clip(G), clip(B)])


def clip(intensity):
    """ Ensure the final pixel intensities are in the range 0-255."""
    intensity = int(round(intensity))
    return min(max(0, intensity), 255)


@dataclass
class Sphere:
    origin: np.ndarray
    radius: float
    color: np.ndarray

    k_diffuse_reflection: float = 0.7
    k_specular_reflection: float = 0.5


@dataclass
class Ray:
    origin: np.ndarray
    dir: np.ndarray


@dataclass
class Camera:
    field_of_view: float

    # todo: Allow camera coordinate system to be moved and rotated

    origin: np.ndarray = np.array([0, 0, 0])
    cam_i: np.ndarray = np.array([1, 0, 0])
    cam_j: np.ndarray = np.array([0, -1, 0])
    cam_k: np.ndarray = np.array([0, 0, 1])
    R = np.array([cam_i, cam_j, cam_k])

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
        self.objects = []
        self.light = None
        self.camera = Camera(field_of_view=45)
        self.background = np.array([25, 25, 25])

    def add_object(self, obj):
        self.objects.append(obj)

    def set_light(self, light):
        self.light = light

    def trace(self, ray_direction):

        """ For a given ray:
        1) Loop through the objects within the scene and check for intersection.
        2) If there are multiple collisions take the closest one.

        3) Find the position of the intersection point.
        4) Cast a ray from that point to the light source.
            4a) Find if there is clear line of sight
            4b) Find out the angles of the light vector hitting the surface.

            5) Determine the color of the ray as a function of: object color, angles, shadow

          """

        ray_origin = self.camera.origin     # starting point of ray
        t_min = np.inf                      # closest intersection distance
        closest_idx = 0                     # index of the object with the closest intersection point

        for obj_index, obj in enumerate(self.objects):      # 1) Loop through the obejcts

            t = intersect_ray_sphere(ray_origin, ray_direction, obj.origin, obj.radius)

            if t < t_min:                   # If this intersection is closer, set the t_min to t and store the index
                t_min = t
                closest_idx = obj_index

        if t_min == np.inf:                 # If there is no collision  -> the color = background and we return
            color = self.background
            return color

        if self.light is None:              # If there is no light source: flat shading
            color = self.objects[closest_idx].color
            return color

        obj = self.objects[closest_idx]

        # Points of interest:
        P = ray_origin + t_min * ray_direction      # Point of collision.
        L = self.light.origin                       # Point of light source
        R = ray_origin                              # Point of ray source
        S = obj.origin                              # Point of sphere center

        # Vectors of interest:
        PL = L - P                                  # Light direction vector
        N = P - obj.origin                          # Normal surface vector
        RP = P - R                                  # Ray vector

        # Angles:
        theta = angle_between(N, PL)
        alpha = angle_between(RP, PL)

        # Shading constants:
        I_source = self.light.intensity
        kd = obj.k_diffuse_reflection
        ks = obj.k_specular_reflection

        I_diffuse = I_source / t_min * (kd * np.cos(theta))
        I_specular = I_source / t_min * (ks * np.sin(alpha))

        # print(obj.color, I_diffuse, I_specular)

        color = shading(surface_color=obj.color, diffuse=I_diffuse, specular=I_specular)

        return color

    @timed
    def render(self, ray_directions):
        result = np.apply_along_axis(self.trace, 0, ray_directions)
        return result

    def plot(self):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = Axes3D(fig)

        # Plot camera coordinate system:
        for x, y, z, i, j, k in self.camera.enum_coordinate_system():
            ax.quiver(x, y, z, i, j, k, length=0.2)

        # Plot spheres in the scene:
        for obj in self.objects:
            ax.scatter(obj.origin[0], obj.origin[1], obj.origin[2], s=5)

        # # Plot pixel locations:
        for corner in self.camera.generate_corners():
            ax.scatter(corner[0], corner[1], corner[2], s=10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()


def intersect_ray_sphere(ray_origin, ray_dir, sphere_origin, sphere_radius):

    """ Returns the distance to the sphere. Returns INF if does not intersect. """

    ro_so = ray_origin - sphere_origin

    a = np.dot(ray_dir, ray_dir)
    b = 2 * np.dot(ray_dir, ro_so)
    c = np.dot(ro_so, ro_so) - sphere_radius * sphere_radius

    discriminant = b * b - 4 * a * c

    if discriminant >= 0:
        disc_sqrt = np.sqrt(discriminant)
        q = (-b - disc_sqrt) / 2.0 if b < 0 else (-b + disc_sqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)

        if t1 >= 0:
            return t1 if t0 < 0 else t0
    else:
        return np.inf


class Light(object):
    def __init__(self, origin, radius, intensity):
        self.origin = origin
        self.radius = radius
        self.intensity = intensity


def main(plotting=False):
    # Resolution settings:
    w = 550
    h = 550

    # Create scene with camera:
    scene = Scene()

    # Populate the scene:
    sphere1 = Sphere(origin=np.array([8, 0, 0]), radius=1.1, color=np.array([200, 85, 85]),
                     k_diffuse_reflection=0.5,
                     k_specular_reflection=0.8)
    sphere2 = Sphere(origin=np.array([11, 3, 0]), radius=0.8, color=np.array([180, 120, 85]),
                     k_diffuse_reflection=0.7,
                     k_specular_reflection=0.7)
    light = Light(origin=np.array([7, -3, 2]), radius=0.5, intensity=1.2)

    scene.add_object(sphere1)
    scene.add_object(sphere2)
    scene.set_light(light)

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

    # Plot scene setup:
    if plotting:
        scene.plot()

    return 0


if __name__ == '__main__':
    main()
