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
        print(f'Time test function {f.__name__}: with arguments:[{args}, {kw}]')
        print(f'Duration: ', '{:10.10f}'.format(end-start))
        return result
    return wrap


@dataclass
class Sphere:
    origin: np.ndarray
    radius: float
    color: np.ndarray


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
        PIX_LOC = np.zeros(shape=(3, width, height))                                    # Allocate numpy 3D array
        yy, zz = np.mgrid[-1:1:complex(0, width), -1:1:complex(0, height)]              # Create pixel grid
        xx = np.ones(shape=yy.shape) * (1 / np.tan(np.radians(self.field_of_view)/2))   # Distance of grid from origin
        PIX_LOC[0, :, :] = xx
        PIX_LOC[1, :, :] = yy
        PIX_LOC[2, :, :] = zz

        # Calculate ray direction vectors:
        RD = np.apply_along_axis(self.dir, 0, PIX_LOC)

        return RD


class Scene:
    def __init__(self):
        self.objects = []
        self.camera = Camera(field_of_view=45)
        self.background = np.array([25, 25, 25])

    def add_object(self, obj):
        self.objects.append(obj)

    def trace(self, ray_direction):
        ray_origin = self.camera.origin

        dist = np.inf
        closes_obj_index = 0
        for obj_index, obj in enumerate(self.objects):
            dist_ray = intersect_ray_sphere(ray_origin, ray_direction, obj.origin, obj.radius)

            if dist_ray < dist:
                dist = dist_ray
                closes_obj_index = obj_index

        if dist == np.inf:
            color = self.background
        else:
            color = self.objects[closes_obj_index].color

        return color

    @timed
    def render(self, RD):
        result = np.apply_along_axis(self.trace, 0, RD)
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

    RO_SO = ray_origin - sphere_origin

    a = np.dot(ray_dir, ray_dir)
    b = 2 * np.dot(ray_dir, RO_SO)
    c = np.dot(RO_SO, RO_SO) - sphere_radius * sphere_radius

    discriminant = b * b - 4 * a * c

    if discriminant >= 0:
        distSqrt = np.sqrt(discriminant)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)

        if t1 >= 0:
            return t1 if t0 < 0 else t0
    else:
        return np.inf


def main(plotting=False):
    # Resolution settings:
    w = 1000
    h = 1000

    # Create scene with camera:
    scene = Scene()

    # Add a sphere object:
    sphere = Sphere(origin=np.array([8, 0, 0]), radius=1.1, color=np.array([200, 85, 85]))
    scene.add_object(sphere)

    # Calculate the ray directions
    print('Generating rays...')
    RD = scene.camera.create_rays(w, h)

    # Render the image:
    print('Rendering...')
    result = scene.render(RD)

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
