from PIL import Image
from time import time
import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
from numpy cimport ndarray

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t

cpdef float intersect_ray_sphere(np.ndarray ray_origin, np.ndarray ray_dir, np.ndarray sphere_origin, float sphere_radius):

    cpdef np.ndarray RO_SO = ray_origin - sphere_origin

    cpdef float a = np.dot(ray_dir, ray_dir)
    cpdef float b = 2 * np.dot(ray_dir, RO_SO)
    cpdef float c = np.dot(RO_SO, RO_SO) - sphere_radius * sphere_radius

    cpdef float discriminant = b * b - 4 * a * c

    cpdef float distSqrt, q, t0, t1, result

    if discriminant >= 0:
        distSqrt = np.sqrt(discriminant)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)

        if t1 >= 0:
            result = t1 if t0 < 0 else t0
    else:
        result = np.inf

    return result


class Sphere:
    def __init__(self, origin, radius, color):
        self.origin = origin
        self.radius = radius
        self.color = color

    def get_color(self):
        return self.color


class Camera:
    def __init__(self, field_of_view):
        self.field_of_view = field_of_view

        # todo: Allow camera coordinate system to be moved and rotated

        self.origin = np.array([0, 0, 0])
        self.cam_i = np.array([1, 0, 0])
        self.cam_j = np.array([0, -1, 0])
        self.cam_k= np.array([0, 0, 1])
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

cpdef np.ndarray trace(np.ndarray ray_direction, np.ndarray ray_origin, list objects):
    cpdef float dist
    cpdef int closest_obj_index
    cpdef float dist_ray
    cpdef np.ndarray color, color_get

    dist = np.inf
    closest_obj_index = 0
    for obj_index, obj in enumerate(objects):
        dist_ray = intersect_ray_sphere(ray_origin, ray_direction, obj.origin, obj.radius)

        if dist_ray < dist:
            dist = dist_ray
            closest_obj_index = obj_index

    if dist == np.inf:
        color = np.array([0, 0, 0])
    else:
        color_get = objects[closest_obj_index].get_color()
        color = np.array([color_get[0], color_get[1], color_get[2]])

    return color


class Scene:
    def __init__(self):
        self.objects = []
        self.camera = Camera(field_of_view=45)
        self.background = np.array([25, 25, 25])

    def add_object(self, obj):
        self.objects.append(obj)

    def trace(self, ray_direction):
        ray_origin= self.camera.origin
        dist = np.inf
        closest_obj_index = 0
        for obj_index, obj in enumerate(self.objects):
            dist_ray = intersect_ray_sphere(ray_origin, ray_direction, obj.origin, obj.radius)

            if dist_ray < dist:
                dist = dist_ray
                closest_obj_index = obj_index

        if dist == np.inf:
            color = self.background
        else:
            color = self.objects[closest_obj_index].color

        return color

    def render(self, RD):
        try:
            result = np.apply_along_axis(trace, 0, RD, self.camera.origin, self.objects)
        except ValueError:
            print(self.camera.origin, self.objects)
            print(type(self.camera.origin), self.objects)

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
    start = time()
    result = scene.render(RD)
    end = time()
    print(f'Duration: ', '{:10.10f}'.format(end-start))

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
