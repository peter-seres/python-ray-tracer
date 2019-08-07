import numpy as np
from scene_objects import Sphere, Plane, Light
from utilities import timed, unit_vector, save_image, pixelize, reflect


class Camera:
    def __init__(self, field_of_view):

        self.field_of_view = field_of_view
        # todo: Allow camera coordinate system to be moved and rotated
        self.origin: np.ndarray = np.array([0, 0, 0])
        self.cam_i: np.ndarray = np.array([1, 0, 0])
        self.cam_j: np.ndarray = np.array([0, -1, 0])
        self.cam_k: np.ndarray = np.array([0, 0, 1])
        self.R = np.array([self.cam_i, self.cam_j, self.cam_k])

    def ray_directions(self, loc):
        """ Calculates the ray direction for a given pixel location <loc>"""

        # todo: Right now this assumes that the camera.origin is the same as global coordinate system origin
        direction = np.matmul(self.R, np.transpose(loc))
        return direction / np.linalg.norm(direction)

    def create_rays(self, width, height):

        # Pixel locations in local coord system:
        pixel_locations = np.zeros(shape=(3, width, height))                            # Allocate numpy 3D array
        yy, zz = np.mgrid[-1:1:complex(0, width), -1:1:complex(0, height)]              # Create pixel grid
        xx = np.ones(shape=yy.shape) * (1 / np.tan(np.radians(self.field_of_view)/2))   # Distance of grid from origin
        pixel_locations[0, :, :] = xx
        pixel_locations[1, :, :] = yy
        pixel_locations[2, :, :] = zz

        # Calculate ray direction vectors:
        ray_directions = np.apply_along_axis(self.ray_directions, 0, pixel_locations)

        return ray_directions


class Scene:
    def __init__(self, cam_fov=90, ambient_light=0.3):
        self.ambient = ambient_light
        self.objects = []
        self.light = None
        self.camera = Camera(field_of_view=cam_fov)
        self.background = np.array([25, 25, 25])

    def get_intersection(self, ray_origin, ray_direction, reflected_obj=None):

        """ For a given ray -> it returns the object the ray hits and the distance to that point """

        intersection = None
        for obj in self.objects:

            if obj == reflected_obj:
                continue

            dist = obj.intersect_ray(ray_origin, ray_direction)

            # If we get an intersection and this intersection is the closest one:
            if dist is not None and (intersection is None or dist < intersection[1]):
                intersection = obj, dist

        return intersection

    def trace(self, ray_direction, ray_origin=None, bounce=0, max_bounce=2, reflected_obj=None):

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

        color = np.array([0.0, 0.0, 0.0])       # Start with darkness

        if bounce > max_bounce:                 # If this is a reflected ray, make sure we don't reflect too many times
            return pixelize(color)

        if ray_origin is None:                  # If ray origin is not specified, it is coming from the camera
            ray_origin = self.camera.origin

        # Loop through all scene objects to check for intersections:
        intersect = self.get_intersection(ray_origin, ray_direction, reflected_obj=reflected_obj)

        if intersect is None:                   # If there is no collision  -> return
            return pixelize(color)

        obj, dist = intersect

        # Lighting and shading
        color += obj.color * self.ambient       # Ambient light

        if self.light is None:                  # If there is no light source: dont do shading
            return pixelize(color)

        # Vectors of interest:
        P = ray_origin + unit_vector(ray_direction) * dist  # Point of collision.
        L = unit_vector(self.light.origin - P)              # From intersection to light direction
        N = obj.get_normal(P)                               # Normal vector at surface
        R = unit_vector(reflect(ray_direction, N))          # Reflection direction

        # If there is a line of sight to the light source, do the lambert shading:
        if self.get_intersection(ray_origin=P, ray_direction=L, reflected_obj=obj) is None:
            lambert_intensity = np.dot(L, N)
            if lambert_intensity > 0:
                color += obj.color * obj.k_lambert * lambert_intensity

        # Reflected light:
        reflected_light = self.trace(ray_direction=R, ray_origin=P, reflected_obj=obj, bounce=bounce+1)
        color += reflected_light * obj.k_specular

        return pixelize(color)

    @timed
    def render(self, ray_directions):
        result = np.apply_along_axis(self.trace, 0, ray_directions)
        return result


def main():
    # Resolution settings:
    w = 1000
    h = 1000

    # Create scene with camera:
    scene = Scene(cam_fov=90)

    # Populate the scene:
    sphere1 = Sphere(origin=np.array([2.5, -1.5, -0.3]), radius=0.7, color=np.array([236, 33, 33]),
                     k_lambert=0.85, k_specular=0.7)

    sphere2 = Sphere(origin=np.array([4, 0, 0]), radius=1.0, color=np.array([51, 153, 255]),
                     k_lambert=0.85, k_specular=0.8)

    sphere3 = Sphere(origin=np.array([2.5, 0.8, -0.5]), radius=0.3, color=np.array([251, 251, 51]),
                     k_lambert=0.85, k_specular=0.7)

    scene.light = Light(origin=np.array([0.0, -2.5, 0.0]))

    plane = Plane(origin=np.array([5.1, 0, 0]), normal=np.array([-1, 0, 0]), k_specular=0.0, k_lambert=1.0)

    scene.objects.append(plane)
    scene.objects.append(sphere1)
    scene.objects.append(sphere2)
    scene.objects.append(sphere3)

    # Calculate the ray directions
    print('Generating rays...')
    ray_directions = scene.camera.create_rays(w, h)

    # Render the image:
    print('Rendering...')
    result = scene.render(ray_directions)

    # Save the image to a .png file:
    save_image(result)

    return 0


if __name__ == '__main__':
    main()
