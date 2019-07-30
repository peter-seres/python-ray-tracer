import time
from PIL import Image
from camera import Camera
from sphere import Sphere


def main():
    # Image resolution
    w = 300     # image width (pixels)
    h = 300     # image height (pixels)

    # The scene contains a camera and a sphere:
    camera = Camera(w, h, origin=[0, -5, 0], field_of_view=45)
    sphere1 = Sphere(position=[50, -5, 0], radius=3.5, color=[255, 0, 0])
    sphere2 = Sphere(position=[50, 5, 0], radius=3.5, color=[0, 255, 0])
    sphere3 = Sphere(position=[70, 5, 0], radius=10, color=[0, 0, 255])

    object_list = [sphere1, sphere2, sphere3]

    # Generate pixels:
    start = time.time()
    pixel_array = camera.cast_rays(object_list)
    end = time.time()
    print(f'Ray tracing took {end-start} seconds with {500*480} rays and {len(object_list)} objects.')

    # Save the data to a png file:
    image = Image.fromarray(pixel_array)
    image.save('result.png')


if __name__ == '__main__':
    main()
