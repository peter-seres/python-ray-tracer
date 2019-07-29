from PIL import Image
import numpy as np


def main():
    w = 500     # image width (pixels)
    h = 480     # image height (pixels)
    ch = 3      # number of channels (RGB)

    pixel_array = 255*np.ones(shape=(w, h, ch), dtype=np.uint8)
    image = Image.fromarray(pixel_array)
    image.save('result.png')


if __name__ == '__main__':
    main()
