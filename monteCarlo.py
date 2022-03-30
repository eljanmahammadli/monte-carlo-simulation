"""
Monte Carlo simulation by ELJAN MAHAMMADLI
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    image = Image.open(image_path, "r")
    width, height = image.size
    thresh = 200
    def fn(x): return 255 if x > thresh else 0
    image = image.convert('L').point(fn, mode='1')
    pixel_values = list(image.getdata())
    if image.mode == "RGB":
        channels = 3
    elif image.mode == "1":
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((width, height, channels))
    return image, pixel_values


def calculate_area(N=1000):
    # load image and reshape it
    image, pixel_values = get_image('Screenshot 2022-01-17 220510.png')
    print(f"\nSHAPE OF CANVAS: {pixel_values.shape}")
    print(f"TOTAL NUMBER OF PIXELS: {image.width * image.height}\n")

    area_list = []

    for n in range(1, N+2, 500):
        # generate random (x, y) pair
        x = np.random.randint(0, image.width, n)
        y = np.random.randint(0, image.height, n)

        # number of black number pixels
        inside_pixels = 0

        # check if random pixel is black
        for i in range(n):
            if pixel_values[x[i]][y[i]][0] == 0:
                inside_pixels += 1

        # show last random points
        if n == 501:
            show_random_points(image, x, y)

        # calculate area using monte carlo
        area = image.width * image.height * (inside_pixels / n)
        area_list.append(area)
        if (n % 100 == 1) or (n == N):
            print(f"Calculated area is {int(area)} using {n} random points")

    return N, area_list


def show_random_points(image, x, y):
    print("\nPLOT OF 1000 RANDOM POINTS ON THE IMAGE")
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(
        'PLOT OF 1000 RANDOM POINTS ON THE IMAGE\nClose this window to continue')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(x, y, marker='.', color='red', s=15)
    plt.show()


def plot_result(N, area_list):
    N = range(1, N+2, 500)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"AREA OF FIGURE IS {int(area_list[-1])}")
    ax.plot(N[1:], area_list[1:], color='red')
    plt.show()
    print(f"\nAREA OF FIGURE IS {area_list[-1]}\n")


# run the model
N, area_list = calculate_area(N=100000)
plot_result(N, area_list)
