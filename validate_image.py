import numpy as np
import matplotlib.pyplot as plt

WIDTH = 320
HEIGHT = 240
FILE_PATH = "d000000.zdp"

depth_data = np.fromfile(FILE_PATH, dtype=np.uint8)

if depth_data.size != WIDTH * HEIGHT:
    print(f"Error: File size {depth_data.size} does not match W*H {WIDTH * HEIGHT}")
else:
    depth_image = depth_data.reshape((HEIGHT, WIDTH))

    plt.imshow(depth_image, cmap='gray', vmin=200, vmax=255)
    plt.title("Depth Buffer")
    plt.colorbar()
    plt.show()
