import numpy as np
import matplotlib.pyplot as plt

WIDTH = 320
HEIGHT = 240
FILE_PATH = "d002305.zdp"

depth_data = np.fromfile(FILE_PATH, dtype=np.float16)

if depth_data.size != WIDTH * HEIGHT:
    print(f"Error: File size {depth_data.size} does not match W*H {WIDTH * HEIGHT}")
else:
    depth_image = depth_data.reshape((HEIGHT, WIDTH))

    plt.imshow(depth_image, cmap='gray', vmin=0.9, vmax=1.0)
    plt.title("Depth Buffer")
    plt.colorbar()
    plt.show()
