import cv2 as cv
import numpy as np

img_array = np.load('filename.npy')

from matplotlib import pyplot as plt

plt.imshow(img_array, cmap='gray')
plt.show()

from PIL import Image

im = Image.fromarray(img_array)
# this might fail if `img_array` contains a data type that is not supported by PIL,
# in which case you could try casting it to a different dtype e.g.:
# im = Image.fromarray(img_array.astype(np.uint8))

im.show()