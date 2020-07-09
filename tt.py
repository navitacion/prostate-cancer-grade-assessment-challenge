import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = glob.glob('./data/grid_256_level_1/img/*.jpg')

t = []

for i in range(5):
    _img = cv2.imread(img_path[i])
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    _img = 255 - _img
    s = _img.sum()
    t.append(s)
    print(s)

    plt.imshow(_img)
    plt.title(s)
    plt.show()

t = np.argsort(np.array(t))[::-1]
print(t)

import random
random.shuffle(t)
print(t)