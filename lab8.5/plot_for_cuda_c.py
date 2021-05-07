import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
from numba import jit as autojit #from numba import autojit

#image = np.zeros((1024, 1536), dtype = np.uint8) #инициализация массива
image = []
with open("rez.dat") as f:
    for line in f:
        image.append([int(x) for x in line.split()])
imshow(image)
show()
