import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
from numba import jit as autojit

@autojit
def density(x, y):
	d=(np.cos(np.sqrt(x**2+y**2))*np.sqrt(0.1+x**2+y**2))+20
	return d

@autojit
def create_image(X,Y,min_x, max_x, min_y, max_y, image):
	height = image.shape[0]
	width = image.shape[1]
	
	pixel_size_x = (max_x - min_x) / width
	pixel_size_y = (max_y - min_y) / height

	for x in range(width):
		X[x] = min_x + x * pixel_size_x
		for y in range(height):
			Y[y] = min_y + y * pixel_size_y
			color = density(X[x], Y[y])
			image[y, x] = color

image = np.zeros((1536, 1536), dtype = np.float)
X = np.zeros(1536, dtype = np.float)
Y = np.zeros(1536, dtype = np.float)

start = timer()
create_image(X, Y, -20.0, 20.0, -20.0, 20.0, image)
dt = timer() - start

grad_d=np.gradient(image,X,Y)
grad_d_mod=np.sqrt(grad_d[0]**2+grad_d[1]**2)

imshow(image,'gray')
show()
imshow(grad_d_mod,'gray')
show()
imshow(0.8*np.exp(-5.0*grad_d_mod/np.max(grad_d_mod)),'gray')
show()
