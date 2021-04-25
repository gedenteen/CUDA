#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

N = 50
x= np.linspace(-6.28,6.28,N)
x=0.2*np.sin(3*x)
x1=np.random.rand(N)
X=x+x1

k=np.arange(N)
plt.plot(k,X)
plt.show()

xt=np.fft.fft(X)

S=np.sqrt(xt.real**2+xt.imag**2)/N
plt.plot(k[1:20]/2,S[1:20])
plt.show()
