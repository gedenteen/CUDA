#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

N = 500
x= np.linspace(-1.0,1.0,N)
k=np.arange(N)

y=np.exp(2*np.pi*1j*x)+0.2*np.exp(2*np.pi*1j*x*20)+\
	0.5*np.exp(2*np.pi*1j*x*30)
yt=np.fft.fft(y)

plt.plot(x,y.imag)
plt.show()

plt.plot(k/2,np.sqrt(yt.real**2+yt.imag**2)/N)
plt.show()

for fr in range(N):
	if fr>20:
		yt[fr]=0+0j

plt.plot(k/2,np.sqrt(yt.real**2+yt.imag**2)/N)
plt.show()

zt=np.fft.ifft(yt)

plt.plot(x,zt.imag)
plt.show()
