#!/bin/bash

nvcc lab3.cu -o lab3.exe

for (( i=1024; i<=2**25; i*=2 )) # ** это возведение в степень
do
	sudo nvprof -m achieved_occupancy ./lab3.exe $i 128
done
