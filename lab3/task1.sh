#!/bin/bash

nvcc lab3.cu -o lab3.exe

for (( i=32; i<=1024; i+=32 ))
do
	sudo nvprof -m achieved_occupancy ./lab3.exe 1048576 $i
done
