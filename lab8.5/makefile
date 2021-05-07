all: cuda_c.exe

cuda_c.exe: cuda_c.o
	nvcc -o $@ $<
cuda_c.o: cuda_c.cu
	nvcc -c -o $@ $<		
		
.PHONY: all clean

clean:
	rm -rf *.o *.exe
