all: e1.exe e4.exe

e1.exe: e1.o
	nvcc -o $@ $< -lcublas
e1.o: e1.cu
	nvcc -c -o $@ $<
	
e4.exe: e4.o
	nvcc -o $@ $< -lcufft
e4.o: e4.cu
	nvcc -c -o $@ $<
		
.PHONY: all clean

clean:
	rm -rf *.o *.exe
