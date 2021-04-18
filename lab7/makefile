all: lab7.exe

lab7.exe: lab7.o
	nvcc -o $@ $<
lab7.o: lab7.cu
	nvcc -c -o $@ $<
		
.PHONY: all clean

clean:
	rm -rf *.o *.exe
