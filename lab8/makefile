all: task1.exe 

task1.exe: task1.o
	nvcc -o $@ $< -lcublas
task1.o: task1.cu
	nvcc -c -o $@ $<
		
.PHONY: all clean

clean:
	rm -rf *.o *.exe
