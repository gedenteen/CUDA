all: task1.exe task2.exe

task1.exe: task1.o
	nvcc -o $@ $< -lcublas
task1.o: task1.cu
	nvcc -c -o $@ $<
		
task2.exe: task2.o
	nvcc -o $@ $< -lcufft
task2.o: task2.cu
	nvcc -c -o $@ $<		
		
.PHONY: all clean

clean:
	rm -rf *.o *.exe
