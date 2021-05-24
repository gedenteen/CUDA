all: main.exe

main.exe: main.o test_cuda.o test_thrust.o test_cublas.o
	nvcc -o $@ $^ -lcublas

main.o: main.cu
	nvcc -c -o $@ $<	
test_cuda.o: test_cuda.cu
	nvcc -c -o $@ $<	
test_thrust.o: test_thrust.cu
	nvcc -c -o $@ $<
test_cublas.o: test_cublas.cu
	nvcc -c -o $@ $<	

.PHONY: all clean run

run:
	./main.exe

clean:
	rm -rf *.o *.exe
	
# $@ $^ - прилинковать все, что в .......... TODO
