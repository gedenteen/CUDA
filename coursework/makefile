all: main.exe

main.exe: obj/main.o obj/test_cuda.o obj/test_thrust.o obj/test_cublas.o
	nvcc -o $@ $^ -lcublas

obj/main.o: main.cu
	nvcc -c -o $@ $<	
obj/test_cuda.o: test_cuda.cu
	nvcc -c -o $@ $<	
obj/test_thrust.o: test_thrust.cu
	nvcc -c -o $@ $<
obj/test_cublas.o: test_cublas.cu
	nvcc -c -o $@ $<	

.PHONY: all clean run

run:
	./main.exe && gnuplot graphs.gpi

clean:
	rm -rf *.o *.exe
	
# $@ - автоматическая переменная, вставить файл из цели (то, что до ':')
# $< - автоматическая переменная, вставить имя первой зависимости (после ':')
# $^ - автоматическая переменная, вставить все зависимости
