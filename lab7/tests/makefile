all: test1.exe test2.exe test3.exe test4.exe test5.exe test6.exe

test1.exe: test1.o
	nvcc -o $@ $<
test1.o: test1.cu
	nvcc -c -o $@ $<

test2.exe: test2.o
	nvcc -o $@ $<
test2.o: test2.cu
	nvcc -c -o $@ $<
	
test3.exe: test3.o
	nvcc -o $@ $<
test3.o: test3.cu
	nvcc -c -o $@ $<
	
test4.exe: test4.o
	nvcc -o $@ $<
test4.o: test4.cu
	nvcc -c -o $@ $<
	
test5.exe: test5.o
	nvcc -o $@ $<
test5.o: test5.cu
	nvcc -c -o $@ $<
	
test6.exe: test6.o
	nvcc -o $@ $<
test6.o: test6.cu
	nvcc -c -o $@ $<
		
.PHONY: all clean

clean:
	rm -rf *.o *.exe

# компиляция объектов по отдельности дает некоторый выигрыш в
# производительности: не нужно компилировать не изменявшийся код
