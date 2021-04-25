all: e1.exe

e1.exe: e1.o
	nvcc -o $@ $< -lcublas
e1.o: e1.cu
	nvcc -c -o $@ $<
		
.PHONY: all clean

clean:
	rm -rf *.o *.exe

# компиляция объектов по отдельности дает некоторый выигрыш в
# производительности: не нужно компилировать не изменявшийся код
